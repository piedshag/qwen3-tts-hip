use std::ffi::c_void;
use std::path::PathBuf;

use qwen3_hip_runtime::kernels::{
    ATTENTION_F32_SOURCE, ELEMENTWISE_F32_SOURCE, LAYOUT_F32_SOURCE, RMSNORM_F32_SOURCE,
    ROPE_BHSD_F32_SOURCE, SOFTMAX_F32_SOURCE,
};
use qwen3_hip_runtime::weights::{TensorArchive, tensor_to_f32};
use qwen3_hip_runtime::{DeviceBuffer, Error, HipFunction, HipModule, HipRuntime, RocblasHandle};
use safetensors::SafeTensors;

#[derive(Clone, Copy)]
struct Dims {
    batch: usize,
    steps: usize,
    rows: usize,
    hidden: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    q_out: usize,
    kv_out: usize,
    intermediate: usize,
    offset: usize,
    theta: f32,
    epsilon: f32,
    scale: f32,
}

struct HostLayer {
    input_gamma: Vec<f32>,
    q_gamma: Vec<f32>,
    k_gamma: Vec<f32>,
    q_weight: Vec<f32>,
    k_weight: Vec<f32>,
    v_weight: Vec<f32>,
    o_weight: Vec<f32>,
    post_attention_gamma: Vec<f32>,
    gate_weight: Vec<f32>,
    up_weight: Vec<f32>,
    down_weight: Vec<f32>,
}

struct DeviceLayer {
    input_gamma: DeviceBuffer<f32>,
    q_gamma: DeviceBuffer<f32>,
    k_gamma: DeviceBuffer<f32>,
    q_weight: DeviceBuffer<f32>,
    k_weight: DeviceBuffer<f32>,
    v_weight: DeviceBuffer<f32>,
    o_weight: DeviceBuffer<f32>,
    post_attention_gamma: DeviceBuffer<f32>,
    gate_weight: DeviceBuffer<f32>,
    up_weight: DeviceBuffer<f32>,
    down_weight: DeviceBuffer<f32>,
}

struct Kernels {
    _rms_module: HipModule,
    _rope_module: HipModule,
    _layout_module: HipModule,
    _softmax_module: HipModule,
    _attention_module: HipModule,
    _elementwise_module: HipModule,
    rmsnorm: HipFunction,
    rope: HipFunction,
    permute: HipFunction,
    softmax: HipFunction,
    scores: HipFunction,
    apply_value: HipFunction,
    residual_add: HipFunction,
    swiglu: HipFunction,
}

struct Workspace {
    normed: DeviceBuffer<f32>,
    q_proj: DeviceBuffer<f32>,
    k_proj: DeviceBuffer<f32>,
    v_proj: DeviceBuffer<f32>,
    q_norm: DeviceBuffer<f32>,
    k_norm: DeviceBuffer<f32>,
    q_bhsd: DeviceBuffer<f32>,
    k_bhsd: DeviceBuffer<f32>,
    v_bhsd: DeviceBuffer<f32>,
    q_rope: DeviceBuffer<f32>,
    k_rope: DeviceBuffer<f32>,
    scores: DeviceBuffer<f32>,
    probs: DeviceBuffer<f32>,
    attended: DeviceBuffer<f32>,
    projected: DeviceBuffer<f32>,
    attention_output: DeviceBuffer<f32>,
    post_norm: DeviceBuffer<f32>,
    gate: DeviceBuffer<f32>,
    up: DeviceBuffer<f32>,
    swiglu: DeviceBuffer<f32>,
    mlp_down: DeviceBuffer<f32>,
}

fn main() -> qwen3_hip_runtime::Result<()> {
    let mut args = std::env::args_os().skip(1);
    let model_dir = args.next().map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("/home/flynn/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5")
    });
    let layer_count = args
        .next()
        .map(|value| {
            value
                .to_string_lossy()
                .parse::<usize>()
                .map_err(|err| Error::InvalidInput(format!("invalid layer count: {err}")))
        })
        .transpose()?
        .unwrap_or(4);
    if layer_count == 0 {
        return Err(Error::InvalidInput(
            "layer count must be non-zero".to_string(),
        ));
    }

    let archive = TensorArchive::open(&model_dir.join("model.safetensors"))?;
    let dims = archive.with_tensors(|tensors| infer_dims(&load_host_layer(tensors, 0, None)?))?;

    let hidden_input = deterministic_hidden(dims.rows * dims.hidden);
    let mut expected = hidden_input.clone();

    let runtime = HipRuntime::new(0)?;
    let blas = runtime.create_blas_handle()?;
    let kernels = compile_kernels(&runtime)?;
    let workspace = Workspace::new(&runtime, dims)?;

    let mut current = runtime.buffer_from_slice(&hidden_input)?;
    let mut next = runtime.empty_buffer::<f32>(dims.rows * dims.hidden)?;
    archive.with_tensors(|tensors| {
        for index in 0..layer_count {
            let host_layer = load_host_layer(tensors, index, Some(dims))?;
            expected = layer_reference(&expected, &host_layer, dims);
            let device_layer = load_device_layer(&runtime, &host_layer)?;
            run_layer_gpu(
                &current,
                &next,
                &device_layer,
                &workspace,
                &kernels,
                &blas,
                dims,
            )?;
            runtime.synchronize()?;
            std::mem::swap(&mut current, &mut next);
        }
        Ok(())
    })?;

    let actual = current.copy_to_host()?;
    let max_abs = max_abs(&actual, &expected);
    let mean_abs = mean_abs(&actual, &expected);
    if max_abs > 5e-4 || mean_abs > 5e-5 {
        return Err(Error::InvalidInput(format!(
            "decoder stack mismatch: layers={layer_count}, max_abs={max_abs}, mean_abs={mean_abs}"
        )));
    }

    println!(
        "Decoder stack smoke OK: layers={layer_count}, batch={}, steps={}, hidden={}, q_heads={}, kv_heads={}, head_dim={}, max_abs={max_abs}, mean_abs={mean_abs}, first8={:?}",
        dims.batch,
        dims.steps,
        dims.hidden,
        dims.q_heads,
        dims.kv_heads,
        dims.head_dim,
        &actual[..8]
    );
    Ok(())
}

impl Workspace {
    fn new(runtime: &HipRuntime, dims: Dims) -> qwen3_hip_runtime::Result<Self> {
        Ok(Self {
            normed: runtime.empty_buffer::<f32>(dims.rows * dims.hidden)?,
            q_proj: runtime.empty_buffer::<f32>(dims.rows * dims.q_out)?,
            k_proj: runtime.empty_buffer::<f32>(dims.rows * dims.kv_out)?,
            v_proj: runtime.empty_buffer::<f32>(dims.rows * dims.kv_out)?,
            q_norm: runtime.empty_buffer::<f32>(dims.rows * dims.q_out)?,
            k_norm: runtime.empty_buffer::<f32>(dims.rows * dims.kv_out)?,
            q_bhsd: runtime
                .empty_buffer::<f32>(dims.batch * dims.q_heads * dims.steps * dims.head_dim)?,
            k_bhsd: runtime
                .empty_buffer::<f32>(dims.batch * dims.kv_heads * dims.steps * dims.head_dim)?,
            v_bhsd: runtime
                .empty_buffer::<f32>(dims.batch * dims.kv_heads * dims.steps * dims.head_dim)?,
            q_rope: runtime
                .empty_buffer::<f32>(dims.batch * dims.q_heads * dims.steps * dims.head_dim)?,
            k_rope: runtime
                .empty_buffer::<f32>(dims.batch * dims.kv_heads * dims.steps * dims.head_dim)?,
            scores: runtime
                .empty_buffer::<f32>(dims.batch * dims.q_heads * dims.steps * dims.steps)?,
            probs: runtime
                .empty_buffer::<f32>(dims.batch * dims.q_heads * dims.steps * dims.steps)?,
            attended: runtime.empty_buffer::<f32>(dims.rows * dims.q_heads * dims.head_dim)?,
            projected: runtime.empty_buffer::<f32>(dims.rows * dims.hidden)?,
            attention_output: runtime.empty_buffer::<f32>(dims.rows * dims.hidden)?,
            post_norm: runtime.empty_buffer::<f32>(dims.rows * dims.hidden)?,
            gate: runtime.empty_buffer::<f32>(dims.rows * dims.intermediate)?,
            up: runtime.empty_buffer::<f32>(dims.rows * dims.intermediate)?,
            swiglu: runtime.empty_buffer::<f32>(dims.rows * dims.intermediate)?,
            mlp_down: runtime.empty_buffer::<f32>(dims.rows * dims.hidden)?,
        })
    }
}

fn compile_kernels(runtime: &HipRuntime) -> qwen3_hip_runtime::Result<Kernels> {
    let rms_module = runtime.compile_module("rmsnorm_f32.cpp", RMSNORM_F32_SOURCE)?;
    let rope_module = runtime.compile_module("rope_bhsd_f32.cpp", ROPE_BHSD_F32_SOURCE)?;
    let layout_module = runtime.compile_module("layout_f32.cpp", LAYOUT_F32_SOURCE)?;
    let softmax_module = runtime.compile_module("softmax_f32.cpp", SOFTMAX_F32_SOURCE)?;
    let attention_module = runtime.compile_module("attention_f32.cpp", ATTENTION_F32_SOURCE)?;
    let elementwise_module =
        runtime.compile_module("elementwise_f32.cpp", ELEMENTWISE_F32_SOURCE)?;
    let rmsnorm = rms_module.function("rmsnorm_f32")?;
    let rope = rope_module.function("rope_bhsd_f32")?;
    let permute = layout_module.function("permute_bshd_to_bhsd_f32")?;
    let softmax = softmax_module.function("masked_softmax_f32")?;
    let scores = attention_module.function("attention_scores_causal_f32")?;
    let apply_value = attention_module.function("attention_apply_value_f32")?;
    let residual_add = elementwise_module.function("residual_add_f32")?;
    let swiglu = elementwise_module.function("swiglu_f32")?;
    Ok(Kernels {
        _rms_module: rms_module,
        _rope_module: rope_module,
        _layout_module: layout_module,
        _softmax_module: softmax_module,
        _attention_module: attention_module,
        _elementwise_module: elementwise_module,
        rmsnorm,
        rope,
        permute,
        softmax,
        scores,
        apply_value,
        residual_add,
        swiglu,
    })
}

fn load_host_layer(
    tensors: &SafeTensors<'_>,
    index: usize,
    expected: Option<Dims>,
) -> qwen3_hip_runtime::Result<HostLayer> {
    let prefix = format!("talker.model.layers.{index}");
    let layer = HostLayer {
        input_gamma: vector_f32(tensors, &format!("{prefix}.input_layernorm.weight"))?,
        q_gamma: vector_f32(tensors, &format!("{prefix}.self_attn.q_norm.weight"))?,
        k_gamma: vector_f32(tensors, &format!("{prefix}.self_attn.k_norm.weight"))?,
        q_weight: linear_weight_transposed_f32(
            tensors,
            &format!("{prefix}.self_attn.q_proj.weight"),
        )?,
        k_weight: linear_weight_transposed_f32(
            tensors,
            &format!("{prefix}.self_attn.k_proj.weight"),
        )?,
        v_weight: linear_weight_transposed_f32(
            tensors,
            &format!("{prefix}.self_attn.v_proj.weight"),
        )?,
        o_weight: linear_weight_transposed_f32(
            tensors,
            &format!("{prefix}.self_attn.o_proj.weight"),
        )?,
        post_attention_gamma: vector_f32(
            tensors,
            &format!("{prefix}.post_attention_layernorm.weight"),
        )?,
        gate_weight: linear_weight_transposed_f32(
            tensors,
            &format!("{prefix}.mlp.gate_proj.weight"),
        )?,
        up_weight: linear_weight_transposed_f32(tensors, &format!("{prefix}.mlp.up_proj.weight"))?,
        down_weight: linear_weight_transposed_f32(
            tensors,
            &format!("{prefix}.mlp.down_proj.weight"),
        )?,
    };
    let dims = infer_dims(&layer)?;
    if let Some(expected) = expected {
        if dims.hidden != expected.hidden
            || dims.q_heads != expected.q_heads
            || dims.kv_heads != expected.kv_heads
            || dims.head_dim != expected.head_dim
            || dims.intermediate != expected.intermediate
        {
            return Err(Error::InvalidInput(format!(
                "layer {index} dims mismatch: hidden={}, q_heads={}, kv_heads={}, head_dim={}, intermediate={}",
                dims.hidden, dims.q_heads, dims.kv_heads, dims.head_dim, dims.intermediate
            )));
        }
    }
    Ok(layer)
}

fn vector_f32(tensors: &SafeTensors<'_>, name: &str) -> qwen3_hip_runtime::Result<Vec<f32>> {
    let tensor = tensors
        .tensor(name)
        .map_err(|err| Error::InvalidInput(format!("failed to load {name}: {err}")))?;
    let shape = tensor.shape();
    if shape.len() != 1 {
        return Err(Error::InvalidInput(format!(
            "{name} rank {}, expected 1",
            shape.len()
        )));
    }
    tensor_to_f32(name, tensor.dtype(), tensor.data(), shape[0])
}

fn linear_weight_transposed_f32(
    tensors: &SafeTensors<'_>,
    name: &str,
) -> qwen3_hip_runtime::Result<Vec<f32>> {
    let tensor = tensors
        .tensor(name)
        .map_err(|err| Error::InvalidInput(format!("failed to load {name}: {err}")))?;
    let shape = tensor.shape();
    if shape.len() != 2 {
        return Err(Error::InvalidInput(format!(
            "{name} rank {}, expected 2",
            shape.len()
        )));
    }
    let out_dim = shape[0];
    let in_dim = shape[1];
    let data = tensor_to_f32(name, tensor.dtype(), tensor.data(), out_dim * in_dim)?;
    let mut transposed = vec![0.0; in_dim * out_dim];
    for out_idx in 0..out_dim {
        for in_idx in 0..in_dim {
            transposed[in_idx * out_dim + out_idx] = data[out_idx * in_dim + in_idx];
        }
    }
    Ok(transposed)
}

fn infer_dims(layer: &HostLayer) -> qwen3_hip_runtime::Result<Dims> {
    let batch = 1;
    let steps = 2;
    let rows = batch * steps;
    let hidden = layer.input_gamma.len();
    let head_dim = layer.q_gamma.len();
    let q_out = layer.q_weight.len() / hidden;
    let kv_out = layer.k_weight.len() / hidden;
    let intermediate = layer.gate_weight.len() / hidden;
    if head_dim == 0
        || q_out % head_dim != 0
        || kv_out % head_dim != 0
        || layer.k_gamma.len() != head_dim
        || layer.v_weight.len() != hidden * kv_out
        || layer.o_weight.len() != q_out * hidden
        || layer.post_attention_gamma.len() != hidden
        || layer.up_weight.len() != hidden * intermediate
        || layer.down_weight.len() != intermediate * hidden
    {
        return Err(Error::InvalidInput("invalid layer dimensions".to_string()));
    }
    Ok(Dims {
        batch,
        steps,
        rows,
        hidden,
        q_heads: q_out / head_dim,
        kv_heads: kv_out / head_dim,
        head_dim,
        q_out,
        kv_out,
        intermediate,
        offset: 0,
        theta: 10_000.0,
        epsilon: 1e-6,
        scale: (head_dim as f32).sqrt().recip(),
    })
}

fn load_device_layer(
    runtime: &HipRuntime,
    layer: &HostLayer,
) -> qwen3_hip_runtime::Result<DeviceLayer> {
    Ok(DeviceLayer {
        input_gamma: runtime.buffer_from_slice(&layer.input_gamma)?,
        q_gamma: runtime.buffer_from_slice(&layer.q_gamma)?,
        k_gamma: runtime.buffer_from_slice(&layer.k_gamma)?,
        q_weight: runtime.buffer_from_slice(&layer.q_weight)?,
        k_weight: runtime.buffer_from_slice(&layer.k_weight)?,
        v_weight: runtime.buffer_from_slice(&layer.v_weight)?,
        o_weight: runtime.buffer_from_slice(&layer.o_weight)?,
        post_attention_gamma: runtime.buffer_from_slice(&layer.post_attention_gamma)?,
        gate_weight: runtime.buffer_from_slice(&layer.gate_weight)?,
        up_weight: runtime.buffer_from_slice(&layer.up_weight)?,
        down_weight: runtime.buffer_from_slice(&layer.down_weight)?,
    })
}

fn run_layer_gpu(
    input: &DeviceBuffer<f32>,
    output: &DeviceBuffer<f32>,
    layer: &DeviceLayer,
    workspace: &Workspace,
    kernels: &Kernels,
    blas: &RocblasHandle,
    dims: Dims,
) -> qwen3_hip_runtime::Result<()> {
    launch_rmsnorm(
        &kernels.rmsnorm,
        input.as_ptr(),
        layer.input_gamma.as_ptr(),
        workspace.normed.as_mut_ptr(),
        dims.rows,
        dims.hidden,
        dims.epsilon,
    )?;
    blas.sgemm_row_major(
        &workspace.normed,
        &layer.q_weight,
        &workspace.q_proj,
        dims.rows,
        dims.q_out,
        dims.hidden,
    )?;
    blas.sgemm_row_major(
        &workspace.normed,
        &layer.k_weight,
        &workspace.k_proj,
        dims.rows,
        dims.kv_out,
        dims.hidden,
    )?;
    blas.sgemm_row_major(
        &workspace.normed,
        &layer.v_weight,
        &workspace.v_proj,
        dims.rows,
        dims.kv_out,
        dims.hidden,
    )?;
    launch_rmsnorm(
        &kernels.rmsnorm,
        workspace.q_proj.as_ptr(),
        layer.q_gamma.as_ptr(),
        workspace.q_norm.as_mut_ptr(),
        dims.rows * dims.q_heads,
        dims.head_dim,
        dims.epsilon,
    )?;
    launch_rmsnorm(
        &kernels.rmsnorm,
        workspace.k_proj.as_ptr(),
        layer.k_gamma.as_ptr(),
        workspace.k_norm.as_mut_ptr(),
        dims.rows * dims.kv_heads,
        dims.head_dim,
        dims.epsilon,
    )?;
    launch_permute(
        &kernels.permute,
        workspace.q_norm.as_ptr(),
        workspace.q_bhsd.as_mut_ptr(),
        dims.batch,
        dims.steps,
        dims.q_heads,
        dims.head_dim,
    )?;
    launch_permute(
        &kernels.permute,
        workspace.k_norm.as_ptr(),
        workspace.k_bhsd.as_mut_ptr(),
        dims.batch,
        dims.steps,
        dims.kv_heads,
        dims.head_dim,
    )?;
    launch_permute(
        &kernels.permute,
        workspace.v_proj.as_ptr(),
        workspace.v_bhsd.as_mut_ptr(),
        dims.batch,
        dims.steps,
        dims.kv_heads,
        dims.head_dim,
    )?;
    launch_rope(
        &kernels.rope,
        workspace.q_bhsd.as_ptr(),
        workspace.q_rope.as_mut_ptr(),
        dims.batch * dims.q_heads * dims.steps * dims.head_dim,
        dims.q_heads,
        dims.steps,
        dims.head_dim,
        dims.offset,
        dims.theta,
    )?;
    launch_rope(
        &kernels.rope,
        workspace.k_bhsd.as_ptr(),
        workspace.k_rope.as_mut_ptr(),
        dims.batch * dims.kv_heads * dims.steps * dims.head_dim,
        dims.kv_heads,
        dims.steps,
        dims.head_dim,
        dims.offset,
        dims.theta,
    )?;
    launch_attention_scores(
        &kernels.scores,
        workspace.q_rope.as_ptr(),
        workspace.k_rope.as_ptr(),
        workspace.scores.as_mut_ptr(),
        dims,
    )?;
    launch_softmax(
        &kernels.softmax,
        workspace.scores.as_ptr(),
        workspace.probs.as_mut_ptr(),
        dims.batch * dims.q_heads * dims.steps,
        dims.steps,
    )?;
    launch_apply_value(
        &kernels.apply_value,
        workspace.probs.as_ptr(),
        workspace.v_bhsd.as_ptr(),
        workspace.attended.as_mut_ptr(),
        dims,
    )?;
    blas.sgemm_row_major(
        &workspace.attended,
        &layer.o_weight,
        &workspace.projected,
        dims.rows,
        dims.hidden,
        dims.q_out,
    )?;
    launch_ternary(
        &kernels.residual_add,
        input.as_ptr(),
        workspace.projected.as_ptr(),
        workspace.attention_output.as_mut_ptr(),
        dims.rows * dims.hidden,
    )?;
    launch_rmsnorm(
        &kernels.rmsnorm,
        workspace.attention_output.as_ptr(),
        layer.post_attention_gamma.as_ptr(),
        workspace.post_norm.as_mut_ptr(),
        dims.rows,
        dims.hidden,
        dims.epsilon,
    )?;
    blas.sgemm_row_major(
        &workspace.post_norm,
        &layer.gate_weight,
        &workspace.gate,
        dims.rows,
        dims.intermediate,
        dims.hidden,
    )?;
    blas.sgemm_row_major(
        &workspace.post_norm,
        &layer.up_weight,
        &workspace.up,
        dims.rows,
        dims.intermediate,
        dims.hidden,
    )?;
    launch_ternary(
        &kernels.swiglu,
        workspace.gate.as_ptr(),
        workspace.up.as_ptr(),
        workspace.swiglu.as_mut_ptr(),
        dims.rows * dims.intermediate,
    )?;
    blas.sgemm_row_major(
        &workspace.swiglu,
        &layer.down_weight,
        &workspace.mlp_down,
        dims.rows,
        dims.hidden,
        dims.intermediate,
    )?;
    launch_ternary(
        &kernels.residual_add,
        workspace.attention_output.as_ptr(),
        workspace.mlp_down.as_ptr(),
        output.as_mut_ptr(),
        dims.rows * dims.hidden,
    )
}

fn launch_rmsnorm(
    function: &HipFunction,
    input: *const c_void,
    gamma: *const c_void,
    output: *mut c_void,
    rows: usize,
    cols: usize,
    epsilon: f32,
) -> qwen3_hip_runtime::Result<()> {
    let mut input = input;
    let mut gamma = gamma;
    let mut output = output;
    let mut rows_i32 = rows as i32;
    let mut cols_i32 = cols as i32;
    let mut epsilon = epsilon;
    let mut params = [
        &mut input as *mut *const c_void as *mut c_void,
        &mut gamma as *mut *const c_void as *mut c_void,
        &mut output as *mut *mut c_void as *mut c_void,
        &mut rows_i32 as *mut i32 as *mut c_void,
        &mut cols_i32 as *mut i32 as *mut c_void,
        &mut epsilon as *mut f32 as *mut c_void,
    ];
    let block = 256u32;
    function.launch((rows as u32, 1, 1), (block, 1, 1), block * 4, &mut params)
}

fn launch_permute(
    function: &HipFunction,
    input: *const c_void,
    output: *mut c_void,
    batch: usize,
    steps: usize,
    heads: usize,
    head_dim: usize,
) -> qwen3_hip_runtime::Result<()> {
    let total = batch * steps * heads * head_dim;
    let mut input = input;
    let mut output = output;
    let mut batch_i32 = batch as i32;
    let mut steps_i32 = steps as i32;
    let mut heads_i32 = heads as i32;
    let mut head_dim_i32 = head_dim as i32;
    let mut total_i32 = total as i32;
    let mut params = [
        &mut input as *mut *const c_void as *mut c_void,
        &mut output as *mut *mut c_void as *mut c_void,
        &mut batch_i32 as *mut i32 as *mut c_void,
        &mut steps_i32 as *mut i32 as *mut c_void,
        &mut heads_i32 as *mut i32 as *mut c_void,
        &mut head_dim_i32 as *mut i32 as *mut c_void,
        &mut total_i32 as *mut i32 as *mut c_void,
    ];
    let block = 256u32;
    function.launch(
        ((total as u32).div_ceil(block), 1, 1),
        (block, 1, 1),
        0,
        &mut params,
    )
}

fn launch_rope(
    function: &HipFunction,
    input: *const c_void,
    output: *mut c_void,
    total: usize,
    heads: usize,
    steps: usize,
    head_dim: usize,
    offset: usize,
    theta: f32,
) -> qwen3_hip_runtime::Result<()> {
    let mut input = input;
    let mut output = output;
    let mut total_i32 = total as i32;
    let mut heads_i32 = heads as i32;
    let mut steps_i32 = steps as i32;
    let mut head_dim_i32 = head_dim as i32;
    let mut offset_i32 = offset as i32;
    let mut theta = theta;
    let mut params = [
        &mut input as *mut *const c_void as *mut c_void,
        &mut output as *mut *mut c_void as *mut c_void,
        &mut total_i32 as *mut i32 as *mut c_void,
        &mut heads_i32 as *mut i32 as *mut c_void,
        &mut steps_i32 as *mut i32 as *mut c_void,
        &mut head_dim_i32 as *mut i32 as *mut c_void,
        &mut offset_i32 as *mut i32 as *mut c_void,
        &mut theta as *mut f32 as *mut c_void,
    ];
    let block = 256u32;
    function.launch(
        ((total as u32).div_ceil(block), 1, 1),
        (block, 1, 1),
        0,
        &mut params,
    )
}

fn launch_attention_scores(
    function: &HipFunction,
    q: *const c_void,
    k: *const c_void,
    scores: *mut c_void,
    dims: Dims,
) -> qwen3_hip_runtime::Result<()> {
    let total = dims.batch * dims.q_heads * dims.steps * dims.steps;
    let mut q = q;
    let mut k = k;
    let mut scores = scores;
    let mut batch = dims.batch as i32;
    let mut q_heads = dims.q_heads as i32;
    let mut kv_heads = dims.kv_heads as i32;
    let mut query_steps = dims.steps as i32;
    let mut key_steps = dims.steps as i32;
    let mut head_dim = dims.head_dim as i32;
    let mut offset = dims.offset as i32;
    let mut scale = dims.scale;
    let mut total_i32 = total as i32;
    let mut params = [
        &mut q as *mut *const c_void as *mut c_void,
        &mut k as *mut *const c_void as *mut c_void,
        &mut scores as *mut *mut c_void as *mut c_void,
        &mut batch as *mut i32 as *mut c_void,
        &mut q_heads as *mut i32 as *mut c_void,
        &mut kv_heads as *mut i32 as *mut c_void,
        &mut query_steps as *mut i32 as *mut c_void,
        &mut key_steps as *mut i32 as *mut c_void,
        &mut head_dim as *mut i32 as *mut c_void,
        &mut offset as *mut i32 as *mut c_void,
        &mut scale as *mut f32 as *mut c_void,
        &mut total_i32 as *mut i32 as *mut c_void,
    ];
    let block = 256u32;
    function.launch(
        ((total as u32).div_ceil(block), 1, 1),
        (block, 1, 1),
        0,
        &mut params,
    )
}

fn launch_softmax(
    function: &HipFunction,
    input: *const c_void,
    output: *mut c_void,
    rows: usize,
    cols: usize,
) -> qwen3_hip_runtime::Result<()> {
    let mut input = input;
    let mut output = output;
    let mut rows_i32 = rows as i32;
    let mut cols_i32 = cols as i32;
    let mut active_cols = cols as i32;
    let mut params = [
        &mut input as *mut *const c_void as *mut c_void,
        &mut output as *mut *mut c_void as *mut c_void,
        &mut rows_i32 as *mut i32 as *mut c_void,
        &mut cols_i32 as *mut i32 as *mut c_void,
        &mut active_cols as *mut i32 as *mut c_void,
    ];
    let block = 256u32;
    function.launch((rows as u32, 1, 1), (block, 1, 1), block * 4, &mut params)
}

fn launch_apply_value(
    function: &HipFunction,
    probs: *const c_void,
    v: *const c_void,
    output: *mut c_void,
    dims: Dims,
) -> qwen3_hip_runtime::Result<()> {
    let total = dims.batch * dims.steps * dims.q_heads * dims.head_dim;
    let mut probs = probs;
    let mut v = v;
    let mut output = output;
    let mut batch = dims.batch as i32;
    let mut q_heads = dims.q_heads as i32;
    let mut kv_heads = dims.kv_heads as i32;
    let mut query_steps = dims.steps as i32;
    let mut key_steps = dims.steps as i32;
    let mut head_dim = dims.head_dim as i32;
    let mut total_i32 = total as i32;
    let mut params = [
        &mut probs as *mut *const c_void as *mut c_void,
        &mut v as *mut *const c_void as *mut c_void,
        &mut output as *mut *mut c_void as *mut c_void,
        &mut batch as *mut i32 as *mut c_void,
        &mut q_heads as *mut i32 as *mut c_void,
        &mut kv_heads as *mut i32 as *mut c_void,
        &mut query_steps as *mut i32 as *mut c_void,
        &mut key_steps as *mut i32 as *mut c_void,
        &mut head_dim as *mut i32 as *mut c_void,
        &mut total_i32 as *mut i32 as *mut c_void,
    ];
    let block = 256u32;
    function.launch(
        ((total as u32).div_ceil(block), 1, 1),
        (block, 1, 1),
        0,
        &mut params,
    )
}

fn launch_ternary(
    function: &HipFunction,
    input_a: *const c_void,
    input_b: *const c_void,
    output: *mut c_void,
    total: usize,
) -> qwen3_hip_runtime::Result<()> {
    let mut input_a = input_a;
    let mut input_b = input_b;
    let mut output = output;
    let mut total_i32 = total as i32;
    let mut params = [
        &mut input_a as *mut *const c_void as *mut c_void,
        &mut input_b as *mut *const c_void as *mut c_void,
        &mut output as *mut *mut c_void as *mut c_void,
        &mut total_i32 as *mut i32 as *mut c_void,
    ];
    let block = 256u32;
    function.launch(
        ((total as u32).div_ceil(block), 1, 1),
        (block, 1, 1),
        0,
        &mut params,
    )
}

fn layer_reference(input: &[f32], layer: &HostLayer, dims: Dims) -> Vec<f32> {
    let normed = rmsnorm_reference(
        input,
        &layer.input_gamma,
        dims.rows,
        dims.hidden,
        dims.epsilon,
    );
    let q = row_major_matmul(&normed, &layer.q_weight, dims.rows, dims.q_out, dims.hidden);
    let k = row_major_matmul(
        &normed,
        &layer.k_weight,
        dims.rows,
        dims.kv_out,
        dims.hidden,
    );
    let v = row_major_matmul(
        &normed,
        &layer.v_weight,
        dims.rows,
        dims.kv_out,
        dims.hidden,
    );
    let q = rmsnorm_reference(
        &q,
        &layer.q_gamma,
        dims.rows * dims.q_heads,
        dims.head_dim,
        dims.epsilon,
    );
    let k = rmsnorm_reference(
        &k,
        &layer.k_gamma,
        dims.rows * dims.kv_heads,
        dims.head_dim,
        dims.epsilon,
    );
    let q = rope_reference(
        &permute_bshd_to_bhsd(&q, dims.batch, dims.steps, dims.q_heads, dims.head_dim),
        dims,
    );
    let k = rope_reference(
        &permute_bshd_to_bhsd(&k, dims.batch, dims.steps, dims.kv_heads, dims.head_dim),
        dims,
    );
    let v = permute_bshd_to_bhsd(&v, dims.batch, dims.steps, dims.kv_heads, dims.head_dim);
    let scores = attention_scores_reference(&q, &k, dims);
    let probs = softmax_rows(&scores, dims.batch * dims.q_heads * dims.steps, dims.steps);
    let attended = attention_apply_value_reference(&probs, &v, dims);
    let projected = row_major_matmul(
        &attended,
        &layer.o_weight,
        dims.rows,
        dims.hidden,
        dims.q_out,
    );
    let attention_output = input
        .iter()
        .zip(projected.iter())
        .map(|(input, projected)| input + projected)
        .collect::<Vec<_>>();
    let post_norm = rmsnorm_reference(
        &attention_output,
        &layer.post_attention_gamma,
        dims.rows,
        dims.hidden,
        dims.epsilon,
    );
    let gate = row_major_matmul(
        &post_norm,
        &layer.gate_weight,
        dims.rows,
        dims.intermediate,
        dims.hidden,
    );
    let up = row_major_matmul(
        &post_norm,
        &layer.up_weight,
        dims.rows,
        dims.intermediate,
        dims.hidden,
    );
    let swiglu = gate
        .iter()
        .zip(up.iter())
        .map(|(gate, up)| gate / (1.0 + (-gate).exp()) * up)
        .collect::<Vec<_>>();
    let down = row_major_matmul(
        &swiglu,
        &layer.down_weight,
        dims.rows,
        dims.hidden,
        dims.intermediate,
    );
    attention_output
        .iter()
        .zip(down.iter())
        .map(|(hidden, down)| hidden + down)
        .collect()
}

fn deterministic_hidden(len: usize) -> Vec<f32> {
    (0..len)
        .map(|idx| ((idx % 31) as f32 - 15.0) / 17.0)
        .collect()
}

fn row_major_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let mut c = vec![0.0; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0;
            for inner in 0..k {
                sum += a[row * k + inner] * b[inner * n + col];
            }
            c[row * n + col] = sum;
        }
    }
    c
}

fn rmsnorm_reference(
    input: &[f32],
    gamma: &[f32],
    rows: usize,
    cols: usize,
    epsilon: f32,
) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    for row in 0..rows {
        let offset = row * cols;
        let sum = input[offset..offset + cols]
            .iter()
            .map(|value| value * value)
            .sum::<f32>();
        let inv_rms = (sum / cols as f32 + epsilon).sqrt().recip();
        for col in 0..cols {
            output[offset + col] = input[offset + col] * inv_rms * gamma[col];
        }
    }
    output
}

fn permute_bshd_to_bhsd(
    input: &[f32],
    batch: usize,
    steps: usize,
    heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    for batch_index in 0..batch {
        for step in 0..steps {
            for head in 0..heads {
                for dim in 0..head_dim {
                    let input_idx =
                        (((batch_index * steps + step) * heads + head) * head_dim) + dim;
                    let output_idx =
                        (((batch_index * heads + head) * steps + step) * head_dim) + dim;
                    output[output_idx] = input[input_idx];
                }
            }
        }
    }
    output
}

fn rope_reference(input: &[f32], dims: Dims) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    let half = dims.head_dim / 2;
    for idx in 0..input.len() {
        let dim_index = idx % dims.head_dim;
        let step_index = (idx / dims.head_dim) % dims.steps;
        let pair_index = dim_index % half;
        let base = idx - dim_index;
        let first = input[base + pair_index];
        let second = input[base + pair_index + half];
        let exponent = (pair_index * 2) as f32 / dims.head_dim as f32;
        let angle = (dims.offset + step_index) as f32 / dims.theta.powf(exponent);
        let cos = angle.cos();
        let sin = angle.sin();
        output[idx] = if dim_index < half {
            first * cos - second * sin
        } else {
            second * cos + first * sin
        };
    }
    output
}

fn attention_scores_reference(q: &[f32], k: &[f32], dims: Dims) -> Vec<f32> {
    let mut scores = vec![0.0; dims.batch * dims.q_heads * dims.steps * dims.steps];
    let n_rep = dims.q_heads / dims.kv_heads;
    for batch_index in 0..dims.batch {
        for q_head in 0..dims.q_heads {
            let kv_head = q_head / n_rep;
            for query_step in 0..dims.steps {
                for key_step in 0..dims.steps {
                    let score_idx = (((batch_index * dims.q_heads + q_head) * dims.steps
                        + query_step)
                        * dims.steps)
                        + key_step;
                    if key_step > dims.offset + query_step {
                        scores[score_idx] = f32::NEG_INFINITY;
                        continue;
                    }
                    let q_base = (((batch_index * dims.q_heads + q_head) * dims.steps + query_step)
                        * dims.head_dim) as usize;
                    let k_base = (((batch_index * dims.kv_heads + kv_head) * dims.steps + key_step)
                        * dims.head_dim) as usize;
                    let mut sum = 0.0;
                    for dim in 0..dims.head_dim {
                        sum += q[q_base + dim] * k[k_base + dim];
                    }
                    scores[score_idx] = sum * dims.scale;
                }
            }
        }
    }
    scores
}

fn softmax_rows(input: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    for row in 0..rows {
        let offset = row * cols;
        let row_max = input[offset..offset + cols]
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let sum = input[offset..offset + cols]
            .iter()
            .map(|value| (value - row_max).exp())
            .sum::<f32>();
        for col in 0..cols {
            output[offset + col] = (input[offset + col] - row_max).exp() / sum;
        }
    }
    output
}

fn attention_apply_value_reference(probs: &[f32], v: &[f32], dims: Dims) -> Vec<f32> {
    let mut output = vec![0.0; dims.batch * dims.steps * dims.q_heads * dims.head_dim];
    let n_rep = dims.q_heads / dims.kv_heads;
    for batch_index in 0..dims.batch {
        for query_step in 0..dims.steps {
            for q_head in 0..dims.q_heads {
                let kv_head = q_head / n_rep;
                for dim in 0..dims.head_dim {
                    let mut sum = 0.0;
                    for key_step in 0..dims.steps {
                        let prob_idx = (((batch_index * dims.q_heads + q_head) * dims.steps
                            + query_step)
                            * dims.steps)
                            + key_step;
                        let v_idx = (((batch_index * dims.kv_heads + kv_head) * dims.steps
                            + key_step)
                            * dims.head_dim)
                            + dim;
                        sum += probs[prob_idx] * v[v_idx];
                    }
                    let out_idx = (((batch_index * dims.steps + query_step) * dims.q_heads
                        + q_head)
                        * dims.head_dim)
                        + dim;
                    output[out_idx] = sum;
                }
            }
        }
    }
    output
}

fn max_abs(actual: &[f32], expected: &[f32]) -> f32 {
    actual
        .iter()
        .zip(expected.iter())
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0f32, f32::max)
}

fn mean_abs(actual: &[f32], expected: &[f32]) -> f32 {
    actual
        .iter()
        .zip(expected.iter())
        .map(|(actual, expected)| (actual - expected).abs())
        .sum::<f32>()
        / actual.len() as f32
}
