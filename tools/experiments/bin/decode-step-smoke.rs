use std::ffi::c_void;
use std::path::PathBuf;

use qwen3_hip_runtime::kernels::{
    ATTENTION_F32_SOURCE, ELEMENTWISE_F32_SOURCE, LAYOUT_F32_SOURCE, RMSNORM_F32_SOURCE,
    ROPE_BHSD_F32_SOURCE, SOFTMAX_F32_SOURCE,
};
use qwen3_hip_runtime::weights::TensorArchive;
use qwen3_hip_runtime::{DeviceBuffer, Error, HipFunction, HipRuntime, RocblasHandle};

const LAYER_PREFIX: &str = "talker.model.layers.0";

struct LayerWeights {
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

#[derive(Clone, Copy)]
struct Dims {
    hidden: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    q_out: usize,
    kv_out: usize,
    intermediate: usize,
    epsilon: f32,
    theta: f32,
    scale: f32,
}

struct Kernels {
    rmsnorm: HipFunction,
    rope: HipFunction,
    permute: HipFunction,
    softmax: HipFunction,
    scores: HipFunction,
    apply_value: HipFunction,
    write_cache: HipFunction,
    residual_add: HipFunction,
    swiglu: HipFunction,
    _modules: Vec<qwen3_hip_runtime::HipModule>,
}

fn main() -> qwen3_hip_runtime::Result<()> {
    let model_dir = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from("/home/flynn/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5")
        });
    let archive = TensorArchive::open(&model_dir.join("model.safetensors"))?;
    let weights = load_weights(&archive)?;
    let dims = infer_dims(&weights)?;

    let prefix_steps = 2usize;
    let decode_steps = 1usize;
    let cache_steps = prefix_steps + decode_steps;
    let full_hidden = deterministic_hidden(cache_steps * dims.hidden);
    let prefix_hidden = full_hidden[..prefix_steps * dims.hidden].to_vec();
    let decode_hidden = full_hidden[prefix_steps * dims.hidden..].to_vec();
    let expected_full = layer_reference(&full_hidden, &weights, dims, cache_steps);
    let expected = &expected_full[prefix_steps * dims.hidden..];

    let runtime = HipRuntime::new(0)?;
    let blas = runtime.create_blas_handle()?;
    let kernels = compile_kernels(&runtime)?;
    let device = DeviceWeights::new(&runtime, &weights)?;

    let prefix_dev = runtime.buffer_from_slice(&prefix_hidden)?;
    let decode_dev = runtime.buffer_from_slice(&decode_hidden)?;
    let output_dev = runtime.empty_buffer::<f32>(dims.hidden)?;

    let normed_prefill = runtime.empty_buffer::<f32>(prefix_steps * dims.hidden)?;
    let k_proj_prefill = runtime.empty_buffer::<f32>(prefix_steps * dims.kv_out)?;
    let v_proj_prefill = runtime.empty_buffer::<f32>(prefix_steps * dims.kv_out)?;
    let k_norm_prefill = runtime.empty_buffer::<f32>(prefix_steps * dims.kv_out)?;
    let k_bhsd_prefill =
        runtime.empty_buffer::<f32>(dims.kv_heads * prefix_steps * dims.head_dim)?;
    let v_bhsd_prefill =
        runtime.empty_buffer::<f32>(dims.kv_heads * prefix_steps * dims.head_dim)?;
    let k_rope_prefill =
        runtime.empty_buffer::<f32>(dims.kv_heads * prefix_steps * dims.head_dim)?;
    let k_cache = runtime.empty_buffer::<f32>(dims.kv_heads * cache_steps * dims.head_dim)?;
    let v_cache = runtime.empty_buffer::<f32>(dims.kv_heads * cache_steps * dims.head_dim)?;

    launch_rmsnorm(
        &kernels.rmsnorm,
        prefix_dev.as_ptr(),
        device.input_gamma.as_ptr(),
        normed_prefill.as_mut_ptr(),
        prefix_steps,
        dims.hidden,
        dims.epsilon,
    )?;
    blas.sgemm_row_major(
        &normed_prefill,
        &device.k_weight,
        &k_proj_prefill,
        prefix_steps,
        dims.kv_out,
        dims.hidden,
    )?;
    blas.sgemm_row_major(
        &normed_prefill,
        &device.v_weight,
        &v_proj_prefill,
        prefix_steps,
        dims.kv_out,
        dims.hidden,
    )?;
    launch_rmsnorm(
        &kernels.rmsnorm,
        k_proj_prefill.as_ptr(),
        device.k_gamma.as_ptr(),
        k_norm_prefill.as_mut_ptr(),
        prefix_steps * dims.kv_heads,
        dims.head_dim,
        dims.epsilon,
    )?;
    launch_permute(
        &kernels.permute,
        k_norm_prefill.as_ptr(),
        k_bhsd_prefill.as_mut_ptr(),
        prefix_steps,
        dims.kv_heads,
        dims.head_dim,
    )?;
    launch_permute(
        &kernels.permute,
        v_proj_prefill.as_ptr(),
        v_bhsd_prefill.as_mut_ptr(),
        prefix_steps,
        dims.kv_heads,
        dims.head_dim,
    )?;
    launch_rope(
        &kernels.rope,
        k_bhsd_prefill.as_ptr(),
        k_rope_prefill.as_mut_ptr(),
        dims.kv_heads * prefix_steps * dims.head_dim,
        dims.kv_heads,
        prefix_steps,
        dims.head_dim,
        0,
        dims.theta,
    )?;
    launch_write_cache(
        &kernels.write_cache,
        k_rope_prefill.as_ptr(),
        k_cache.as_mut_ptr(),
        dims.kv_heads,
        prefix_steps,
        cache_steps,
        dims.head_dim,
        0,
    )?;
    launch_write_cache(
        &kernels.write_cache,
        v_bhsd_prefill.as_ptr(),
        v_cache.as_mut_ptr(),
        dims.kv_heads,
        prefix_steps,
        cache_steps,
        dims.head_dim,
        0,
    )?;

    run_decode_step(
        &runtime,
        &decode_dev,
        &output_dev,
        &device,
        &k_cache,
        &v_cache,
        &kernels,
        &blas,
        dims,
        prefix_steps,
        cache_steps,
    )?;
    runtime.synchronize()?;

    let actual = output_dev.copy_to_host()?;
    let max_abs = max_abs(&actual, expected);
    let mean_abs = mean_abs(&actual, expected);
    if max_abs > 2e-4 || mean_abs > 2e-5 {
        return Err(Error::InvalidInput(format!(
            "decode-step mismatch: max_abs={max_abs}, mean_abs={mean_abs}"
        )));
    }

    println!(
        "Decode-step smoke OK: layer={LAYER_PREFIX}, prefix_steps={prefix_steps}, cache_steps={cache_steps}, hidden={}, q_heads={}, kv_heads={}, head_dim={}, max_abs={max_abs}, mean_abs={mean_abs}, first8={:?}",
        dims.hidden,
        dims.q_heads,
        dims.kv_heads,
        dims.head_dim,
        &actual[..8]
    );
    Ok(())
}

struct DeviceWeights {
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

impl DeviceWeights {
    fn new(runtime: &HipRuntime, weights: &LayerWeights) -> qwen3_hip_runtime::Result<Self> {
        Ok(Self {
            input_gamma: runtime.buffer_from_slice(&weights.input_gamma)?,
            q_gamma: runtime.buffer_from_slice(&weights.q_gamma)?,
            k_gamma: runtime.buffer_from_slice(&weights.k_gamma)?,
            q_weight: runtime.buffer_from_slice(&weights.q_weight)?,
            k_weight: runtime.buffer_from_slice(&weights.k_weight)?,
            v_weight: runtime.buffer_from_slice(&weights.v_weight)?,
            o_weight: runtime.buffer_from_slice(&weights.o_weight)?,
            post_attention_gamma: runtime.buffer_from_slice(&weights.post_attention_gamma)?,
            gate_weight: runtime.buffer_from_slice(&weights.gate_weight)?,
            up_weight: runtime.buffer_from_slice(&weights.up_weight)?,
            down_weight: runtime.buffer_from_slice(&weights.down_weight)?,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn run_decode_step(
    runtime: &HipRuntime,
    input: &DeviceBuffer<f32>,
    output: &DeviceBuffer<f32>,
    weights: &DeviceWeights,
    k_cache: &DeviceBuffer<f32>,
    v_cache: &DeviceBuffer<f32>,
    kernels: &Kernels,
    blas: &RocblasHandle,
    dims: Dims,
    offset: usize,
    cache_steps: usize,
) -> qwen3_hip_runtime::Result<()> {
    let normed = runtime.empty_buffer::<f32>(dims.hidden)?;
    let q_proj = runtime.empty_buffer::<f32>(dims.q_out)?;
    let k_proj = runtime.empty_buffer::<f32>(dims.kv_out)?;
    let v_proj = runtime.empty_buffer::<f32>(dims.kv_out)?;
    let q_norm = runtime.empty_buffer::<f32>(dims.q_out)?;
    let k_norm = runtime.empty_buffer::<f32>(dims.kv_out)?;
    let q_bhsd = runtime.empty_buffer::<f32>(dims.q_out)?;
    let k_bhsd = runtime.empty_buffer::<f32>(dims.kv_out)?;
    let v_bhsd = runtime.empty_buffer::<f32>(dims.kv_out)?;
    let q_rope = runtime.empty_buffer::<f32>(dims.q_out)?;
    let k_rope = runtime.empty_buffer::<f32>(dims.kv_out)?;
    let scores = runtime.empty_buffer::<f32>(dims.q_heads * cache_steps)?;
    let probs = runtime.empty_buffer::<f32>(dims.q_heads * cache_steps)?;
    let attended = runtime.empty_buffer::<f32>(dims.q_out)?;
    let projected = runtime.empty_buffer::<f32>(dims.hidden)?;
    let attention_output = runtime.empty_buffer::<f32>(dims.hidden)?;
    let post_norm = runtime.empty_buffer::<f32>(dims.hidden)?;
    let gate = runtime.empty_buffer::<f32>(dims.intermediate)?;
    let up = runtime.empty_buffer::<f32>(dims.intermediate)?;
    let swiglu = runtime.empty_buffer::<f32>(dims.intermediate)?;
    let mlp_down = runtime.empty_buffer::<f32>(dims.hidden)?;

    launch_rmsnorm(
        &kernels.rmsnorm,
        input.as_ptr(),
        weights.input_gamma.as_ptr(),
        normed.as_mut_ptr(),
        1,
        dims.hidden,
        dims.epsilon,
    )?;
    blas.sgemm_row_major(
        &normed,
        &weights.q_weight,
        &q_proj,
        1,
        dims.q_out,
        dims.hidden,
    )?;
    blas.sgemm_row_major(
        &normed,
        &weights.k_weight,
        &k_proj,
        1,
        dims.kv_out,
        dims.hidden,
    )?;
    blas.sgemm_row_major(
        &normed,
        &weights.v_weight,
        &v_proj,
        1,
        dims.kv_out,
        dims.hidden,
    )?;
    launch_rmsnorm(
        &kernels.rmsnorm,
        q_proj.as_ptr(),
        weights.q_gamma.as_ptr(),
        q_norm.as_mut_ptr(),
        dims.q_heads,
        dims.head_dim,
        dims.epsilon,
    )?;
    launch_rmsnorm(
        &kernels.rmsnorm,
        k_proj.as_ptr(),
        weights.k_gamma.as_ptr(),
        k_norm.as_mut_ptr(),
        dims.kv_heads,
        dims.head_dim,
        dims.epsilon,
    )?;
    launch_permute(
        &kernels.permute,
        q_norm.as_ptr(),
        q_bhsd.as_mut_ptr(),
        1,
        dims.q_heads,
        dims.head_dim,
    )?;
    launch_permute(
        &kernels.permute,
        k_norm.as_ptr(),
        k_bhsd.as_mut_ptr(),
        1,
        dims.kv_heads,
        dims.head_dim,
    )?;
    launch_permute(
        &kernels.permute,
        v_proj.as_ptr(),
        v_bhsd.as_mut_ptr(),
        1,
        dims.kv_heads,
        dims.head_dim,
    )?;
    launch_rope(
        &kernels.rope,
        q_bhsd.as_ptr(),
        q_rope.as_mut_ptr(),
        dims.q_out,
        dims.q_heads,
        1,
        dims.head_dim,
        offset,
        dims.theta,
    )?;
    launch_rope(
        &kernels.rope,
        k_bhsd.as_ptr(),
        k_rope.as_mut_ptr(),
        dims.kv_out,
        dims.kv_heads,
        1,
        dims.head_dim,
        offset,
        dims.theta,
    )?;
    launch_write_cache(
        &kernels.write_cache,
        k_rope.as_ptr(),
        k_cache.as_mut_ptr(),
        dims.kv_heads,
        1,
        cache_steps,
        dims.head_dim,
        offset,
    )?;
    launch_write_cache(
        &kernels.write_cache,
        v_bhsd.as_ptr(),
        v_cache.as_mut_ptr(),
        dims.kv_heads,
        1,
        cache_steps,
        dims.head_dim,
        offset,
    )?;
    launch_attention_scores(
        &kernels.scores,
        q_rope.as_ptr(),
        k_cache.as_ptr(),
        scores.as_mut_ptr(),
        dims.q_heads,
        dims.kv_heads,
        1,
        cache_steps,
        dims.head_dim,
        offset,
        dims.scale,
    )?;
    launch_softmax(
        &kernels.softmax,
        scores.as_ptr(),
        probs.as_mut_ptr(),
        dims.q_heads,
        cache_steps,
    )?;
    launch_apply_value(
        &kernels.apply_value,
        probs.as_ptr(),
        v_cache.as_ptr(),
        attended.as_mut_ptr(),
        dims.q_heads,
        dims.kv_heads,
        1,
        cache_steps,
        dims.head_dim,
    )?;
    blas.sgemm_row_major(
        &attended,
        &weights.o_weight,
        &projected,
        1,
        dims.hidden,
        dims.q_out,
    )?;
    launch_ternary(
        &kernels.residual_add,
        input.as_ptr(),
        projected.as_ptr(),
        attention_output.as_mut_ptr(),
        dims.hidden,
    )?;
    launch_rmsnorm(
        &kernels.rmsnorm,
        attention_output.as_ptr(),
        weights.post_attention_gamma.as_ptr(),
        post_norm.as_mut_ptr(),
        1,
        dims.hidden,
        dims.epsilon,
    )?;
    blas.sgemm_row_major(
        &post_norm,
        &weights.gate_weight,
        &gate,
        1,
        dims.intermediate,
        dims.hidden,
    )?;
    blas.sgemm_row_major(
        &post_norm,
        &weights.up_weight,
        &up,
        1,
        dims.intermediate,
        dims.hidden,
    )?;
    launch_ternary(
        &kernels.swiglu,
        gate.as_ptr(),
        up.as_ptr(),
        swiglu.as_mut_ptr(),
        dims.intermediate,
    )?;
    blas.sgemm_row_major(
        &swiglu,
        &weights.down_weight,
        &mlp_down,
        1,
        dims.hidden,
        dims.intermediate,
    )?;
    launch_ternary(
        &kernels.residual_add,
        attention_output.as_ptr(),
        mlp_down.as_ptr(),
        output.as_mut_ptr(),
        dims.hidden,
    )
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
    let write_cache = attention_module.function("write_kv_cache_f32")?;
    let residual_add = elementwise_module.function("residual_add_f32")?;
    let swiglu = elementwise_module.function("swiglu_f32")?;
    Ok(Kernels {
        rmsnorm,
        rope,
        permute,
        softmax,
        scores,
        apply_value,
        write_cache,
        residual_add,
        swiglu,
        _modules: vec![
            rms_module,
            rope_module,
            layout_module,
            softmax_module,
            attention_module,
            elementwise_module,
        ],
    })
}

fn load_weights(archive: &TensorArchive) -> qwen3_hip_runtime::Result<LayerWeights> {
    Ok(LayerWeights {
        input_gamma: archive.vector_f32(&format!("{LAYER_PREFIX}.input_layernorm.weight"))?,
        q_gamma: archive.vector_f32(&format!("{LAYER_PREFIX}.self_attn.q_norm.weight"))?,
        k_gamma: archive.vector_f32(&format!("{LAYER_PREFIX}.self_attn.k_norm.weight"))?,
        q_weight: archive
            .linear_weight_transposed_f32(&format!("{LAYER_PREFIX}.self_attn.q_proj.weight"))?
            .0,
        k_weight: archive
            .linear_weight_transposed_f32(&format!("{LAYER_PREFIX}.self_attn.k_proj.weight"))?
            .0,
        v_weight: archive
            .linear_weight_transposed_f32(&format!("{LAYER_PREFIX}.self_attn.v_proj.weight"))?
            .0,
        o_weight: archive
            .linear_weight_transposed_f32(&format!("{LAYER_PREFIX}.self_attn.o_proj.weight"))?
            .0,
        post_attention_gamma: archive
            .vector_f32(&format!("{LAYER_PREFIX}.post_attention_layernorm.weight"))?,
        gate_weight: archive
            .linear_weight_transposed_f32(&format!("{LAYER_PREFIX}.mlp.gate_proj.weight"))?
            .0,
        up_weight: archive
            .linear_weight_transposed_f32(&format!("{LAYER_PREFIX}.mlp.up_proj.weight"))?
            .0,
        down_weight: archive
            .linear_weight_transposed_f32(&format!("{LAYER_PREFIX}.mlp.down_proj.weight"))?
            .0,
    })
}

fn infer_dims(weights: &LayerWeights) -> qwen3_hip_runtime::Result<Dims> {
    let hidden = weights.input_gamma.len();
    let head_dim = weights.q_gamma.len();
    let q_out = weights.q_weight.len() / hidden;
    let kv_out = weights.k_weight.len() / hidden;
    let intermediate = weights.gate_weight.len() / hidden;
    if head_dim == 0
        || q_out % head_dim != 0
        || kv_out % head_dim != 0
        || weights.k_gamma.len() != head_dim
        || weights.v_weight.len() != hidden * kv_out
        || weights.o_weight.len() != q_out * hidden
        || weights.post_attention_gamma.len() != hidden
        || weights.up_weight.len() != hidden * intermediate
        || weights.down_weight.len() != intermediate * hidden
    {
        return Err(Error::InvalidInput("invalid layer dimensions".to_string()));
    }
    Ok(Dims {
        hidden,
        q_heads: q_out / head_dim,
        kv_heads: kv_out / head_dim,
        head_dim,
        q_out,
        kv_out,
        intermediate,
        epsilon: 1e-6,
        theta: 10_000.0,
        scale: (head_dim as f32).sqrt().recip(),
    })
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
    steps: usize,
    heads: usize,
    head_dim: usize,
) -> qwen3_hip_runtime::Result<()> {
    let total = steps * heads * head_dim;
    let mut input = input;
    let mut output = output;
    let mut batch = 1i32;
    let mut steps_i32 = steps as i32;
    let mut heads_i32 = heads as i32;
    let mut head_dim_i32 = head_dim as i32;
    let mut total_i32 = total as i32;
    let mut params = [
        &mut input as *mut *const c_void as *mut c_void,
        &mut output as *mut *mut c_void as *mut c_void,
        &mut batch as *mut i32 as *mut c_void,
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

#[allow(clippy::too_many_arguments)]
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

#[allow(clippy::too_many_arguments)]
fn launch_write_cache(
    function: &HipFunction,
    input: *const c_void,
    cache: *mut c_void,
    heads: usize,
    input_steps: usize,
    cache_steps: usize,
    head_dim: usize,
    offset: usize,
) -> qwen3_hip_runtime::Result<()> {
    let total = heads * input_steps * head_dim;
    let mut input = input;
    let mut cache = cache;
    let mut batch = 1i32;
    let mut heads_i32 = heads as i32;
    let mut input_steps_i32 = input_steps as i32;
    let mut cache_steps_i32 = cache_steps as i32;
    let mut head_dim_i32 = head_dim as i32;
    let mut offset_i32 = offset as i32;
    let mut total_i32 = total as i32;
    let mut params = [
        &mut input as *mut *const c_void as *mut c_void,
        &mut cache as *mut *mut c_void as *mut c_void,
        &mut batch as *mut i32 as *mut c_void,
        &mut heads_i32 as *mut i32 as *mut c_void,
        &mut input_steps_i32 as *mut i32 as *mut c_void,
        &mut cache_steps_i32 as *mut i32 as *mut c_void,
        &mut head_dim_i32 as *mut i32 as *mut c_void,
        &mut offset_i32 as *mut i32 as *mut c_void,
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

#[allow(clippy::too_many_arguments)]
fn launch_attention_scores(
    function: &HipFunction,
    q: *const c_void,
    k: *const c_void,
    scores: *mut c_void,
    q_heads: usize,
    kv_heads: usize,
    query_steps: usize,
    key_steps: usize,
    head_dim: usize,
    offset: usize,
    scale: f32,
) -> qwen3_hip_runtime::Result<()> {
    let total = q_heads * query_steps * key_steps;
    let mut q = q;
    let mut k = k;
    let mut scores = scores;
    let mut batch = 1i32;
    let mut q_heads_i32 = q_heads as i32;
    let mut kv_heads_i32 = kv_heads as i32;
    let mut query_steps_i32 = query_steps as i32;
    let mut key_steps_i32 = key_steps as i32;
    let mut head_dim_i32 = head_dim as i32;
    let mut offset_i32 = offset as i32;
    let mut scale = scale;
    let mut total_i32 = total as i32;
    let mut params = [
        &mut q as *mut *const c_void as *mut c_void,
        &mut k as *mut *const c_void as *mut c_void,
        &mut scores as *mut *mut c_void as *mut c_void,
        &mut batch as *mut i32 as *mut c_void,
        &mut q_heads_i32 as *mut i32 as *mut c_void,
        &mut kv_heads_i32 as *mut i32 as *mut c_void,
        &mut query_steps_i32 as *mut i32 as *mut c_void,
        &mut key_steps_i32 as *mut i32 as *mut c_void,
        &mut head_dim_i32 as *mut i32 as *mut c_void,
        &mut offset_i32 as *mut i32 as *mut c_void,
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

#[allow(clippy::too_many_arguments)]
fn launch_apply_value(
    function: &HipFunction,
    probs: *const c_void,
    v: *const c_void,
    output: *mut c_void,
    q_heads: usize,
    kv_heads: usize,
    query_steps: usize,
    key_steps: usize,
    head_dim: usize,
) -> qwen3_hip_runtime::Result<()> {
    let total = query_steps * q_heads * head_dim;
    let mut probs = probs;
    let mut v = v;
    let mut output = output;
    let mut batch = 1i32;
    let mut q_heads_i32 = q_heads as i32;
    let mut kv_heads_i32 = kv_heads as i32;
    let mut query_steps_i32 = query_steps as i32;
    let mut key_steps_i32 = key_steps as i32;
    let mut head_dim_i32 = head_dim as i32;
    let mut total_i32 = total as i32;
    let mut params = [
        &mut probs as *mut *const c_void as *mut c_void,
        &mut v as *mut *const c_void as *mut c_void,
        &mut output as *mut *mut c_void as *mut c_void,
        &mut batch as *mut i32 as *mut c_void,
        &mut q_heads_i32 as *mut i32 as *mut c_void,
        &mut kv_heads_i32 as *mut i32 as *mut c_void,
        &mut query_steps_i32 as *mut i32 as *mut c_void,
        &mut key_steps_i32 as *mut i32 as *mut c_void,
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

fn layer_reference(input: &[f32], weights: &LayerWeights, dims: Dims, steps: usize) -> Vec<f32> {
    let normed = rmsnorm_reference(
        input,
        &weights.input_gamma,
        steps,
        dims.hidden,
        dims.epsilon,
    );
    let q = row_major_matmul(&normed, &weights.q_weight, steps, dims.q_out, dims.hidden);
    let k = row_major_matmul(&normed, &weights.k_weight, steps, dims.kv_out, dims.hidden);
    let v = row_major_matmul(&normed, &weights.v_weight, steps, dims.kv_out, dims.hidden);
    let q = rmsnorm_reference(
        &q,
        &weights.q_gamma,
        steps * dims.q_heads,
        dims.head_dim,
        dims.epsilon,
    );
    let k = rmsnorm_reference(
        &k,
        &weights.k_gamma,
        steps * dims.kv_heads,
        dims.head_dim,
        dims.epsilon,
    );
    let q = rope_reference(
        &permute_bshd_to_bhsd(&q, steps, dims.q_heads, dims.head_dim),
        steps,
        dims.head_dim,
        0,
        dims.theta,
    );
    let k = rope_reference(
        &permute_bshd_to_bhsd(&k, steps, dims.kv_heads, dims.head_dim),
        steps,
        dims.head_dim,
        0,
        dims.theta,
    );
    let v = permute_bshd_to_bhsd(&v, steps, dims.kv_heads, dims.head_dim);
    let scores = attention_scores_reference(&q, &k, dims, steps, steps, 0);
    let probs = softmax_rows(&scores, dims.q_heads * steps, steps);
    let attended = attention_apply_value_reference(&probs, &v, dims, steps, steps);
    let projected = row_major_matmul(&attended, &weights.o_weight, steps, dims.hidden, dims.q_out);
    let attention_output = input
        .iter()
        .zip(projected.iter())
        .map(|(input, projected)| input + projected)
        .collect::<Vec<_>>();
    let post_norm = rmsnorm_reference(
        &attention_output,
        &weights.post_attention_gamma,
        steps,
        dims.hidden,
        dims.epsilon,
    );
    let gate = row_major_matmul(
        &post_norm,
        &weights.gate_weight,
        steps,
        dims.intermediate,
        dims.hidden,
    );
    let up = row_major_matmul(
        &post_norm,
        &weights.up_weight,
        steps,
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
        &weights.down_weight,
        steps,
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

fn permute_bshd_to_bhsd(input: &[f32], steps: usize, heads: usize, head_dim: usize) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    for step in 0..steps {
        for head in 0..heads {
            for dim in 0..head_dim {
                let input_idx = ((step * heads + head) * head_dim) + dim;
                let output_idx = ((head * steps + step) * head_dim) + dim;
                output[output_idx] = input[input_idx];
            }
        }
    }
    output
}

fn rope_reference(
    input: &[f32],
    steps: usize,
    head_dim: usize,
    offset: usize,
    theta: f32,
) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    let half = head_dim / 2;
    for idx in 0..input.len() {
        let dim_index = idx % head_dim;
        let step_index = (idx / head_dim) % steps;
        let pair_index = dim_index % half;
        let base = idx - dim_index;
        let first = input[base + pair_index];
        let second = input[base + pair_index + half];
        let exponent = (pair_index * 2) as f32 / head_dim as f32;
        let angle = (offset + step_index) as f32 / theta.powf(exponent);
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

fn attention_scores_reference(
    q: &[f32],
    k: &[f32],
    dims: Dims,
    query_steps: usize,
    key_steps: usize,
    offset: usize,
) -> Vec<f32> {
    let mut scores = vec![0.0; dims.q_heads * query_steps * key_steps];
    let n_rep = dims.q_heads / dims.kv_heads;
    for q_head in 0..dims.q_heads {
        let kv_head = q_head / n_rep;
        for query_step in 0..query_steps {
            for key_step in 0..key_steps {
                let score_idx = ((q_head * query_steps + query_step) * key_steps) + key_step;
                if key_step > offset + query_step {
                    scores[score_idx] = f32::NEG_INFINITY;
                    continue;
                }
                let q_base = ((q_head * query_steps + query_step) * dims.head_dim) as usize;
                let k_base = ((kv_head * key_steps + key_step) * dims.head_dim) as usize;
                let mut sum = 0.0;
                for dim in 0..dims.head_dim {
                    sum += q[q_base + dim] * k[k_base + dim];
                }
                scores[score_idx] = sum * dims.scale;
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

fn attention_apply_value_reference(
    probs: &[f32],
    v: &[f32],
    dims: Dims,
    query_steps: usize,
    key_steps: usize,
) -> Vec<f32> {
    let mut output = vec![0.0; query_steps * dims.q_heads * dims.head_dim];
    let n_rep = dims.q_heads / dims.kv_heads;
    for query_step in 0..query_steps {
        for q_head in 0..dims.q_heads {
            let kv_head = q_head / n_rep;
            for dim in 0..dims.head_dim {
                let mut sum = 0.0;
                for key_step in 0..key_steps {
                    let prob_idx = ((q_head * query_steps + query_step) * key_steps) + key_step;
                    let v_idx = ((kv_head * key_steps + key_step) * dims.head_dim) + dim;
                    sum += probs[prob_idx] * v[v_idx];
                }
                let out_idx = ((query_step * dims.q_heads + q_head) * dims.head_dim) + dim;
                output[out_idx] = sum;
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
