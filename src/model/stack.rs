use std::ffi::c_void;
use std::path::Path;

use safetensors::SafeTensors;

use crate::blas::RocblasHandle;
use crate::buffer::DeviceBuffer;
use crate::error::{Error, Result};
use crate::kernel::{HipFunction, HipModule};
use crate::kernels::{
    ATTENTION_F32_SOURCE, ELEMENTWISE_F32_SOURCE, LAYOUT_F32_SOURCE, RMSNORM_F32_SOURCE,
    ROPE_BHSD_F32_SOURCE, SOFTMAX_F32_SOURCE,
};
use crate::runtime::HipRuntime;
use crate::weights::{TensorArchive, tensor_to_f32};

#[derive(Clone, Copy, Debug)]
pub struct DecoderStackDims {
    pub batch: usize,
    pub steps: usize,
    pub rows: usize,
    pub hidden: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub q_out: usize,
    pub kv_out: usize,
    pub intermediate: usize,
    pub offset: usize,
    pub theta: f32,
    pub epsilon: f32,
    pub scale: f32,
}

pub struct DecoderStack {
    dims: DecoderStackDims,
    layers: Vec<DeviceDecoderLayer>,
    blas: RocblasHandle,
    kernels: StackKernels,
    workspace: StackWorkspace,
}

struct HostDecoderLayer {
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

struct DeviceDecoderLayer {
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

struct StackKernels {
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

struct StackWorkspace {
    hidden_scratch: DeviceBuffer<f32>,
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

impl DecoderStack {
    pub fn load(runtime: &HipRuntime, model_dir: &Path, layer_count: usize) -> Result<Self> {
        if layer_count == 0 {
            return Err(Error::InvalidInput(
                "layer count must be non-zero".to_string(),
            ));
        }
        let archive = TensorArchive::open(&model_dir.join("model.safetensors"))?;
        let (dims, layers) = archive.with_tensors(|tensors| {
            let first = load_host_layer(tensors, 0, None)?;
            let dims = infer_dims(&first)?;
            let mut layers = Vec::with_capacity(layer_count);
            layers.push(DeviceDecoderLayer::from_host(runtime, &first)?);
            for index in 1..layer_count {
                let host = load_host_layer(tensors, index, Some(dims))?;
                layers.push(DeviceDecoderLayer::from_host(runtime, &host)?);
            }
            Ok((dims, layers))
        })?;

        Ok(Self {
            dims,
            layers,
            blas: runtime.create_blas_handle()?,
            kernels: StackKernels::compile(runtime)?,
            workspace: StackWorkspace::new(runtime, dims)?,
        })
    }

    pub fn dims(&self) -> DecoderStackDims {
        self.dims
    }

    pub fn input_len(&self) -> usize {
        self.dims.rows * self.dims.hidden
    }

    pub fn forward(&self, input: &DeviceBuffer<f32>, output: &DeviceBuffer<f32>) -> Result<()> {
        if input.len() != self.input_len() || output.len() != self.input_len() {
            return Err(Error::InvalidInput(format!(
                "input/output length must be {}, got input={}, output={}",
                self.input_len(),
                input.len(),
                output.len()
            )));
        }

        let mut current_is_output = false;
        for (index, layer) in self.layers.iter().enumerate() {
            let layer_input = if index == 0 {
                input
            } else if current_is_output {
                output
            } else {
                &self.workspace.hidden_scratch
            };
            let layer_output = if current_is_output {
                &self.workspace.hidden_scratch
            } else {
                output
            };
            self.run_layer(layer_input, layer_output, layer)?;
            current_is_output = !current_is_output;
        }

        if !current_is_output {
            output.copy_from_device(&self.workspace.hidden_scratch)?;
        }
        Ok(())
    }

    fn run_layer(
        &self,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        layer: &DeviceDecoderLayer,
    ) -> Result<()> {
        let d = self.dims;
        let w = &self.workspace;
        let k = &self.kernels;
        launch_rmsnorm(
            &k.rmsnorm,
            input.as_ptr(),
            layer.input_gamma.as_ptr(),
            w.normed.as_mut_ptr(),
            d.rows,
            d.hidden,
            d.epsilon,
        )?;
        self.blas.sgemm_row_major(
            &w.normed,
            &layer.q_weight,
            &w.q_proj,
            d.rows,
            d.q_out,
            d.hidden,
        )?;
        self.blas.sgemm_row_major(
            &w.normed,
            &layer.k_weight,
            &w.k_proj,
            d.rows,
            d.kv_out,
            d.hidden,
        )?;
        self.blas.sgemm_row_major(
            &w.normed,
            &layer.v_weight,
            &w.v_proj,
            d.rows,
            d.kv_out,
            d.hidden,
        )?;
        launch_rmsnorm(
            &k.rmsnorm,
            w.q_proj.as_ptr(),
            layer.q_gamma.as_ptr(),
            w.q_norm.as_mut_ptr(),
            d.rows * d.q_heads,
            d.head_dim,
            d.epsilon,
        )?;
        launch_rmsnorm(
            &k.rmsnorm,
            w.k_proj.as_ptr(),
            layer.k_gamma.as_ptr(),
            w.k_norm.as_mut_ptr(),
            d.rows * d.kv_heads,
            d.head_dim,
            d.epsilon,
        )?;
        launch_permute(
            &k.permute,
            w.q_norm.as_ptr(),
            w.q_bhsd.as_mut_ptr(),
            d.batch,
            d.steps,
            d.q_heads,
            d.head_dim,
        )?;
        launch_permute(
            &k.permute,
            w.k_norm.as_ptr(),
            w.k_bhsd.as_mut_ptr(),
            d.batch,
            d.steps,
            d.kv_heads,
            d.head_dim,
        )?;
        launch_permute(
            &k.permute,
            w.v_proj.as_ptr(),
            w.v_bhsd.as_mut_ptr(),
            d.batch,
            d.steps,
            d.kv_heads,
            d.head_dim,
        )?;
        launch_rope(
            &k.rope,
            w.q_bhsd.as_ptr(),
            w.q_rope.as_mut_ptr(),
            d.batch * d.q_heads * d.steps * d.head_dim,
            d.q_heads,
            d.steps,
            d.head_dim,
            d.offset,
            d.theta,
        )?;
        launch_rope(
            &k.rope,
            w.k_bhsd.as_ptr(),
            w.k_rope.as_mut_ptr(),
            d.batch * d.kv_heads * d.steps * d.head_dim,
            d.kv_heads,
            d.steps,
            d.head_dim,
            d.offset,
            d.theta,
        )?;
        launch_attention_scores(
            &k.scores,
            w.q_rope.as_ptr(),
            w.k_rope.as_ptr(),
            w.scores.as_mut_ptr(),
            d,
        )?;
        launch_softmax(
            &k.softmax,
            w.scores.as_ptr(),
            w.probs.as_mut_ptr(),
            d.batch * d.q_heads * d.steps,
            d.steps,
        )?;
        launch_apply_value(
            &k.apply_value,
            w.probs.as_ptr(),
            w.v_bhsd.as_ptr(),
            w.attended.as_mut_ptr(),
            d,
        )?;
        self.blas.sgemm_row_major(
            &w.attended,
            &layer.o_weight,
            &w.projected,
            d.rows,
            d.hidden,
            d.q_out,
        )?;
        launch_ternary(
            &k.residual_add,
            input.as_ptr(),
            w.projected.as_ptr(),
            w.attention_output.as_mut_ptr(),
            d.rows * d.hidden,
        )?;
        launch_rmsnorm(
            &k.rmsnorm,
            w.attention_output.as_ptr(),
            layer.post_attention_gamma.as_ptr(),
            w.post_norm.as_mut_ptr(),
            d.rows,
            d.hidden,
            d.epsilon,
        )?;
        self.blas.sgemm_row_major(
            &w.post_norm,
            &layer.gate_weight,
            &w.gate,
            d.rows,
            d.intermediate,
            d.hidden,
        )?;
        self.blas.sgemm_row_major(
            &w.post_norm,
            &layer.up_weight,
            &w.up,
            d.rows,
            d.intermediate,
            d.hidden,
        )?;
        launch_ternary(
            &k.swiglu,
            w.gate.as_ptr(),
            w.up.as_ptr(),
            w.swiglu.as_mut_ptr(),
            d.rows * d.intermediate,
        )?;
        self.blas.sgemm_row_major(
            &w.swiglu,
            &layer.down_weight,
            &w.mlp_down,
            d.rows,
            d.hidden,
            d.intermediate,
        )?;
        launch_ternary(
            &k.residual_add,
            w.attention_output.as_ptr(),
            w.mlp_down.as_ptr(),
            output.as_mut_ptr(),
            d.rows * d.hidden,
        )
    }
}

impl DeviceDecoderLayer {
    fn from_host(runtime: &HipRuntime, layer: &HostDecoderLayer) -> Result<Self> {
        Ok(Self {
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
}

impl StackKernels {
    fn compile(runtime: &HipRuntime) -> Result<Self> {
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
        Ok(Self {
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
}

impl StackWorkspace {
    fn new(runtime: &HipRuntime, d: DecoderStackDims) -> Result<Self> {
        Ok(Self {
            hidden_scratch: runtime.empty_buffer::<f32>(d.rows * d.hidden)?,
            normed: runtime.empty_buffer::<f32>(d.rows * d.hidden)?,
            q_proj: runtime.empty_buffer::<f32>(d.rows * d.q_out)?,
            k_proj: runtime.empty_buffer::<f32>(d.rows * d.kv_out)?,
            v_proj: runtime.empty_buffer::<f32>(d.rows * d.kv_out)?,
            q_norm: runtime.empty_buffer::<f32>(d.rows * d.q_out)?,
            k_norm: runtime.empty_buffer::<f32>(d.rows * d.kv_out)?,
            q_bhsd: runtime.empty_buffer::<f32>(d.batch * d.q_heads * d.steps * d.head_dim)?,
            k_bhsd: runtime.empty_buffer::<f32>(d.batch * d.kv_heads * d.steps * d.head_dim)?,
            v_bhsd: runtime.empty_buffer::<f32>(d.batch * d.kv_heads * d.steps * d.head_dim)?,
            q_rope: runtime.empty_buffer::<f32>(d.batch * d.q_heads * d.steps * d.head_dim)?,
            k_rope: runtime.empty_buffer::<f32>(d.batch * d.kv_heads * d.steps * d.head_dim)?,
            scores: runtime.empty_buffer::<f32>(d.batch * d.q_heads * d.steps * d.steps)?,
            probs: runtime.empty_buffer::<f32>(d.batch * d.q_heads * d.steps * d.steps)?,
            attended: runtime.empty_buffer::<f32>(d.rows * d.q_heads * d.head_dim)?,
            projected: runtime.empty_buffer::<f32>(d.rows * d.hidden)?,
            attention_output: runtime.empty_buffer::<f32>(d.rows * d.hidden)?,
            post_norm: runtime.empty_buffer::<f32>(d.rows * d.hidden)?,
            gate: runtime.empty_buffer::<f32>(d.rows * d.intermediate)?,
            up: runtime.empty_buffer::<f32>(d.rows * d.intermediate)?,
            swiglu: runtime.empty_buffer::<f32>(d.rows * d.intermediate)?,
            mlp_down: runtime.empty_buffer::<f32>(d.rows * d.hidden)?,
        })
    }
}

fn load_host_layer(
    tensors: &SafeTensors<'_>,
    index: usize,
    expected: Option<DecoderStackDims>,
) -> Result<HostDecoderLayer> {
    let prefix = format!("talker.model.layers.{index}");
    let layer = HostDecoderLayer {
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

fn infer_dims(layer: &HostDecoderLayer) -> Result<DecoderStackDims> {
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
    Ok(DecoderStackDims {
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

fn vector_f32(tensors: &SafeTensors<'_>, name: &str) -> Result<Vec<f32>> {
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

fn linear_weight_transposed_f32(tensors: &SafeTensors<'_>, name: &str) -> Result<Vec<f32>> {
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

fn launch_rmsnorm(
    function: &HipFunction,
    input: *const c_void,
    gamma: *const c_void,
    output: *mut c_void,
    rows: usize,
    cols: usize,
    epsilon: f32,
) -> Result<()> {
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
) -> Result<()> {
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
) -> Result<()> {
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
    d: DecoderStackDims,
) -> Result<()> {
    let total = d.batch * d.q_heads * d.steps * d.steps;
    let mut q = q;
    let mut k = k;
    let mut scores = scores;
    let mut batch = d.batch as i32;
    let mut q_heads = d.q_heads as i32;
    let mut kv_heads = d.kv_heads as i32;
    let mut query_steps = d.steps as i32;
    let mut key_steps = d.steps as i32;
    let mut head_dim = d.head_dim as i32;
    let mut offset = d.offset as i32;
    let mut scale = d.scale;
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
) -> Result<()> {
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
    d: DecoderStackDims,
) -> Result<()> {
    let total = d.batch * d.steps * d.q_heads * d.head_dim;
    let mut probs = probs;
    let mut v = v;
    let mut output = output;
    let mut batch = d.batch as i32;
    let mut q_heads = d.q_heads as i32;
    let mut kv_heads = d.kv_heads as i32;
    let mut query_steps = d.steps as i32;
    let mut key_steps = d.steps as i32;
    let mut head_dim = d.head_dim as i32;
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
) -> Result<()> {
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
