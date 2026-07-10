use std::cell::RefCell;
use std::ffi::c_void;
use std::path::Path;
use std::time::Instant;

use safetensors::SafeTensors;

use crate::blas::RocblasHandle;
use crate::buffer::DeviceBuffer;
use crate::error::{Error, Result};
use crate::graph::HipStream;
use crate::kernel::{HipFunction, HipModule};
use crate::kernels::{
    ATTENTION_F32_SOURCE, ELEMENTWISE_F32_SOURCE, LAYOUT_F32_SOURCE, RMSNORM_F32_SOURCE,
    ROPE_BHSD_F32_SOURCE, SOFTMAX_F32_SOURCE,
};
use crate::runtime::HipRuntime;
use crate::weights::{TensorArchive, tensor_to_f32};

#[derive(Clone, Copy, Debug)]
pub struct DecodeStepDims {
    pub hidden: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub q_out: usize,
    pub kv_out: usize,
    pub intermediate: usize,
    pub max_cache_steps: usize,
    pub epsilon: f32,
    pub theta: f32,
    pub scale: f32,
}

#[derive(Clone, Debug, Default)]
pub struct DecodeStepProfile {
    pub input_norm_seconds: f64,
    pub qkv_gemm_seconds: f64,
    pub qk_layout_cache_seconds: f64,
    pub attention_seconds: f64,
    pub output_gemm_residual_seconds: f64,
    pub post_norm_seconds: f64,
    pub gate_up_gemm_seconds: f64,
    pub swiglu_seconds: f64,
    pub down_gemm_residual_seconds: f64,
    pub final_copy_seconds: f64,
}

impl DecodeStepProfile {
    pub fn total_seconds(&self) -> f64 {
        self.input_norm_seconds
            + self.qkv_gemm_seconds
            + self.qk_layout_cache_seconds
            + self.attention_seconds
            + self.output_gemm_residual_seconds
            + self.post_norm_seconds
            + self.gate_up_gemm_seconds
            + self.swiglu_seconds
            + self.down_gemm_residual_seconds
            + self.final_copy_seconds
    }

    fn add(&mut self, other: &Self) {
        self.input_norm_seconds += other.input_norm_seconds;
        self.qkv_gemm_seconds += other.qkv_gemm_seconds;
        self.qk_layout_cache_seconds += other.qk_layout_cache_seconds;
        self.attention_seconds += other.attention_seconds;
        self.output_gemm_residual_seconds += other.output_gemm_residual_seconds;
        self.post_norm_seconds += other.post_norm_seconds;
        self.gate_up_gemm_seconds += other.gate_up_gemm_seconds;
        self.swiglu_seconds += other.swiglu_seconds;
        self.down_gemm_residual_seconds += other.down_gemm_residual_seconds;
        self.final_copy_seconds += other.final_copy_seconds;
    }
}

pub struct DecodeStepLayer {
    dims: DecodeStepDims,
    weights: DecodeDeviceWeights,
    blas: RocblasHandle,
    kernels: DecodeKernels,
    workspace: DecodeWorkspace,
}

pub struct DecodeStepStack {
    weights: Vec<DecodeDeviceWeights>,
    blas: RocblasHandle,
    kernels: DecodeKernels,
    runtime: HipRuntime,
    storage: RefCell<DecodeCacheStorage>,
    current_scratch: DeviceBuffer<f32>,
}

struct DecodeCacheStorage {
    dims: DecodeStepDims,
    caches: Vec<DecodeLayerCache>,
    workspace: DecodeWorkspace,
    prefix_a: DeviceBuffer<f32>,
    prefix_b: DeviceBuffer<f32>,
}

struct DecodeHostWeights {
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

struct DecodeDeviceWeights {
    input_gamma: DeviceBuffer<f32>,
    q_gamma: DeviceBuffer<f32>,
    k_gamma: DeviceBuffer<f32>,
    q_weight: DeviceBuffer<f32>,
    k_weight: DeviceBuffer<f32>,
    v_weight: DeviceBuffer<f32>,
    qkv_weight: DeviceBuffer<f32>,
    o_weight: DeviceBuffer<f32>,
    post_attention_gamma: DeviceBuffer<f32>,
    gate_weight: DeviceBuffer<f32>,
    up_weight: DeviceBuffer<f32>,
    gate_up_weight: DeviceBuffer<f32>,
    down_weight: DeviceBuffer<f32>,
}

struct DecodeLayerCache {
    k_cache: DeviceBuffer<f32>,
    v_cache: DeviceBuffer<f32>,
}

struct DecodeKernels {
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
    scores_cache: HipFunction,
    apply_value: HipFunction,
    apply_value_cache: HipFunction,
    write_cache: HipFunction,
    residual_add: HipFunction,
    swiglu: HipFunction,
}

struct DecodeWorkspace {
    prefill_normed: DeviceBuffer<f32>,
    prefill_q_proj: DeviceBuffer<f32>,
    prefill_k_proj: DeviceBuffer<f32>,
    prefill_v_proj: DeviceBuffer<f32>,
    prefill_q_norm: DeviceBuffer<f32>,
    prefill_k_norm: DeviceBuffer<f32>,
    prefill_q_bhsd: DeviceBuffer<f32>,
    prefill_k_bhsd: DeviceBuffer<f32>,
    prefill_v_bhsd: DeviceBuffer<f32>,
    prefill_q_rope: DeviceBuffer<f32>,
    prefill_k_rope: DeviceBuffer<f32>,
    prefill_scores: DeviceBuffer<f32>,
    prefill_probs: DeviceBuffer<f32>,
    prefill_attended: DeviceBuffer<f32>,
    prefill_projected: DeviceBuffer<f32>,
    prefill_attention_output: DeviceBuffer<f32>,
    prefill_post_norm: DeviceBuffer<f32>,
    prefill_gate: DeviceBuffer<f32>,
    prefill_up: DeviceBuffer<f32>,
    prefill_swiglu: DeviceBuffer<f32>,
    prefill_mlp_down: DeviceBuffer<f32>,
    k_cache: DeviceBuffer<f32>,
    v_cache: DeviceBuffer<f32>,
    normed: DeviceBuffer<f32>,
    qkv_proj: DeviceBuffer<f32>,
    q_norm: DeviceBuffer<f32>,
    k_norm: DeviceBuffer<f32>,
    q_rope: DeviceBuffer<f32>,
    k_rope: DeviceBuffer<f32>,
    scores: DeviceBuffer<f32>,
    probs: DeviceBuffer<f32>,
    attended: DeviceBuffer<f32>,
    projected: DeviceBuffer<f32>,
    attention_output: DeviceBuffer<f32>,
    post_norm: DeviceBuffer<f32>,
    gate_up: DeviceBuffer<f32>,
    swiglu: DeviceBuffer<f32>,
    mlp_down: DeviceBuffer<f32>,
}

impl DecodeStepLayer {
    pub fn load(
        runtime: &HipRuntime,
        model_dir: &Path,
        layer_index: usize,
        max_cache_steps: usize,
    ) -> Result<Self> {
        if max_cache_steps == 0 {
            return Err(Error::InvalidInput(
                "max cache steps must be non-zero".to_string(),
            ));
        }
        let host = load_host_weights(model_dir, layer_index)?;
        let dims = infer_dims(&host, max_cache_steps, 1e-6, 10_000.0)?;
        Ok(Self {
            dims,
            weights: DecodeDeviceWeights::from_host(runtime, &host)?,
            blas: runtime.create_blas_handle()?,
            kernels: DecodeKernels::compile(runtime)?,
            workspace: DecodeWorkspace::new(runtime, dims)?,
        })
    }

    pub fn dims(&self) -> DecodeStepDims {
        self.dims
    }

    pub fn input_len(&self) -> usize {
        self.dims.hidden
    }

    pub fn prefill(&self, prefix: &DeviceBuffer<f32>, prefix_steps: usize) -> Result<()> {
        let d = self.dims;
        let w = &self.workspace;
        let k = &self.kernels;
        if prefix_steps == 0 || prefix_steps > d.max_cache_steps {
            return Err(Error::InvalidInput(format!(
                "prefix steps {prefix_steps} outside 1..={}",
                d.max_cache_steps
            )));
        }
        if prefix.len() < prefix_steps * d.hidden {
            return Err(Error::InvalidInput(format!(
                "prefix length {} is smaller than {}",
                prefix.len(),
                prefix_steps * d.hidden
            )));
        }

        launch_rmsnorm(
            &k.rmsnorm,
            prefix.as_ptr(),
            self.weights.input_gamma.as_ptr(),
            w.prefill_normed.as_mut_ptr(),
            prefix_steps,
            d.hidden,
            d.epsilon,
        )?;
        self.blas.sgemm_row_major(
            &w.prefill_normed,
            &self.weights.k_weight,
            &w.prefill_k_proj,
            prefix_steps,
            d.kv_out,
            d.hidden,
        )?;
        self.blas.sgemm_row_major(
            &w.prefill_normed,
            &self.weights.v_weight,
            &w.prefill_v_proj,
            prefix_steps,
            d.kv_out,
            d.hidden,
        )?;
        launch_rmsnorm(
            &k.rmsnorm,
            w.prefill_k_proj.as_ptr(),
            self.weights.k_gamma.as_ptr(),
            w.prefill_k_norm.as_mut_ptr(),
            prefix_steps * d.kv_heads,
            d.head_dim,
            d.epsilon,
        )?;
        launch_permute(
            &k.permute,
            w.prefill_k_norm.as_ptr(),
            w.prefill_k_bhsd.as_mut_ptr(),
            prefix_steps,
            d.kv_heads,
            d.head_dim,
        )?;
        launch_permute(
            &k.permute,
            w.prefill_v_proj.as_ptr(),
            w.prefill_v_bhsd.as_mut_ptr(),
            prefix_steps,
            d.kv_heads,
            d.head_dim,
        )?;
        launch_rope(
            &k.rope,
            w.prefill_k_bhsd.as_ptr(),
            w.prefill_k_rope.as_mut_ptr(),
            d.kv_heads * prefix_steps * d.head_dim,
            d.kv_heads,
            prefix_steps,
            d.head_dim,
            0,
            d.theta,
        )?;
        launch_write_cache(
            &k.write_cache,
            w.prefill_k_rope.as_ptr(),
            w.k_cache.as_mut_ptr(),
            d.kv_heads,
            prefix_steps,
            d.max_cache_steps,
            d.head_dim,
            0,
        )?;
        launch_write_cache(
            &k.write_cache,
            w.prefill_v_bhsd.as_ptr(),
            w.v_cache.as_mut_ptr(),
            d.kv_heads,
            prefix_steps,
            d.max_cache_steps,
            d.head_dim,
            0,
        )
    }

    pub fn prefill_forward(
        &self,
        prefix: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        prefix_steps: usize,
    ) -> Result<()> {
        let d = self.dims;
        let w = &self.workspace;
        let k = &self.kernels;
        if prefix_steps == 0 || prefix_steps > d.max_cache_steps {
            return Err(Error::InvalidInput(format!(
                "prefix steps {prefix_steps} outside 1..={}",
                d.max_cache_steps
            )));
        }
        if prefix.len() < prefix_steps * d.hidden || output.len() < prefix_steps * d.hidden {
            return Err(Error::InvalidInput(format!(
                "prefill input/output length must be at least {}, got input={}, output={}",
                prefix_steps * d.hidden,
                prefix.len(),
                output.len()
            )));
        }

        launch_rmsnorm(
            &k.rmsnorm,
            prefix.as_ptr(),
            self.weights.input_gamma.as_ptr(),
            w.prefill_normed.as_mut_ptr(),
            prefix_steps,
            d.hidden,
            d.epsilon,
        )?;
        self.blas.sgemm_row_major(
            &w.prefill_normed,
            &self.weights.q_weight,
            &w.prefill_q_proj,
            prefix_steps,
            d.q_out,
            d.hidden,
        )?;
        self.blas.sgemm_row_major(
            &w.prefill_normed,
            &self.weights.k_weight,
            &w.prefill_k_proj,
            prefix_steps,
            d.kv_out,
            d.hidden,
        )?;
        self.blas.sgemm_row_major(
            &w.prefill_normed,
            &self.weights.v_weight,
            &w.prefill_v_proj,
            prefix_steps,
            d.kv_out,
            d.hidden,
        )?;
        launch_rmsnorm(
            &k.rmsnorm,
            w.prefill_q_proj.as_ptr(),
            self.weights.q_gamma.as_ptr(),
            w.prefill_q_norm.as_mut_ptr(),
            prefix_steps * d.q_heads,
            d.head_dim,
            d.epsilon,
        )?;
        launch_rmsnorm(
            &k.rmsnorm,
            w.prefill_k_proj.as_ptr(),
            self.weights.k_gamma.as_ptr(),
            w.prefill_k_norm.as_mut_ptr(),
            prefix_steps * d.kv_heads,
            d.head_dim,
            d.epsilon,
        )?;
        launch_permute(
            &k.permute,
            w.prefill_q_norm.as_ptr(),
            w.prefill_q_bhsd.as_mut_ptr(),
            prefix_steps,
            d.q_heads,
            d.head_dim,
        )?;
        launch_permute(
            &k.permute,
            w.prefill_k_norm.as_ptr(),
            w.prefill_k_bhsd.as_mut_ptr(),
            prefix_steps,
            d.kv_heads,
            d.head_dim,
        )?;
        launch_permute(
            &k.permute,
            w.prefill_v_proj.as_ptr(),
            w.prefill_v_bhsd.as_mut_ptr(),
            prefix_steps,
            d.kv_heads,
            d.head_dim,
        )?;
        launch_rope(
            &k.rope,
            w.prefill_q_bhsd.as_ptr(),
            w.prefill_q_rope.as_mut_ptr(),
            d.q_heads * prefix_steps * d.head_dim,
            d.q_heads,
            prefix_steps,
            d.head_dim,
            0,
            d.theta,
        )?;
        launch_rope(
            &k.rope,
            w.prefill_k_bhsd.as_ptr(),
            w.prefill_k_rope.as_mut_ptr(),
            d.kv_heads * prefix_steps * d.head_dim,
            d.kv_heads,
            prefix_steps,
            d.head_dim,
            0,
            d.theta,
        )?;
        launch_write_cache(
            &k.write_cache,
            w.prefill_k_rope.as_ptr(),
            w.k_cache.as_mut_ptr(),
            d.kv_heads,
            prefix_steps,
            d.max_cache_steps,
            d.head_dim,
            0,
        )?;
        launch_write_cache(
            &k.write_cache,
            w.prefill_v_bhsd.as_ptr(),
            w.v_cache.as_mut_ptr(),
            d.kv_heads,
            prefix_steps,
            d.max_cache_steps,
            d.head_dim,
            0,
        )?;
        launch_attention_scores(
            &k.scores,
            w.prefill_q_rope.as_ptr(),
            w.prefill_k_rope.as_ptr(),
            w.prefill_scores.as_mut_ptr(),
            d.q_heads,
            d.kv_heads,
            prefix_steps,
            prefix_steps,
            d.head_dim,
            0,
            d.scale,
        )?;
        launch_softmax(
            &k.softmax,
            w.prefill_scores.as_ptr(),
            w.prefill_probs.as_mut_ptr(),
            d.q_heads * prefix_steps,
            prefix_steps,
        )?;
        launch_apply_value(
            &k.apply_value,
            w.prefill_probs.as_ptr(),
            w.prefill_v_bhsd.as_ptr(),
            w.prefill_attended.as_mut_ptr(),
            d.q_heads,
            d.kv_heads,
            prefix_steps,
            prefix_steps,
            d.head_dim,
        )?;
        self.blas.sgemm_row_major(
            &w.prefill_attended,
            &self.weights.o_weight,
            &w.prefill_projected,
            prefix_steps,
            d.hidden,
            d.q_out,
        )?;
        launch_ternary(
            &k.residual_add,
            prefix.as_ptr(),
            w.prefill_projected.as_ptr(),
            w.prefill_attention_output.as_mut_ptr(),
            prefix_steps * d.hidden,
        )?;
        launch_rmsnorm(
            &k.rmsnorm,
            w.prefill_attention_output.as_ptr(),
            self.weights.post_attention_gamma.as_ptr(),
            w.prefill_post_norm.as_mut_ptr(),
            prefix_steps,
            d.hidden,
            d.epsilon,
        )?;
        self.blas.sgemm_row_major(
            &w.prefill_post_norm,
            &self.weights.gate_weight,
            &w.prefill_gate,
            prefix_steps,
            d.intermediate,
            d.hidden,
        )?;
        self.blas.sgemm_row_major(
            &w.prefill_post_norm,
            &self.weights.up_weight,
            &w.prefill_up,
            prefix_steps,
            d.intermediate,
            d.hidden,
        )?;
        launch_ternary(
            &k.swiglu,
            w.prefill_gate.as_ptr(),
            w.prefill_up.as_ptr(),
            w.prefill_swiglu.as_mut_ptr(),
            prefix_steps * d.intermediate,
        )?;
        self.blas.sgemm_row_major(
            &w.prefill_swiglu,
            &self.weights.down_weight,
            &w.prefill_mlp_down,
            prefix_steps,
            d.hidden,
            d.intermediate,
        )?;
        launch_ternary(
            &k.residual_add,
            w.prefill_attention_output.as_ptr(),
            w.prefill_mlp_down.as_ptr(),
            output.as_mut_ptr(),
            prefix_steps * d.hidden,
        )
    }

    pub fn decode_step(
        &self,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        offset: usize,
    ) -> Result<()> {
        let d = self.dims;
        if input.len() != d.hidden || output.len() != d.hidden {
            return Err(Error::InvalidInput(format!(
                "decode input/output length must be {}, got input={}, output={}",
                d.hidden,
                input.len(),
                output.len()
            )));
        }
        if offset >= d.max_cache_steps {
            return Err(Error::InvalidInput(format!(
                "offset {offset} outside cache length {}",
                d.max_cache_steps
            )));
        }

        let w = &self.workspace;
        let k = &self.kernels;
        let active_steps = offset + 1;
        launch_rmsnorm(
            &k.rmsnorm,
            input.as_ptr(),
            self.weights.input_gamma.as_ptr(),
            w.normed.as_mut_ptr(),
            1,
            d.hidden,
            d.epsilon,
        )?;
        self.blas.sgemm_row_major(
            &w.normed,
            &self.weights.qkv_weight,
            &w.qkv_proj,
            1,
            d.q_out + 2 * d.kv_out,
            d.hidden,
        )?;
        let q_proj = w.qkv_proj.as_ptr_at(0)?;
        let k_proj = w.qkv_proj.as_ptr_at(d.q_out)?;
        let v_proj = w.qkv_proj.as_ptr_at(d.q_out + d.kv_out)?;
        launch_rmsnorm(
            &k.rmsnorm,
            q_proj,
            self.weights.q_gamma.as_ptr(),
            w.q_norm.as_mut_ptr(),
            d.q_heads,
            d.head_dim,
            d.epsilon,
        )?;
        launch_rmsnorm(
            &k.rmsnorm,
            k_proj,
            self.weights.k_gamma.as_ptr(),
            w.k_norm.as_mut_ptr(),
            d.kv_heads,
            d.head_dim,
            d.epsilon,
        )?;
        launch_rope(
            &k.rope,
            w.q_norm.as_ptr(),
            w.q_rope.as_mut_ptr(),
            d.q_out,
            d.q_heads,
            1,
            d.head_dim,
            offset,
            d.theta,
        )?;
        launch_rope(
            &k.rope,
            w.k_norm.as_ptr(),
            w.k_rope.as_mut_ptr(),
            d.kv_out,
            d.kv_heads,
            1,
            d.head_dim,
            offset,
            d.theta,
        )?;
        launch_write_cache(
            &k.write_cache,
            w.k_rope.as_ptr(),
            w.k_cache.as_mut_ptr(),
            d.kv_heads,
            1,
            d.max_cache_steps,
            d.head_dim,
            offset,
        )?;
        launch_write_cache(
            &k.write_cache,
            v_proj,
            w.v_cache.as_mut_ptr(),
            d.kv_heads,
            1,
            d.max_cache_steps,
            d.head_dim,
            offset,
        )?;
        launch_attention_scores_cache(
            &k.scores_cache,
            w.q_rope.as_ptr(),
            w.k_cache.as_ptr(),
            w.scores.as_mut_ptr(),
            d.q_heads,
            d.kv_heads,
            1,
            active_steps,
            d.max_cache_steps,
            d.head_dim,
            offset,
            d.scale,
        )?;
        launch_softmax(
            &k.softmax,
            w.scores.as_ptr(),
            w.probs.as_mut_ptr(),
            d.q_heads,
            active_steps,
        )?;
        launch_apply_value_cache(
            &k.apply_value_cache,
            w.probs.as_ptr(),
            w.v_cache.as_ptr(),
            w.attended.as_mut_ptr(),
            d.q_heads,
            d.kv_heads,
            1,
            active_steps,
            d.max_cache_steps,
            d.head_dim,
        )?;
        self.blas.sgemm_row_major(
            &w.attended,
            &self.weights.o_weight,
            &w.projected,
            1,
            d.hidden,
            d.q_out,
        )?;
        launch_ternary(
            &k.residual_add,
            input.as_ptr(),
            w.projected.as_ptr(),
            w.attention_output.as_mut_ptr(),
            d.hidden,
        )?;
        launch_rmsnorm(
            &k.rmsnorm,
            w.attention_output.as_ptr(),
            self.weights.post_attention_gamma.as_ptr(),
            w.post_norm.as_mut_ptr(),
            1,
            d.hidden,
            d.epsilon,
        )?;
        self.blas.sgemm_row_major(
            &w.post_norm,
            &self.weights.gate_up_weight,
            &w.gate_up,
            1,
            2 * d.intermediate,
            d.hidden,
        )?;
        let gate = w.gate_up.as_ptr_at(0)?;
        let up = w.gate_up.as_ptr_at(d.intermediate)?;
        launch_ternary(&k.swiglu, gate, up, w.swiglu.as_mut_ptr(), d.intermediate)?;
        self.blas.sgemm_row_major(
            &w.swiglu,
            &self.weights.down_weight,
            &w.mlp_down,
            1,
            d.hidden,
            d.intermediate,
        )?;
        launch_ternary(
            &k.residual_add,
            w.attention_output.as_ptr(),
            w.mlp_down.as_ptr(),
            output.as_mut_ptr(),
            d.hidden,
        )
    }
}

impl DecodeStepStack {
    pub fn load(
        runtime: &HipRuntime,
        model_dir: &Path,
        layer_count: usize,
        max_cache_steps: usize,
    ) -> Result<Self> {
        Self::load_with_prefix(
            runtime,
            model_dir,
            "talker.model.layers",
            layer_count,
            max_cache_steps,
            1e-6,
            10_000.0,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn load_with_prefix(
        runtime: &HipRuntime,
        model_dir: &Path,
        layer_prefix: &str,
        layer_count: usize,
        max_cache_steps: usize,
        epsilon: f32,
        theta: f32,
    ) -> Result<Self> {
        if layer_count == 0 {
            return Err(Error::InvalidInput(
                "layer count must be non-zero".to_string(),
            ));
        }
        let archive = TensorArchive::open(&model_dir.join("model.safetensors"))?;
        let (weights, caches, dims) = archive.with_tensors(|tensors| {
            let mut weights = Vec::with_capacity(layer_count);
            let mut caches = Vec::with_capacity(layer_count);
            let mut dims = None;
            for index in 0..layer_count {
                let host = load_host_weights_from_tensors(tensors, layer_prefix, index)?;
                let layer_dims = infer_dims(&host, max_cache_steps, epsilon, theta)?;
                if let Some(expected) = dims {
                    validate_stack_dims(index, layer_dims, expected)?;
                } else {
                    dims = Some(layer_dims);
                }
                weights.push(DecodeDeviceWeights::from_host(runtime, &host)?);
                caches.push(DecodeLayerCache::new(runtime, layer_dims)?);
            }
            Ok((weights, caches, dims.expect("layer count checked above")))
        })?;
        let storage = DecodeCacheStorage {
            dims,
            caches,
            workspace: DecodeWorkspace::new(runtime, dims)?,
            prefix_a: runtime.empty_buffer::<f32>(max_cache_steps * dims.hidden)?,
            prefix_b: runtime.empty_buffer::<f32>(max_cache_steps * dims.hidden)?,
        };
        Ok(Self {
            weights,
            blas: runtime.create_blas_handle()?,
            kernels: DecodeKernels::compile(runtime)?,
            runtime: runtime.clone(),
            storage: RefCell::new(storage),
            current_scratch: runtime.empty_buffer::<f32>(dims.hidden)?,
        })
    }

    fn ensure_cache_capacity(&self, required_steps: usize) -> Result<()> {
        let current_steps = self.storage.borrow().dims.max_cache_steps;
        if required_steps <= current_steps {
            return Ok(());
        }
        let target_steps = next_cache_capacity(current_steps, required_steps)?;
        self.runtime.synchronize()?;

        let current = self.storage.borrow();
        let mut dims = current.dims;
        dims.max_cache_steps = target_steps;
        let mut caches = Vec::with_capacity(current.caches.len());
        for cache in &current.caches {
            let expanded = DecodeLayerCache::new(&self.runtime, dims)?;
            copy_cache_heads(&expanded.k_cache, &cache.k_cache, current.dims, dims)?;
            copy_cache_heads(&expanded.v_cache, &cache.v_cache, current.dims, dims)?;
            caches.push(expanded);
        }
        let prefix_len = target_steps
            .checked_mul(dims.hidden)
            .ok_or_else(|| Error::InvalidInput("prefix buffer size overflow".to_string()))?;
        let replacement = DecodeCacheStorage {
            dims,
            caches,
            workspace: DecodeWorkspace::new(&self.runtime, dims)?,
            prefix_a: self.runtime.empty_buffer::<f32>(prefix_len)?,
            prefix_b: self.runtime.empty_buffer::<f32>(prefix_len)?,
        };
        drop(current);
        *self.storage.borrow_mut() = replacement;
        Ok(())
    }

    pub fn copy_prefill_step_to(&self, output: &DeviceBuffer<f32>, step: usize) -> Result<()> {
        let storage = self.storage.borrow();
        if output.len() != storage.dims.hidden {
            return Err(Error::InvalidInput(format!(
                "prefill step output length must be {}, got {}",
                storage.dims.hidden,
                output.len()
            )));
        }
        if step >= storage.dims.max_cache_steps {
            return Err(Error::InvalidInput(format!(
                "prefill step {step} outside cache length {}",
                storage.dims.max_cache_steps
            )));
        }
        let source = if (self.weights.len() - 1) % 2 == 0 {
            &storage.prefix_a
        } else {
            &storage.prefix_b
        };
        output.copy_from_device_range(source, step * storage.dims.hidden, storage.dims.hidden)
    }

    pub fn dims(&self) -> DecodeStepDims {
        self.storage.borrow().dims
    }

    pub fn input_len(&self) -> usize {
        self.storage.borrow().dims.hidden
    }

    pub fn prefill(&self, prefix: &DeviceBuffer<f32>, prefix_steps: usize) -> Result<()> {
        if prefix_steps == 0 {
            return Err(Error::InvalidInput(format!(
                "prefix steps must be non-zero, got {prefix_steps}"
            )));
        }
        self.ensure_cache_capacity(prefix_steps)?;
        let storage = self.storage.borrow();
        if prefix.len() < prefix_steps * storage.dims.hidden {
            return Err(Error::InvalidInput(format!(
                "prefix length {} is smaller than {}",
                prefix.len(),
                prefix_steps * storage.dims.hidden
            )));
        }

        for (index, (weights, cache)) in self.weights.iter().zip(storage.caches.iter()).enumerate()
        {
            match index {
                0 => self.prefill_forward(
                    &storage,
                    prefix,
                    &storage.prefix_a,
                    weights,
                    cache,
                    prefix_steps,
                )?,
                index if index % 2 == 1 => self.prefill_forward(
                    &storage,
                    &storage.prefix_a,
                    &storage.prefix_b,
                    weights,
                    cache,
                    prefix_steps,
                )?,
                _ => self.prefill_forward(
                    &storage,
                    &storage.prefix_b,
                    &storage.prefix_a,
                    weights,
                    cache,
                    prefix_steps,
                )?,
            }
        }
        Ok(())
    }

    pub fn decode_step_on_stream(
        &self,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        offset: usize,
        stream: &HipStream,
    ) -> Result<()> {
        let storage = self.storage.borrow();
        if input.len() != storage.dims.hidden || output.len() != storage.dims.hidden {
            return Err(Error::InvalidInput(format!(
                "decode input/output length must be {}, got input={}, output={}",
                storage.dims.hidden,
                input.len(),
                output.len()
            )));
        }
        if offset >= storage.dims.max_cache_steps {
            return Err(Error::InvalidInput(format!(
                "offset {offset} outside cache length {}",
                storage.dims.max_cache_steps
            )));
        }
        if self.weights.len() == 1 {
            return self.decode_step_layer_on_stream(
                &storage,
                input,
                output,
                &self.weights[0],
                &storage.caches[0],
                offset,
                stream,
            );
        }

        self.decode_step_layer_on_stream(
            &storage,
            input,
            &self.current_scratch,
            &self.weights[0],
            &storage.caches[0],
            offset,
            stream,
        )?;
        let mut current_is_scratch = true;
        for (weights, cache) in self.weights.iter().zip(storage.caches.iter()).skip(1) {
            if current_is_scratch {
                self.decode_step_layer_on_stream(
                    &storage,
                    &self.current_scratch,
                    output,
                    weights,
                    cache,
                    offset,
                    stream,
                )?;
            } else {
                self.decode_step_layer_on_stream(
                    &storage,
                    output,
                    &self.current_scratch,
                    weights,
                    cache,
                    offset,
                    stream,
                )?;
            }
            current_is_scratch = !current_is_scratch;
        }
        if current_is_scratch {
            output.copy_from_device_on_stream(&self.current_scratch, stream)?;
        }
        Ok(())
    }

    pub fn decode_step(
        &self,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        offset: usize,
    ) -> Result<()> {
        let required_steps = offset
            .checked_add(1)
            .ok_or_else(|| Error::InvalidInput("cache position overflow".to_string()))?;
        self.ensure_cache_capacity(required_steps)?;
        let storage = self.storage.borrow();
        if input.len() != storage.dims.hidden || output.len() != storage.dims.hidden {
            return Err(Error::InvalidInput(format!(
                "decode input/output length must be {}, got input={}, output={}",
                storage.dims.hidden,
                input.len(),
                output.len()
            )));
        }
        if self.weights.len() == 1 {
            return self.decode_step_layer(
                &storage,
                input,
                output,
                &self.weights[0],
                &storage.caches[0],
                offset,
            );
        }

        self.decode_step_layer(
            &storage,
            input,
            &self.current_scratch,
            &self.weights[0],
            &storage.caches[0],
            offset,
        )?;
        let mut current_is_scratch = true;
        for (weights, cache) in self.weights.iter().zip(storage.caches.iter()).skip(1) {
            if current_is_scratch {
                self.decode_step_layer(
                    &storage,
                    &self.current_scratch,
                    output,
                    weights,
                    cache,
                    offset,
                )?;
            } else {
                self.decode_step_layer(
                    &storage,
                    output,
                    &self.current_scratch,
                    weights,
                    cache,
                    offset,
                )?;
            }
            current_is_scratch = !current_is_scratch;
        }
        if current_is_scratch {
            output.copy_from_device(&self.current_scratch)?;
        }
        Ok(())
    }

    pub fn decode_step_profiled(
        &self,
        runtime: &HipRuntime,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        offset: usize,
    ) -> Result<DecodeStepProfile> {
        let required_steps = offset
            .checked_add(1)
            .ok_or_else(|| Error::InvalidInput("cache position overflow".to_string()))?;
        self.ensure_cache_capacity(required_steps)?;
        let storage = self.storage.borrow();
        if input.len() != storage.dims.hidden || output.len() != storage.dims.hidden {
            return Err(Error::InvalidInput(format!(
                "decode input/output length must be {}, got input={}, output={}",
                storage.dims.hidden,
                input.len(),
                output.len()
            )));
        }
        let mut profile = DecodeStepProfile::default();
        if self.weights.len() == 1 {
            profile.add(&self.decode_step_layer_profiled(
                &storage,
                runtime,
                input,
                output,
                &self.weights[0],
                &storage.caches[0],
                offset,
            )?);
            return Ok(profile);
        }

        profile.add(&self.decode_step_layer_profiled(
            &storage,
            runtime,
            input,
            &self.current_scratch,
            &self.weights[0],
            &storage.caches[0],
            offset,
        )?);
        let mut current_is_scratch = true;
        for (weights, cache) in self.weights.iter().zip(storage.caches.iter()).skip(1) {
            if current_is_scratch {
                profile.add(&self.decode_step_layer_profiled(
                    &storage,
                    runtime,
                    &self.current_scratch,
                    output,
                    weights,
                    cache,
                    offset,
                )?);
            } else {
                profile.add(&self.decode_step_layer_profiled(
                    &storage,
                    runtime,
                    output,
                    &self.current_scratch,
                    weights,
                    cache,
                    offset,
                )?);
            }
            current_is_scratch = !current_is_scratch;
        }
        if current_is_scratch {
            let start = Instant::now();
            output.copy_from_device(&self.current_scratch)?;
            runtime.synchronize()?;
            profile.final_copy_seconds += start.elapsed().as_secs_f64();
        }
        Ok(profile)
    }

    fn prefill_forward(
        &self,
        storage: &DecodeCacheStorage,
        prefix: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        weights: &DecodeDeviceWeights,
        cache: &DecodeLayerCache,
        prefix_steps: usize,
    ) -> Result<()> {
        let d = storage.dims;
        let w = &storage.workspace;
        let k = &self.kernels;
        launch_rmsnorm(
            &k.rmsnorm,
            prefix.as_ptr(),
            weights.input_gamma.as_ptr(),
            w.prefill_normed.as_mut_ptr(),
            prefix_steps,
            d.hidden,
            d.epsilon,
        )?;
        self.blas.sgemm_row_major(
            &w.prefill_normed,
            &weights.q_weight,
            &w.prefill_q_proj,
            prefix_steps,
            d.q_out,
            d.hidden,
        )?;
        self.blas.sgemm_row_major(
            &w.prefill_normed,
            &weights.k_weight,
            &w.prefill_k_proj,
            prefix_steps,
            d.kv_out,
            d.hidden,
        )?;
        self.blas.sgemm_row_major(
            &w.prefill_normed,
            &weights.v_weight,
            &w.prefill_v_proj,
            prefix_steps,
            d.kv_out,
            d.hidden,
        )?;
        launch_rmsnorm(
            &k.rmsnorm,
            w.prefill_q_proj.as_ptr(),
            weights.q_gamma.as_ptr(),
            w.prefill_q_norm.as_mut_ptr(),
            prefix_steps * d.q_heads,
            d.head_dim,
            d.epsilon,
        )?;
        launch_rmsnorm(
            &k.rmsnorm,
            w.prefill_k_proj.as_ptr(),
            weights.k_gamma.as_ptr(),
            w.prefill_k_norm.as_mut_ptr(),
            prefix_steps * d.kv_heads,
            d.head_dim,
            d.epsilon,
        )?;
        launch_permute(
            &k.permute,
            w.prefill_q_norm.as_ptr(),
            w.prefill_q_bhsd.as_mut_ptr(),
            prefix_steps,
            d.q_heads,
            d.head_dim,
        )?;
        launch_permute(
            &k.permute,
            w.prefill_k_norm.as_ptr(),
            w.prefill_k_bhsd.as_mut_ptr(),
            prefix_steps,
            d.kv_heads,
            d.head_dim,
        )?;
        launch_permute(
            &k.permute,
            w.prefill_v_proj.as_ptr(),
            w.prefill_v_bhsd.as_mut_ptr(),
            prefix_steps,
            d.kv_heads,
            d.head_dim,
        )?;
        launch_rope(
            &k.rope,
            w.prefill_q_bhsd.as_ptr(),
            w.prefill_q_rope.as_mut_ptr(),
            d.q_heads * prefix_steps * d.head_dim,
            d.q_heads,
            prefix_steps,
            d.head_dim,
            0,
            d.theta,
        )?;
        launch_rope(
            &k.rope,
            w.prefill_k_bhsd.as_ptr(),
            w.prefill_k_rope.as_mut_ptr(),
            d.kv_heads * prefix_steps * d.head_dim,
            d.kv_heads,
            prefix_steps,
            d.head_dim,
            0,
            d.theta,
        )?;
        launch_write_cache(
            &k.write_cache,
            w.prefill_k_rope.as_ptr(),
            cache.k_cache.as_mut_ptr(),
            d.kv_heads,
            prefix_steps,
            d.max_cache_steps,
            d.head_dim,
            0,
        )?;
        launch_write_cache(
            &k.write_cache,
            w.prefill_v_bhsd.as_ptr(),
            cache.v_cache.as_mut_ptr(),
            d.kv_heads,
            prefix_steps,
            d.max_cache_steps,
            d.head_dim,
            0,
        )?;
        launch_attention_scores(
            &k.scores,
            w.prefill_q_rope.as_ptr(),
            w.prefill_k_rope.as_ptr(),
            w.prefill_scores.as_mut_ptr(),
            d.q_heads,
            d.kv_heads,
            prefix_steps,
            prefix_steps,
            d.head_dim,
            0,
            d.scale,
        )?;
        launch_softmax(
            &k.softmax,
            w.prefill_scores.as_ptr(),
            w.prefill_probs.as_mut_ptr(),
            d.q_heads * prefix_steps,
            prefix_steps,
        )?;
        launch_apply_value(
            &k.apply_value,
            w.prefill_probs.as_ptr(),
            w.prefill_v_bhsd.as_ptr(),
            w.prefill_attended.as_mut_ptr(),
            d.q_heads,
            d.kv_heads,
            prefix_steps,
            prefix_steps,
            d.head_dim,
        )?;
        self.blas.sgemm_row_major(
            &w.prefill_attended,
            &weights.o_weight,
            &w.prefill_projected,
            prefix_steps,
            d.hidden,
            d.q_out,
        )?;
        launch_ternary(
            &k.residual_add,
            prefix.as_ptr(),
            w.prefill_projected.as_ptr(),
            w.prefill_attention_output.as_mut_ptr(),
            prefix_steps * d.hidden,
        )?;
        launch_rmsnorm(
            &k.rmsnorm,
            w.prefill_attention_output.as_ptr(),
            weights.post_attention_gamma.as_ptr(),
            w.prefill_post_norm.as_mut_ptr(),
            prefix_steps,
            d.hidden,
            d.epsilon,
        )?;
        self.blas.sgemm_row_major(
            &w.prefill_post_norm,
            &weights.gate_weight,
            &w.prefill_gate,
            prefix_steps,
            d.intermediate,
            d.hidden,
        )?;
        self.blas.sgemm_row_major(
            &w.prefill_post_norm,
            &weights.up_weight,
            &w.prefill_up,
            prefix_steps,
            d.intermediate,
            d.hidden,
        )?;
        launch_ternary(
            &k.swiglu,
            w.prefill_gate.as_ptr(),
            w.prefill_up.as_ptr(),
            w.prefill_swiglu.as_mut_ptr(),
            prefix_steps * d.intermediate,
        )?;
        self.blas.sgemm_row_major(
            &w.prefill_swiglu,
            &weights.down_weight,
            &w.prefill_mlp_down,
            prefix_steps,
            d.hidden,
            d.intermediate,
        )?;
        launch_ternary(
            &k.residual_add,
            w.prefill_attention_output.as_ptr(),
            w.prefill_mlp_down.as_ptr(),
            output.as_mut_ptr(),
            prefix_steps * d.hidden,
        )
    }

    fn decode_step_layer(
        &self,
        storage: &DecodeCacheStorage,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        weights: &DecodeDeviceWeights,
        cache: &DecodeLayerCache,
        offset: usize,
    ) -> Result<()> {
        let d = storage.dims;
        let w = &storage.workspace;
        let k = &self.kernels;
        let active_steps = offset + 1;
        launch_rmsnorm(
            &k.rmsnorm,
            input.as_ptr(),
            weights.input_gamma.as_ptr(),
            w.normed.as_mut_ptr(),
            1,
            d.hidden,
            d.epsilon,
        )?;
        self.blas.sgemm_row_major(
            &w.normed,
            &weights.qkv_weight,
            &w.qkv_proj,
            1,
            d.q_out + 2 * d.kv_out,
            d.hidden,
        )?;
        let q_proj = w.qkv_proj.as_ptr_at(0)?;
        let k_proj = w.qkv_proj.as_ptr_at(d.q_out)?;
        let v_proj = w.qkv_proj.as_ptr_at(d.q_out + d.kv_out)?;
        launch_rmsnorm(
            &k.rmsnorm,
            q_proj,
            weights.q_gamma.as_ptr(),
            w.q_norm.as_mut_ptr(),
            d.q_heads,
            d.head_dim,
            d.epsilon,
        )?;
        launch_rmsnorm(
            &k.rmsnorm,
            k_proj,
            weights.k_gamma.as_ptr(),
            w.k_norm.as_mut_ptr(),
            d.kv_heads,
            d.head_dim,
            d.epsilon,
        )?;
        launch_rope(
            &k.rope,
            w.q_norm.as_ptr(),
            w.q_rope.as_mut_ptr(),
            d.q_out,
            d.q_heads,
            1,
            d.head_dim,
            offset,
            d.theta,
        )?;
        launch_rope(
            &k.rope,
            w.k_norm.as_ptr(),
            w.k_rope.as_mut_ptr(),
            d.kv_out,
            d.kv_heads,
            1,
            d.head_dim,
            offset,
            d.theta,
        )?;
        launch_write_cache(
            &k.write_cache,
            w.k_rope.as_ptr(),
            cache.k_cache.as_mut_ptr(),
            d.kv_heads,
            1,
            d.max_cache_steps,
            d.head_dim,
            offset,
        )?;
        launch_write_cache(
            &k.write_cache,
            v_proj,
            cache.v_cache.as_mut_ptr(),
            d.kv_heads,
            1,
            d.max_cache_steps,
            d.head_dim,
            offset,
        )?;
        launch_attention_scores_cache(
            &k.scores_cache,
            w.q_rope.as_ptr(),
            cache.k_cache.as_ptr(),
            w.scores.as_mut_ptr(),
            d.q_heads,
            d.kv_heads,
            1,
            active_steps,
            d.max_cache_steps,
            d.head_dim,
            offset,
            d.scale,
        )?;
        launch_softmax(
            &k.softmax,
            w.scores.as_ptr(),
            w.probs.as_mut_ptr(),
            d.q_heads,
            active_steps,
        )?;
        launch_apply_value_cache(
            &k.apply_value_cache,
            w.probs.as_ptr(),
            cache.v_cache.as_ptr(),
            w.attended.as_mut_ptr(),
            d.q_heads,
            d.kv_heads,
            1,
            active_steps,
            d.max_cache_steps,
            d.head_dim,
        )?;
        self.blas.sgemm_row_major(
            &w.attended,
            &weights.o_weight,
            &w.projected,
            1,
            d.hidden,
            d.q_out,
        )?;
        launch_ternary(
            &k.residual_add,
            input.as_ptr(),
            w.projected.as_ptr(),
            w.attention_output.as_mut_ptr(),
            d.hidden,
        )?;
        launch_rmsnorm(
            &k.rmsnorm,
            w.attention_output.as_ptr(),
            weights.post_attention_gamma.as_ptr(),
            w.post_norm.as_mut_ptr(),
            1,
            d.hidden,
            d.epsilon,
        )?;
        self.blas.sgemm_row_major(
            &w.post_norm,
            &weights.gate_up_weight,
            &w.gate_up,
            1,
            2 * d.intermediate,
            d.hidden,
        )?;
        let gate = w.gate_up.as_ptr_at(0)?;
        let up = w.gate_up.as_ptr_at(d.intermediate)?;
        launch_ternary(&k.swiglu, gate, up, w.swiglu.as_mut_ptr(), d.intermediate)?;
        self.blas.sgemm_row_major(
            &w.swiglu,
            &weights.down_weight,
            &w.mlp_down,
            1,
            d.hidden,
            d.intermediate,
        )?;
        launch_ternary(
            &k.residual_add,
            w.attention_output.as_ptr(),
            w.mlp_down.as_ptr(),
            output.as_mut_ptr(),
            d.hidden,
        )
    }

    fn decode_step_layer_profiled(
        &self,
        storage: &DecodeCacheStorage,
        runtime: &HipRuntime,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        weights: &DecodeDeviceWeights,
        cache: &DecodeLayerCache,
        offset: usize,
    ) -> Result<DecodeStepProfile> {
        let d = storage.dims;
        let w = &storage.workspace;
        let k = &self.kernels;
        let active_steps = offset + 1;
        let mut profile = DecodeStepProfile::default();

        let start = Instant::now();
        launch_rmsnorm(
            &k.rmsnorm,
            input.as_ptr(),
            weights.input_gamma.as_ptr(),
            w.normed.as_mut_ptr(),
            1,
            d.hidden,
            d.epsilon,
        )?;
        runtime.synchronize()?;
        profile.input_norm_seconds += start.elapsed().as_secs_f64();

        let start = Instant::now();
        self.blas.sgemm_row_major(
            &w.normed,
            &weights.qkv_weight,
            &w.qkv_proj,
            1,
            d.q_out + 2 * d.kv_out,
            d.hidden,
        )?;
        runtime.synchronize()?;
        profile.qkv_gemm_seconds += start.elapsed().as_secs_f64();

        let q_proj = w.qkv_proj.as_ptr_at(0)?;
        let k_proj = w.qkv_proj.as_ptr_at(d.q_out)?;
        let v_proj = w.qkv_proj.as_ptr_at(d.q_out + d.kv_out)?;

        let start = Instant::now();
        launch_rmsnorm(
            &k.rmsnorm,
            q_proj,
            weights.q_gamma.as_ptr(),
            w.q_norm.as_mut_ptr(),
            d.q_heads,
            d.head_dim,
            d.epsilon,
        )?;
        launch_rmsnorm(
            &k.rmsnorm,
            k_proj,
            weights.k_gamma.as_ptr(),
            w.k_norm.as_mut_ptr(),
            d.kv_heads,
            d.head_dim,
            d.epsilon,
        )?;
        launch_rope(
            &k.rope,
            w.q_norm.as_ptr(),
            w.q_rope.as_mut_ptr(),
            d.q_out,
            d.q_heads,
            1,
            d.head_dim,
            offset,
            d.theta,
        )?;
        launch_rope(
            &k.rope,
            w.k_norm.as_ptr(),
            w.k_rope.as_mut_ptr(),
            d.kv_out,
            d.kv_heads,
            1,
            d.head_dim,
            offset,
            d.theta,
        )?;
        launch_write_cache(
            &k.write_cache,
            w.k_rope.as_ptr(),
            cache.k_cache.as_mut_ptr(),
            d.kv_heads,
            1,
            d.max_cache_steps,
            d.head_dim,
            offset,
        )?;
        launch_write_cache(
            &k.write_cache,
            v_proj,
            cache.v_cache.as_mut_ptr(),
            d.kv_heads,
            1,
            d.max_cache_steps,
            d.head_dim,
            offset,
        )?;
        runtime.synchronize()?;
        profile.qk_layout_cache_seconds += start.elapsed().as_secs_f64();

        let start = Instant::now();
        launch_attention_scores_cache(
            &k.scores_cache,
            w.q_rope.as_ptr(),
            cache.k_cache.as_ptr(),
            w.scores.as_mut_ptr(),
            d.q_heads,
            d.kv_heads,
            1,
            active_steps,
            d.max_cache_steps,
            d.head_dim,
            offset,
            d.scale,
        )?;
        launch_softmax(
            &k.softmax,
            w.scores.as_ptr(),
            w.probs.as_mut_ptr(),
            d.q_heads,
            active_steps,
        )?;
        launch_apply_value_cache(
            &k.apply_value_cache,
            w.probs.as_ptr(),
            cache.v_cache.as_ptr(),
            w.attended.as_mut_ptr(),
            d.q_heads,
            d.kv_heads,
            1,
            active_steps,
            d.max_cache_steps,
            d.head_dim,
        )?;
        runtime.synchronize()?;
        profile.attention_seconds += start.elapsed().as_secs_f64();

        let start = Instant::now();
        self.blas.sgemm_row_major(
            &w.attended,
            &weights.o_weight,
            &w.projected,
            1,
            d.hidden,
            d.q_out,
        )?;
        launch_ternary(
            &k.residual_add,
            input.as_ptr(),
            w.projected.as_ptr(),
            w.attention_output.as_mut_ptr(),
            d.hidden,
        )?;
        runtime.synchronize()?;
        profile.output_gemm_residual_seconds += start.elapsed().as_secs_f64();

        let start = Instant::now();
        launch_rmsnorm(
            &k.rmsnorm,
            w.attention_output.as_ptr(),
            weights.post_attention_gamma.as_ptr(),
            w.post_norm.as_mut_ptr(),
            1,
            d.hidden,
            d.epsilon,
        )?;
        runtime.synchronize()?;
        profile.post_norm_seconds += start.elapsed().as_secs_f64();

        let start = Instant::now();
        self.blas.sgemm_row_major(
            &w.post_norm,
            &weights.gate_up_weight,
            &w.gate_up,
            1,
            2 * d.intermediate,
            d.hidden,
        )?;
        runtime.synchronize()?;
        profile.gate_up_gemm_seconds += start.elapsed().as_secs_f64();

        let gate = w.gate_up.as_ptr_at(0)?;
        let up = w.gate_up.as_ptr_at(d.intermediate)?;
        let start = Instant::now();
        launch_ternary(&k.swiglu, gate, up, w.swiglu.as_mut_ptr(), d.intermediate)?;
        runtime.synchronize()?;
        profile.swiglu_seconds += start.elapsed().as_secs_f64();

        let start = Instant::now();
        self.blas.sgemm_row_major(
            &w.swiglu,
            &weights.down_weight,
            &w.mlp_down,
            1,
            d.hidden,
            d.intermediate,
        )?;
        launch_ternary(
            &k.residual_add,
            w.attention_output.as_ptr(),
            w.mlp_down.as_ptr(),
            output.as_mut_ptr(),
            d.hidden,
        )?;
        runtime.synchronize()?;
        profile.down_gemm_residual_seconds += start.elapsed().as_secs_f64();
        Ok(profile)
    }

    fn decode_step_layer_on_stream(
        &self,
        storage: &DecodeCacheStorage,
        input: &DeviceBuffer<f32>,
        output: &DeviceBuffer<f32>,
        weights: &DecodeDeviceWeights,
        cache: &DecodeLayerCache,
        offset: usize,
        stream: &HipStream,
    ) -> Result<()> {
        let d = storage.dims;
        let w = &storage.workspace;
        let k = &self.kernels;
        let active_steps = offset + 1;
        launch_rmsnorm_on_stream(
            &k.rmsnorm,
            input.as_ptr(),
            weights.input_gamma.as_ptr(),
            w.normed.as_mut_ptr(),
            1,
            d.hidden,
            d.epsilon,
            Some(stream),
        )?;
        self.blas.sgemm_row_major_on_stream(
            &w.normed,
            &weights.qkv_weight,
            &w.qkv_proj,
            1,
            d.q_out + 2 * d.kv_out,
            d.hidden,
            stream,
        )?;
        let q_proj = w.qkv_proj.as_ptr_at(0)?;
        let k_proj = w.qkv_proj.as_ptr_at(d.q_out)?;
        let v_proj = w.qkv_proj.as_ptr_at(d.q_out + d.kv_out)?;
        launch_rmsnorm_on_stream(
            &k.rmsnorm,
            q_proj,
            weights.q_gamma.as_ptr(),
            w.q_norm.as_mut_ptr(),
            d.q_heads,
            d.head_dim,
            d.epsilon,
            Some(stream),
        )?;
        launch_rmsnorm_on_stream(
            &k.rmsnorm,
            k_proj,
            weights.k_gamma.as_ptr(),
            w.k_norm.as_mut_ptr(),
            d.kv_heads,
            d.head_dim,
            d.epsilon,
            Some(stream),
        )?;
        launch_rope_on_stream(
            &k.rope,
            w.q_norm.as_ptr(),
            w.q_rope.as_mut_ptr(),
            d.q_out,
            d.q_heads,
            1,
            d.head_dim,
            offset,
            d.theta,
            Some(stream),
        )?;
        launch_rope_on_stream(
            &k.rope,
            w.k_norm.as_ptr(),
            w.k_rope.as_mut_ptr(),
            d.kv_out,
            d.kv_heads,
            1,
            d.head_dim,
            offset,
            d.theta,
            Some(stream),
        )?;
        launch_write_cache_on_stream(
            &k.write_cache,
            w.k_rope.as_ptr(),
            cache.k_cache.as_mut_ptr(),
            d.kv_heads,
            1,
            d.max_cache_steps,
            d.head_dim,
            offset,
            Some(stream),
        )?;
        launch_write_cache_on_stream(
            &k.write_cache,
            v_proj,
            cache.v_cache.as_mut_ptr(),
            d.kv_heads,
            1,
            d.max_cache_steps,
            d.head_dim,
            offset,
            Some(stream),
        )?;
        launch_attention_scores_cache_on_stream(
            &k.scores_cache,
            w.q_rope.as_ptr(),
            cache.k_cache.as_ptr(),
            w.scores.as_mut_ptr(),
            d.q_heads,
            d.kv_heads,
            1,
            active_steps,
            d.max_cache_steps,
            d.head_dim,
            offset,
            d.scale,
            Some(stream),
        )?;
        launch_softmax_on_stream(
            &k.softmax,
            w.scores.as_ptr(),
            w.probs.as_mut_ptr(),
            d.q_heads,
            active_steps,
            Some(stream),
        )?;
        launch_apply_value_cache_on_stream(
            &k.apply_value_cache,
            w.probs.as_ptr(),
            cache.v_cache.as_ptr(),
            w.attended.as_mut_ptr(),
            d.q_heads,
            d.kv_heads,
            1,
            active_steps,
            d.max_cache_steps,
            d.head_dim,
            Some(stream),
        )?;
        self.blas.sgemm_row_major_on_stream(
            &w.attended,
            &weights.o_weight,
            &w.projected,
            1,
            d.hidden,
            d.q_out,
            stream,
        )?;
        launch_ternary_on_stream(
            &k.residual_add,
            input.as_ptr(),
            w.projected.as_ptr(),
            w.attention_output.as_mut_ptr(),
            d.hidden,
            Some(stream),
        )?;
        launch_rmsnorm_on_stream(
            &k.rmsnorm,
            w.attention_output.as_ptr(),
            weights.post_attention_gamma.as_ptr(),
            w.post_norm.as_mut_ptr(),
            1,
            d.hidden,
            d.epsilon,
            Some(stream),
        )?;
        self.blas.sgemm_row_major_on_stream(
            &w.post_norm,
            &weights.gate_up_weight,
            &w.gate_up,
            1,
            2 * d.intermediate,
            d.hidden,
            stream,
        )?;
        let gate = w.gate_up.as_ptr_at(0)?;
        let up = w.gate_up.as_ptr_at(d.intermediate)?;
        launch_ternary_on_stream(
            &k.swiglu,
            gate,
            up,
            w.swiglu.as_mut_ptr(),
            d.intermediate,
            Some(stream),
        )?;
        self.blas.sgemm_row_major_on_stream(
            &w.swiglu,
            &weights.down_weight,
            &w.mlp_down,
            1,
            d.hidden,
            d.intermediate,
            stream,
        )?;
        launch_ternary_on_stream(
            &k.residual_add,
            w.attention_output.as_ptr(),
            w.mlp_down.as_ptr(),
            output.as_mut_ptr(),
            d.hidden,
            Some(stream),
        )
    }
}

fn copy_cache_heads(
    destination: &DeviceBuffer<f32>,
    source: &DeviceBuffer<f32>,
    source_dims: DecodeStepDims,
    destination_dims: DecodeStepDims,
) -> Result<()> {
    if source_dims.kv_heads != destination_dims.kv_heads
        || source_dims.head_dim != destination_dims.head_dim
        || source_dims.max_cache_steps > destination_dims.max_cache_steps
    {
        return Err(Error::InvalidInput(
            "invalid KV-cache growth dimensions".to_string(),
        ));
    }
    let source_head_len = source_dims
        .max_cache_steps
        .checked_mul(source_dims.head_dim)
        .ok_or_else(|| Error::InvalidInput("KV-cache head size overflow".to_string()))?;
    let destination_head_len = destination_dims
        .max_cache_steps
        .checked_mul(destination_dims.head_dim)
        .ok_or_else(|| Error::InvalidInput("KV-cache head size overflow".to_string()))?;
    for head in 0..source_dims.kv_heads {
        destination.copy_from_device_range_at(
            head * destination_head_len,
            source,
            head * source_head_len,
            source_head_len,
        )?;
    }
    Ok(())
}

fn next_cache_capacity(current_steps: usize, required_steps: usize) -> Result<usize> {
    if current_steps == 0 {
        return Err(Error::InvalidInput(
            "cache capacity must be non-zero".to_string(),
        ));
    }
    if required_steps <= current_steps {
        return Ok(current_steps);
    }
    let doubled = current_steps.checked_mul(2).unwrap_or(required_steps);
    if required_steps <= doubled {
        Ok(doubled)
    } else {
        Ok(required_steps
            .checked_next_power_of_two()
            .unwrap_or(required_steps))
    }
}

#[cfg(test)]
mod cache_growth_tests {
    use super::*;

    #[test]
    fn cache_capacity_doubles_until_it_covers_required_steps() {
        assert_eq!(next_cache_capacity(16, 16).unwrap(), 16);
        assert_eq!(next_cache_capacity(16, 17).unwrap(), 32);
        assert_eq!(next_cache_capacity(16, 33).unwrap(), 64);
    }

    #[test]
    fn cache_capacity_rejects_zero_initial_size() {
        assert!(next_cache_capacity(0, 1).is_err());
    }
}

impl DecodeLayerCache {
    fn new(runtime: &HipRuntime, dims: DecodeStepDims) -> Result<Self> {
        Ok(Self {
            k_cache: runtime
                .empty_buffer::<f32>(dims.kv_heads * dims.max_cache_steps * dims.head_dim)?,
            v_cache: runtime
                .empty_buffer::<f32>(dims.kv_heads * dims.max_cache_steps * dims.head_dim)?,
        })
    }
}

fn validate_stack_dims(
    index: usize,
    actual: DecodeStepDims,
    expected: DecodeStepDims,
) -> Result<()> {
    if actual.hidden != expected.hidden
        || actual.q_heads != expected.q_heads
        || actual.kv_heads != expected.kv_heads
        || actual.head_dim != expected.head_dim
        || actual.intermediate != expected.intermediate
        || actual.max_cache_steps != expected.max_cache_steps
    {
        return Err(Error::InvalidInput(format!(
            "decode layer {index} dims mismatch: {actual:?} != {expected:?}"
        )));
    }
    Ok(())
}

fn concat_row_major_columns(parts: &[(&[f32], usize)], rows: usize) -> Result<Vec<f32>> {
    let cols = parts.iter().map(|(_, cols)| *cols).sum::<usize>();
    let mut output = vec![0.0; rows * cols];
    for (part, part_cols) in parts {
        if part.len() != rows * *part_cols {
            return Err(Error::InvalidInput(format!(
                "cannot concatenate row-major matrix with length {}, rows={rows}, cols={part_cols}",
                part.len()
            )));
        }
    }
    for row in 0..rows {
        let mut dst_col = 0usize;
        for (part, part_cols) in parts {
            let src_start = row * *part_cols;
            let dst_start = row * cols + dst_col;
            output[dst_start..dst_start + *part_cols]
                .copy_from_slice(&part[src_start..src_start + *part_cols]);
            dst_col += *part_cols;
        }
    }
    Ok(output)
}

impl DecodeDeviceWeights {
    fn from_host(runtime: &HipRuntime, weights: &DecodeHostWeights) -> Result<Self> {
        Ok(Self {
            input_gamma: runtime.buffer_from_slice(&weights.input_gamma)?,
            q_gamma: runtime.buffer_from_slice(&weights.q_gamma)?,
            k_gamma: runtime.buffer_from_slice(&weights.k_gamma)?,
            q_weight: runtime.buffer_from_slice(&weights.q_weight)?,
            k_weight: runtime.buffer_from_slice(&weights.k_weight)?,
            v_weight: runtime.buffer_from_slice(&weights.v_weight)?,
            qkv_weight: runtime.buffer_from_slice(&concat_row_major_columns(
                &[
                    (
                        &weights.q_weight,
                        weights.q_weight.len() / weights.input_gamma.len(),
                    ),
                    (
                        &weights.k_weight,
                        weights.k_weight.len() / weights.input_gamma.len(),
                    ),
                    (
                        &weights.v_weight,
                        weights.v_weight.len() / weights.input_gamma.len(),
                    ),
                ],
                weights.input_gamma.len(),
            )?)?,
            o_weight: runtime.buffer_from_slice(&weights.o_weight)?,
            post_attention_gamma: runtime.buffer_from_slice(&weights.post_attention_gamma)?,
            gate_weight: runtime.buffer_from_slice(&weights.gate_weight)?,
            up_weight: runtime.buffer_from_slice(&weights.up_weight)?,
            gate_up_weight: runtime.buffer_from_slice(&concat_row_major_columns(
                &[
                    (
                        &weights.gate_weight,
                        weights.gate_weight.len() / weights.input_gamma.len(),
                    ),
                    (
                        &weights.up_weight,
                        weights.up_weight.len() / weights.input_gamma.len(),
                    ),
                ],
                weights.input_gamma.len(),
            )?)?,
            down_weight: runtime.buffer_from_slice(&weights.down_weight)?,
        })
    }
}

impl DecodeKernels {
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
        let scores_cache = attention_module.function("attention_scores_cache_f32")?;
        let apply_value = attention_module.function("attention_apply_value_f32")?;
        let apply_value_cache = attention_module.function("attention_apply_value_cache_f32")?;
        let write_cache = attention_module.function("write_kv_cache_f32")?;
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
            scores_cache,
            apply_value,
            apply_value_cache,
            write_cache,
            residual_add,
            swiglu,
        })
    }
}

impl DecodeWorkspace {
    fn new(runtime: &HipRuntime, d: DecodeStepDims) -> Result<Self> {
        Ok(Self {
            prefill_normed: runtime.empty_buffer::<f32>(d.max_cache_steps * d.hidden)?,
            prefill_q_proj: runtime.empty_buffer::<f32>(d.max_cache_steps * d.q_out)?,
            prefill_k_proj: runtime.empty_buffer::<f32>(d.max_cache_steps * d.kv_out)?,
            prefill_v_proj: runtime.empty_buffer::<f32>(d.max_cache_steps * d.kv_out)?,
            prefill_q_norm: runtime.empty_buffer::<f32>(d.max_cache_steps * d.q_out)?,
            prefill_k_norm: runtime.empty_buffer::<f32>(d.max_cache_steps * d.kv_out)?,
            prefill_q_bhsd: runtime
                .empty_buffer::<f32>(d.q_heads * d.max_cache_steps * d.head_dim)?,
            prefill_k_bhsd: runtime
                .empty_buffer::<f32>(d.kv_heads * d.max_cache_steps * d.head_dim)?,
            prefill_v_bhsd: runtime
                .empty_buffer::<f32>(d.kv_heads * d.max_cache_steps * d.head_dim)?,
            prefill_q_rope: runtime
                .empty_buffer::<f32>(d.q_heads * d.max_cache_steps * d.head_dim)?,
            prefill_k_rope: runtime
                .empty_buffer::<f32>(d.kv_heads * d.max_cache_steps * d.head_dim)?,
            prefill_scores: runtime
                .empty_buffer::<f32>(d.q_heads * d.max_cache_steps * d.max_cache_steps)?,
            prefill_probs: runtime
                .empty_buffer::<f32>(d.q_heads * d.max_cache_steps * d.max_cache_steps)?,
            prefill_attended: runtime
                .empty_buffer::<f32>(d.max_cache_steps * d.q_heads * d.head_dim)?,
            prefill_projected: runtime.empty_buffer::<f32>(d.max_cache_steps * d.hidden)?,
            prefill_attention_output: runtime.empty_buffer::<f32>(d.max_cache_steps * d.hidden)?,
            prefill_post_norm: runtime.empty_buffer::<f32>(d.max_cache_steps * d.hidden)?,
            prefill_gate: runtime.empty_buffer::<f32>(d.max_cache_steps * d.intermediate)?,
            prefill_up: runtime.empty_buffer::<f32>(d.max_cache_steps * d.intermediate)?,
            prefill_swiglu: runtime.empty_buffer::<f32>(d.max_cache_steps * d.intermediate)?,
            prefill_mlp_down: runtime.empty_buffer::<f32>(d.max_cache_steps * d.hidden)?,
            k_cache: runtime.empty_buffer::<f32>(d.kv_heads * d.max_cache_steps * d.head_dim)?,
            v_cache: runtime.empty_buffer::<f32>(d.kv_heads * d.max_cache_steps * d.head_dim)?,
            normed: runtime.empty_buffer::<f32>(d.hidden)?,
            qkv_proj: runtime.empty_buffer::<f32>(d.q_out + 2 * d.kv_out)?,
            q_norm: runtime.empty_buffer::<f32>(d.q_out)?,
            k_norm: runtime.empty_buffer::<f32>(d.kv_out)?,
            q_rope: runtime.empty_buffer::<f32>(d.q_out)?,
            k_rope: runtime.empty_buffer::<f32>(d.kv_out)?,
            scores: runtime.empty_buffer::<f32>(d.q_heads * d.max_cache_steps)?,
            probs: runtime.empty_buffer::<f32>(d.q_heads * d.max_cache_steps)?,
            attended: runtime.empty_buffer::<f32>(d.q_out)?,
            projected: runtime.empty_buffer::<f32>(d.hidden)?,
            attention_output: runtime.empty_buffer::<f32>(d.hidden)?,
            post_norm: runtime.empty_buffer::<f32>(d.hidden)?,
            gate_up: runtime.empty_buffer::<f32>(2 * d.intermediate)?,
            swiglu: runtime.empty_buffer::<f32>(d.intermediate)?,
            mlp_down: runtime.empty_buffer::<f32>(d.hidden)?,
        })
    }
}

fn load_host_weights(model_dir: &Path, layer_index: usize) -> Result<DecodeHostWeights> {
    let archive = TensorArchive::open(&model_dir.join("model.safetensors"))?;
    let prefix = format!("talker.model.layers.{layer_index}");
    Ok(DecodeHostWeights {
        input_gamma: archive.vector_f32(&format!("{prefix}.input_layernorm.weight"))?,
        q_gamma: archive.vector_f32(&format!("{prefix}.self_attn.q_norm.weight"))?,
        k_gamma: archive.vector_f32(&format!("{prefix}.self_attn.k_norm.weight"))?,
        q_weight: archive
            .linear_weight_transposed_f32(&format!("{prefix}.self_attn.q_proj.weight"))?
            .0,
        k_weight: archive
            .linear_weight_transposed_f32(&format!("{prefix}.self_attn.k_proj.weight"))?
            .0,
        v_weight: archive
            .linear_weight_transposed_f32(&format!("{prefix}.self_attn.v_proj.weight"))?
            .0,
        o_weight: archive
            .linear_weight_transposed_f32(&format!("{prefix}.self_attn.o_proj.weight"))?
            .0,
        post_attention_gamma: archive
            .vector_f32(&format!("{prefix}.post_attention_layernorm.weight"))?,
        gate_weight: archive
            .linear_weight_transposed_f32(&format!("{prefix}.mlp.gate_proj.weight"))?
            .0,
        up_weight: archive
            .linear_weight_transposed_f32(&format!("{prefix}.mlp.up_proj.weight"))?
            .0,
        down_weight: archive
            .linear_weight_transposed_f32(&format!("{prefix}.mlp.down_proj.weight"))?
            .0,
    })
}

fn load_host_weights_from_tensors(
    tensors: &SafeTensors<'_>,
    layer_prefix: &str,
    layer_index: usize,
) -> Result<DecodeHostWeights> {
    let prefix = format!("{layer_prefix}.{layer_index}");
    Ok(DecodeHostWeights {
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

fn infer_dims(
    weights: &DecodeHostWeights,
    max_cache_steps: usize,
    epsilon: f32,
    theta: f32,
) -> Result<DecodeStepDims> {
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
    Ok(DecodeStepDims {
        hidden,
        q_heads: q_out / head_dim,
        kv_heads: kv_out / head_dim,
        head_dim,
        q_out,
        kv_out,
        intermediate,
        max_cache_steps,
        epsilon,
        theta,
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
) -> Result<()> {
    launch_rmsnorm_on_stream(function, input, gamma, output, rows, cols, epsilon, None)
}

#[allow(clippy::too_many_arguments)]
fn launch_rmsnorm_on_stream(
    function: &HipFunction,
    input: *const c_void,
    gamma: *const c_void,
    output: *mut c_void,
    rows: usize,
    cols: usize,
    epsilon: f32,
    stream: Option<&HipStream>,
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
    function.launch_on_stream(
        (rows as u32, 1, 1),
        (block, 1, 1),
        block * 4,
        &mut params,
        stream,
    )
}

fn launch_permute(
    function: &HipFunction,
    input: *const c_void,
    output: *mut c_void,
    steps: usize,
    heads: usize,
    head_dim: usize,
) -> Result<()> {
    launch_permute_on_stream(function, input, output, steps, heads, head_dim, None)
}

fn launch_permute_on_stream(
    function: &HipFunction,
    input: *const c_void,
    output: *mut c_void,
    steps: usize,
    heads: usize,
    head_dim: usize,
    stream: Option<&HipStream>,
) -> Result<()> {
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
    function.launch_on_stream(
        ((total as u32).div_ceil(block), 1, 1),
        (block, 1, 1),
        0,
        &mut params,
        stream,
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
    launch_rope_on_stream(
        function, input, output, total, heads, steps, head_dim, offset, theta, None,
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_rope_on_stream(
    function: &HipFunction,
    input: *const c_void,
    output: *mut c_void,
    total: usize,
    heads: usize,
    steps: usize,
    head_dim: usize,
    offset: usize,
    theta: f32,
    stream: Option<&HipStream>,
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
    function.launch_on_stream(
        ((total as u32).div_ceil(block), 1, 1),
        (block, 1, 1),
        0,
        &mut params,
        stream,
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
) -> Result<()> {
    launch_write_cache_on_stream(
        function,
        input,
        cache,
        heads,
        input_steps,
        cache_steps,
        head_dim,
        offset,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_write_cache_on_stream(
    function: &HipFunction,
    input: *const c_void,
    cache: *mut c_void,
    heads: usize,
    input_steps: usize,
    cache_steps: usize,
    head_dim: usize,
    offset: usize,
    stream: Option<&HipStream>,
) -> Result<()> {
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
    function.launch_on_stream(
        ((total as u32).div_ceil(block), 1, 1),
        (block, 1, 1),
        0,
        &mut params,
        stream,
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
) -> Result<()> {
    launch_attention_scores_on_stream(
        function,
        q,
        k,
        scores,
        q_heads,
        kv_heads,
        query_steps,
        key_steps,
        head_dim,
        offset,
        scale,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_attention_scores_on_stream(
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
    stream: Option<&HipStream>,
) -> Result<()> {
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
    function.launch_on_stream(
        ((total as u32).div_ceil(block), 1, 1),
        (block, 1, 1),
        0,
        &mut params,
        stream,
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_attention_scores_cache(
    function: &HipFunction,
    q: *const c_void,
    k: *const c_void,
    scores: *mut c_void,
    q_heads: usize,
    kv_heads: usize,
    query_steps: usize,
    key_steps: usize,
    cache_steps: usize,
    head_dim: usize,
    offset: usize,
    scale: f32,
) -> Result<()> {
    launch_attention_scores_cache_on_stream(
        function,
        q,
        k,
        scores,
        q_heads,
        kv_heads,
        query_steps,
        key_steps,
        cache_steps,
        head_dim,
        offset,
        scale,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_attention_scores_cache_on_stream(
    function: &HipFunction,
    q: *const c_void,
    k: *const c_void,
    scores: *mut c_void,
    q_heads: usize,
    kv_heads: usize,
    query_steps: usize,
    key_steps: usize,
    cache_steps: usize,
    head_dim: usize,
    offset: usize,
    scale: f32,
    stream: Option<&HipStream>,
) -> Result<()> {
    let total = q_heads * query_steps * key_steps;
    let mut q = q;
    let mut k = k;
    let mut scores = scores;
    let mut batch = 1i32;
    let mut q_heads_i32 = q_heads as i32;
    let mut kv_heads_i32 = kv_heads as i32;
    let mut query_steps_i32 = query_steps as i32;
    let mut key_steps_i32 = key_steps as i32;
    let mut cache_steps_i32 = cache_steps as i32;
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
        &mut cache_steps_i32 as *mut i32 as *mut c_void,
        &mut head_dim_i32 as *mut i32 as *mut c_void,
        &mut offset_i32 as *mut i32 as *mut c_void,
        &mut scale as *mut f32 as *mut c_void,
        &mut total_i32 as *mut i32 as *mut c_void,
    ];
    let block = 256u32;
    function.launch_on_stream(
        ((total as u32).div_ceil(block), 1, 1),
        (block, 1, 1),
        0,
        &mut params,
        stream,
    )
}

fn launch_softmax(
    function: &HipFunction,
    input: *const c_void,
    output: *mut c_void,
    rows: usize,
    cols: usize,
) -> Result<()> {
    launch_softmax_on_stream(function, input, output, rows, cols, None)
}

fn launch_softmax_on_stream(
    function: &HipFunction,
    input: *const c_void,
    output: *mut c_void,
    rows: usize,
    cols: usize,
    stream: Option<&HipStream>,
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
    function.launch_on_stream(
        (rows as u32, 1, 1),
        (block, 1, 1),
        block * 4,
        &mut params,
        stream,
    )
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
) -> Result<()> {
    launch_apply_value_on_stream(
        function,
        probs,
        v,
        output,
        q_heads,
        kv_heads,
        query_steps,
        key_steps,
        head_dim,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_apply_value_on_stream(
    function: &HipFunction,
    probs: *const c_void,
    v: *const c_void,
    output: *mut c_void,
    q_heads: usize,
    kv_heads: usize,
    query_steps: usize,
    key_steps: usize,
    head_dim: usize,
    stream: Option<&HipStream>,
) -> Result<()> {
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
    function.launch_on_stream(
        ((total as u32).div_ceil(block), 1, 1),
        (block, 1, 1),
        0,
        &mut params,
        stream,
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_apply_value_cache(
    function: &HipFunction,
    probs: *const c_void,
    v: *const c_void,
    output: *mut c_void,
    q_heads: usize,
    kv_heads: usize,
    query_steps: usize,
    key_steps: usize,
    cache_steps: usize,
    head_dim: usize,
) -> Result<()> {
    launch_apply_value_cache_on_stream(
        function,
        probs,
        v,
        output,
        q_heads,
        kv_heads,
        query_steps,
        key_steps,
        cache_steps,
        head_dim,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn launch_apply_value_cache_on_stream(
    function: &HipFunction,
    probs: *const c_void,
    v: *const c_void,
    output: *mut c_void,
    q_heads: usize,
    kv_heads: usize,
    query_steps: usize,
    key_steps: usize,
    cache_steps: usize,
    head_dim: usize,
    stream: Option<&HipStream>,
) -> Result<()> {
    let total = query_steps * q_heads * head_dim;
    let mut probs = probs;
    let mut v = v;
    let mut output = output;
    let mut batch = 1i32;
    let mut q_heads_i32 = q_heads as i32;
    let mut kv_heads_i32 = kv_heads as i32;
    let mut query_steps_i32 = query_steps as i32;
    let mut key_steps_i32 = key_steps as i32;
    let mut cache_steps_i32 = cache_steps as i32;
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
        &mut cache_steps_i32 as *mut i32 as *mut c_void,
        &mut head_dim_i32 as *mut i32 as *mut c_void,
        &mut total_i32 as *mut i32 as *mut c_void,
    ];
    let block = 256u32;
    function.launch_on_stream(
        ((total as u32).div_ceil(block), 1, 1),
        (block, 1, 1),
        0,
        &mut params,
        stream,
    )
}

fn launch_ternary(
    function: &HipFunction,
    input_a: *const c_void,
    input_b: *const c_void,
    output: *mut c_void,
    total: usize,
) -> Result<()> {
    launch_ternary_on_stream(function, input_a, input_b, output, total, None)
}

fn launch_ternary_on_stream(
    function: &HipFunction,
    input_a: *const c_void,
    input_b: *const c_void,
    output: *mut c_void,
    total: usize,
    stream: Option<&HipStream>,
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
    function.launch_on_stream(
        ((total as u32).div_ceil(block), 1, 1),
        (block, 1, 1),
        0,
        &mut params,
        stream,
    )
}
