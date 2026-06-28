use std::ffi::c_void;
use std::path::Path;

use safetensors::tensor::TensorView;

use crate::blas::RocblasHandle;
use crate::buffer::DeviceBuffer;
use crate::config::Qwen3TtsConfig;
use crate::decode::DecodeStepStack;
use crate::error::{Error, Result};
use crate::kernel::{HipFunction, HipModule};
use crate::kernels::{
    ARGMAX_F32_SOURCE, ELEMENTWISE_F32_SOURCE, EMBEDDING_F32_SOURCE, RMSNORM_F32_SOURCE,
    SUPPRESSION_F32_SOURCE,
};
use crate::model::sampling::{SamplingConfig, select_token};
use crate::runtime::HipRuntime;
use crate::weights::{TensorArchive, tensor_to_f32};

pub struct TalkerPrefillOutput {
    pub hidden: Vec<f32>,
    pub logits: Vec<f32>,
    pub suppressed_logits: Vec<f32>,
    pub semantic_token: i32,
}

pub struct HipTalker {
    hidden_size: usize,
    vocab_size: usize,
    codec_eos_token: usize,
    epsilon: f32,
    stack: DecodeStepStack,
    blas: RocblasHandle,
    kernels: TalkerKernels,
    norm_gamma: DeviceBuffer<f32>,
    codec_head: DeviceBuffer<f32>,
    codec_embedding: DeviceBuffer<f32>,
    pre_norm: DeviceBuffer<f32>,
    hidden: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    suppressed_logits: DeviceBuffer<f32>,
    token: DeviceBuffer<i32>,
    semantic_embedding: DeviceBuffer<f32>,
    step_temp: DeviceBuffer<f32>,
    step_input: DeviceBuffer<f32>,
}

struct TalkerKernels {
    _rms_module: HipModule,
    _argmax_module: HipModule,
    _suppression_module: HipModule,
    _embedding_module: HipModule,
    _elementwise_module: HipModule,
    rmsnorm: HipFunction,
    argmax: HipFunction,
    suppress: HipFunction,
    repetition_penalty: HipFunction,
    embedding_lookup: HipFunction,
    residual_add: HipFunction,
}

impl HipTalker {
    pub fn load(runtime: &HipRuntime, model_dir: &Path, max_cache_steps: usize) -> Result<Self> {
        let config = Qwen3TtsConfig::load(model_dir)?;
        let stack = DecodeStepStack::load_with_prefix(
            runtime,
            model_dir,
            "talker.model.layers",
            config.talker.layers,
            max_cache_steps,
            config.talker.rms_eps,
            config.talker.rope_theta,
        )?;
        let dims = stack.dims();
        if dims.hidden != config.talker.hidden {
            return Err(Error::InvalidInput(format!(
                "talker hidden {} does not match config hidden {}",
                dims.hidden, config.talker.hidden
            )));
        }
        if dims.intermediate != config.talker.intermediate
            || dims.q_heads != config.talker.q_heads
            || dims.kv_heads != config.talker.kv_heads
            || dims.head_dim != config.talker.head_dim
        {
            return Err(Error::InvalidInput(format!(
                "talker stack dims {:?} do not match config {:?}",
                dims, config.talker
            )));
        }
        let hidden_size = config.talker.hidden;
        let vocab_size = config.talker.vocab;
        let archive = TensorArchive::open(&model_dir.join("model.safetensors"))?;
        let norm_gamma = archive.vector_f32("talker.model.norm.weight")?;
        let (codec_head, head_in, vocab) =
            archive.linear_weight_transposed_f32("talker.codec_head.weight")?;
        let codec_embedding = embedding_table_f32(
            archive.tensor("talker.model.codec_embedding.weight")?,
            vocab_size,
            hidden_size,
        )?;
        if norm_gamma.len() != hidden_size || head_in != hidden_size || vocab != vocab_size {
            return Err(Error::InvalidInput(format!(
                "invalid talker head shapes: norm={}, head_in={head_in}, vocab={vocab}",
                norm_gamma.len()
            )));
        }

        Ok(Self {
            hidden_size,
            vocab_size,
            codec_eos_token: config.tokens.codec_eos,
            epsilon: config.talker.rms_eps,
            stack,
            blas: runtime.create_blas_handle()?,
            kernels: TalkerKernels::compile(runtime)?,
            norm_gamma: runtime.buffer_from_slice(&norm_gamma)?,
            codec_head: runtime.buffer_from_slice(&codec_head)?,
            codec_embedding: runtime.buffer_from_slice(&codec_embedding)?,
            pre_norm: runtime.empty_buffer::<f32>(hidden_size)?,
            hidden: runtime.empty_buffer::<f32>(hidden_size)?,
            logits: runtime.empty_buffer::<f32>(vocab_size)?,
            suppressed_logits: runtime.empty_buffer::<f32>(vocab_size)?,
            token: runtime.empty_buffer::<i32>(1)?,
            semantic_embedding: runtime.empty_buffer::<f32>(hidden_size)?,
            step_temp: runtime.empty_buffer::<f32>(hidden_size)?,
            step_input: runtime.empty_buffer::<f32>(hidden_size)?,
        })
    }

    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn prefill_from_host(
        &self,
        runtime: &HipRuntime,
        prefill: &[f32],
        steps: usize,
    ) -> Result<TalkerPrefillOutput> {
        if prefill.len() != steps * self.hidden_size {
            return Err(Error::InvalidInput(format!(
                "prefill length {} does not match steps*hidden {}",
                prefill.len(),
                steps * self.hidden_size
            )));
        }
        let prefill = runtime.buffer_from_slice(prefill)?;
        self.prefill(&prefill, steps)
    }

    pub fn prefill(
        &self,
        prefill: &DeviceBuffer<f32>,
        steps: usize,
    ) -> Result<TalkerPrefillOutput> {
        if steps == 0 {
            return Err(Error::InvalidInput(
                "prefill steps must be non-zero".to_string(),
            ));
        }
        self.stack.prefill(prefill, steps)?;
        self.stack.copy_prefill_step_to(&self.pre_norm, steps - 1)?;
        launch_rmsnorm(
            &self.kernels.rmsnorm,
            self.pre_norm.as_ptr(),
            self.norm_gamma.as_ptr(),
            self.hidden.as_mut_ptr(),
            self.hidden_size,
            self.epsilon,
        )?;
        self.blas.sgemm_row_major(
            &self.hidden,
            &self.codec_head,
            &self.logits,
            1,
            self.vocab_size,
            self.hidden_size,
        )?;
        launch_suppress(
            &self.kernels.suppress,
            &self.logits,
            &self.suppressed_logits,
            self.vocab_size,
            self.codec_eos_token,
        )?;
        launch_argmax(
            &self.kernels.argmax,
            &self.suppressed_logits,
            &self.token,
            self.vocab_size,
        )?;
        let token = self.token.copy_to_host()?[0];
        Ok(TalkerPrefillOutput {
            hidden: self.hidden.copy_to_host()?,
            logits: self.logits.copy_to_host()?,
            suppressed_logits: self.suppressed_logits.copy_to_host()?,
            semantic_token: token,
        })
    }

    pub fn prefill_token(&self, prefill: &DeviceBuffer<f32>, steps: usize) -> Result<i32> {
        if steps == 0 {
            return Err(Error::InvalidInput(
                "prefill steps must be non-zero".to_string(),
            ));
        }
        self.stack.prefill(prefill, steps)?;
        self.stack.copy_prefill_step_to(&self.pre_norm, steps - 1)?;
        self.compute_logits_and_token()
    }

    pub(crate) fn prefill_token_with_sampling(
        &self,
        prefill: &DeviceBuffer<f32>,
        steps: usize,
        sampling: SamplingConfig,
        rng_state: &mut u64,
    ) -> Result<i32> {
        if steps == 0 {
            return Err(Error::InvalidInput(
                "prefill steps must be non-zero".to_string(),
            ));
        }
        self.stack.prefill(prefill, steps)?;
        self.stack.copy_prefill_step_to(&self.pre_norm, steps - 1)?;
        self.compute_logits_and_token_with_options(None, 1.0, sampling, rng_state)
    }

    pub fn decode_step_from_host(
        &self,
        runtime: &HipRuntime,
        input: &[f32],
        offset: usize,
    ) -> Result<TalkerPrefillOutput> {
        if input.len() != self.hidden_size {
            return Err(Error::InvalidInput(format!(
                "decode input length {} does not match hidden {}",
                input.len(),
                self.hidden_size
            )));
        }
        let input = runtime.buffer_from_slice(input)?;
        self.decode_step(&input, offset)
    }

    pub fn decode_step(
        &self,
        input: &DeviceBuffer<f32>,
        offset: usize,
    ) -> Result<TalkerPrefillOutput> {
        self.stack.decode_step(input, &self.pre_norm, offset)?;
        launch_rmsnorm(
            &self.kernels.rmsnorm,
            self.pre_norm.as_ptr(),
            self.norm_gamma.as_ptr(),
            self.hidden.as_mut_ptr(),
            self.hidden_size,
            self.epsilon,
        )?;
        self.blas.sgemm_row_major(
            &self.hidden,
            &self.codec_head,
            &self.logits,
            1,
            self.vocab_size,
            self.hidden_size,
        )?;
        launch_suppress(
            &self.kernels.suppress,
            &self.logits,
            &self.suppressed_logits,
            self.vocab_size,
            self.codec_eos_token,
        )?;
        launch_argmax(
            &self.kernels.argmax,
            &self.suppressed_logits,
            &self.token,
            self.vocab_size,
        )?;
        let token = self.token.copy_to_host()?[0];
        Ok(TalkerPrefillOutput {
            hidden: self.hidden.copy_to_host()?,
            logits: self.logits.copy_to_host()?,
            suppressed_logits: self.suppressed_logits.copy_to_host()?,
            semantic_token: token,
        })
    }

    pub fn prepare_code_predictor_prefix(&self, output: &DeviceBuffer<f32>) -> Result<()> {
        if output.len() != 2 * self.hidden_size {
            return Err(Error::InvalidInput(format!(
                "CodePredictor prefix output length must be {}, got {}",
                2 * self.hidden_size,
                output.len()
            )));
        }
        launch_embedding_lookup(
            &self.kernels.embedding_lookup,
            &self.codec_embedding,
            &self.token,
            &self.semantic_embedding,
            self.hidden_size,
        )?;
        output.copy_from_device_range_at(0, &self.hidden, 0, self.hidden_size)?;
        output.copy_from_device_range_at(
            self.hidden_size,
            &self.semantic_embedding,
            0,
            self.hidden_size,
        )
    }

    pub fn build_step_input(
        &self,
        acoustic_embedding_sum: &DeviceBuffer<f32>,
        trailing_text: &DeviceBuffer<f32>,
    ) -> Result<()> {
        if acoustic_embedding_sum.len() != self.hidden_size
            || trailing_text.len() != self.hidden_size
        {
            return Err(Error::InvalidInput(format!(
                "step input parts must have length {}, got acoustic={}, trailing={}",
                self.hidden_size,
                acoustic_embedding_sum.len(),
                trailing_text.len()
            )));
        }
        launch_embedding_lookup(
            &self.kernels.embedding_lookup,
            &self.codec_embedding,
            &self.token,
            &self.semantic_embedding,
            self.hidden_size,
        )?;
        launch_residual_add(
            &self.kernels.residual_add,
            self.semantic_embedding.as_ptr(),
            acoustic_embedding_sum.as_ptr(),
            self.step_temp.as_mut_ptr(),
            self.hidden_size,
        )?;
        launch_residual_add(
            &self.kernels.residual_add,
            self.step_temp.as_ptr(),
            trailing_text.as_ptr(),
            self.step_input.as_mut_ptr(),
            self.hidden_size,
        )
    }

    pub fn decode_prepared_step(&self, offset: usize) -> Result<TalkerPrefillOutput> {
        self.decode_step(&self.step_input, offset)
    }

    pub fn decode_prepared_token(&self, offset: usize) -> Result<i32> {
        self.stack
            .decode_step(&self.step_input, &self.pre_norm, offset)?;
        self.compute_logits_and_token()
    }

    pub fn decode_prepared_token_with_repetition(
        &self,
        offset: usize,
        previous_tokens: &DeviceBuffer<i32>,
        repetition_penalty: f32,
    ) -> Result<i32> {
        if repetition_penalty <= 0.0 {
            return Err(Error::InvalidInput(
                "repetition_penalty must be positive".to_string(),
            ));
        }
        self.stack
            .decode_step(&self.step_input, &self.pre_norm, offset)?;
        self.compute_logits_and_token_with_options(
            Some(previous_tokens),
            repetition_penalty,
            SamplingConfig {
                do_sample: false,
                top_k: 0,
                top_p: 1.0,
                temperature: 1.0,
            },
            &mut 0,
        )
    }

    pub(crate) fn decode_prepared_token_with_options(
        &self,
        offset: usize,
        previous_tokens: &DeviceBuffer<i32>,
        repetition_penalty: f32,
        sampling: SamplingConfig,
        rng_state: &mut u64,
    ) -> Result<i32> {
        if repetition_penalty <= 0.0 {
            return Err(Error::InvalidInput(
                "repetition_penalty must be positive".to_string(),
            ));
        }
        self.stack
            .decode_step(&self.step_input, &self.pre_norm, offset)?;
        self.compute_logits_and_token_with_options(
            Some(previous_tokens),
            repetition_penalty,
            sampling,
            rng_state,
        )
    }

    fn compute_logits_and_token(&self) -> Result<i32> {
        self.compute_logits_and_token_with_options(
            None,
            1.0,
            SamplingConfig {
                do_sample: false,
                top_k: 0,
                top_p: 1.0,
                temperature: 1.0,
            },
            &mut 0,
        )
    }

    fn compute_logits_and_token_with_options(
        &self,
        previous_tokens: Option<&DeviceBuffer<i32>>,
        repetition_penalty: f32,
        sampling: SamplingConfig,
        rng_state: &mut u64,
    ) -> Result<i32> {
        launch_rmsnorm(
            &self.kernels.rmsnorm,
            self.pre_norm.as_ptr(),
            self.norm_gamma.as_ptr(),
            self.hidden.as_mut_ptr(),
            self.hidden_size,
            self.epsilon,
        )?;
        self.blas.sgemm_row_major(
            &self.hidden,
            &self.codec_head,
            &self.logits,
            1,
            self.vocab_size,
            self.hidden_size,
        )?;
        let needs_host_selection =
            sampling.do_sample || previous_tokens.is_some_and(|tokens| !tokens.is_empty());
        if let Some(previous_tokens) = previous_tokens.filter(|tokens| !tokens.is_empty()) {
            launch_repetition_penalty(
                &self.kernels.repetition_penalty,
                &self.logits,
                previous_tokens,
                self.vocab_size,
                repetition_penalty,
            )?;
        }
        launch_suppress(
            &self.kernels.suppress,
            &self.logits,
            &self.suppressed_logits,
            self.vocab_size,
            self.codec_eos_token,
        )?;
        if needs_host_selection {
            let logits = self.suppressed_logits.copy_to_host()?;
            let token = select_token(&logits, sampling, rng_state)?;
            self.token.copy_from_host(&[token])?;
            return Ok(token);
        }
        launch_argmax(
            &self.kernels.argmax,
            &self.suppressed_logits,
            &self.token,
            self.vocab_size,
        )?;
        Ok(self.token.copy_to_host()?[0])
    }
}

impl TalkerKernels {
    fn compile(runtime: &HipRuntime) -> Result<Self> {
        let rms_module = runtime.compile_module("talker_rmsnorm_f32.cpp", RMSNORM_F32_SOURCE)?;
        let argmax_module = runtime.compile_module("talker_argmax_f32.cpp", ARGMAX_F32_SOURCE)?;
        let suppression_module =
            runtime.compile_module("talker_suppression_f32.cpp", SUPPRESSION_F32_SOURCE)?;
        let embedding_module =
            runtime.compile_module("talker_embedding_f32.cpp", EMBEDDING_F32_SOURCE)?;
        let elementwise_module =
            runtime.compile_module("talker_elementwise_f32.cpp", ELEMENTWISE_F32_SOURCE)?;
        Ok(Self {
            rmsnorm: rms_module.function("rmsnorm_f32")?,
            argmax: argmax_module.function("argmax_rows_f32")?,
            suppress: suppression_module.function("suppress_codec_logits_f32")?,
            repetition_penalty: suppression_module.function("apply_repetition_penalty_f32")?,
            embedding_lookup: embedding_module.function("embedding_lookup_f32")?,
            residual_add: elementwise_module.function("residual_add_f32")?,
            _rms_module: rms_module,
            _argmax_module: argmax_module,
            _suppression_module: suppression_module,
            _embedding_module: embedding_module,
            _elementwise_module: elementwise_module,
        })
    }
}

fn embedding_table_f32(
    tensor: TensorView<'_>,
    vocab_size: usize,
    hidden: usize,
) -> Result<Vec<f32>> {
    let shape = tensor.shape();
    if shape.len() != 2 || shape[0] != vocab_size || shape[1] != hidden {
        return Err(Error::InvalidInput(format!(
            "talker.model.codec_embedding.weight shape {shape:?}; expected [{vocab_size}, {hidden}]"
        )));
    }
    tensor_to_f32(
        "talker.model.codec_embedding.weight",
        tensor.dtype(),
        tensor.data(),
        vocab_size * hidden,
    )
}

fn launch_rmsnorm(
    function: &HipFunction,
    input: *const c_void,
    gamma: *const c_void,
    output: *mut c_void,
    cols: usize,
    epsilon: f32,
) -> Result<()> {
    let mut input = input;
    let mut gamma = gamma;
    let mut output = output;
    let mut rows_i32 = 1i32;
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
    function.launch((1, 1, 1), (block, 1, 1), block * 4, &mut params)
}

fn launch_suppress(
    function: &HipFunction,
    input: &DeviceBuffer<f32>,
    output: &DeviceBuffer<f32>,
    vocab_size: usize,
    eos_token: usize,
) -> Result<()> {
    let mut input_ptr = input.as_ptr();
    let mut output_ptr = output.as_mut_ptr();
    let mut vocab_i32 = vocab_size as i32;
    let mut eos_i32 = eos_token as i32;
    let mut params = [
        &mut input_ptr as *mut *const c_void as *mut c_void,
        &mut output_ptr as *mut *mut c_void as *mut c_void,
        &mut vocab_i32 as *mut i32 as *mut c_void,
        &mut eos_i32 as *mut i32 as *mut c_void,
    ];
    let block = 256u32;
    let grid = (vocab_size as u32).div_ceil(block);
    function.launch((grid, 1, 1), (block, 1, 1), 0, &mut params)
}

fn launch_repetition_penalty(
    function: &HipFunction,
    logits: &DeviceBuffer<f32>,
    tokens: &DeviceBuffer<i32>,
    vocab_size: usize,
    penalty: f32,
) -> Result<()> {
    let mut logits_ptr = logits.as_mut_ptr();
    let mut tokens_ptr = tokens.as_ptr();
    let mut token_count_i32 = tokens.len() as i32;
    let mut vocab_size_i32 = vocab_size as i32;
    let mut penalty = penalty;
    let mut params = [
        &mut logits_ptr as *mut *mut c_void as *mut c_void,
        &mut tokens_ptr as *mut *const c_void as *mut c_void,
        &mut token_count_i32 as *mut i32 as *mut c_void,
        &mut vocab_size_i32 as *mut i32 as *mut c_void,
        &mut penalty as *mut f32 as *mut c_void,
    ];
    function.launch((1, 1, 1), (1, 1, 1), 0, &mut params)
}

fn launch_embedding_lookup(
    function: &HipFunction,
    table: &DeviceBuffer<f32>,
    token: &DeviceBuffer<i32>,
    output: &DeviceBuffer<f32>,
    cols: usize,
) -> Result<()> {
    let mut table_ptr = table.as_ptr();
    let mut token_ptr = token.as_ptr();
    let mut output_ptr = output.as_mut_ptr();
    let mut cols_i32 = cols as i32;
    let mut params = [
        &mut table_ptr as *mut *const c_void as *mut c_void,
        &mut token_ptr as *mut *const c_void as *mut c_void,
        &mut output_ptr as *mut *mut c_void as *mut c_void,
        &mut cols_i32 as *mut i32 as *mut c_void,
    ];
    let block = 256u32;
    let grid = (cols as u32).div_ceil(block);
    function.launch((grid, 1, 1), (block, 1, 1), 0, &mut params)
}

fn launch_residual_add(
    function: &HipFunction,
    residual: *const c_void,
    update: *const c_void,
    output: *mut c_void,
    total: usize,
) -> Result<()> {
    let mut residual = residual;
    let mut update = update;
    let mut output = output;
    let mut total_i32 = total as i32;
    let mut params = [
        &mut residual as *mut *const c_void as *mut c_void,
        &mut update as *mut *const c_void as *mut c_void,
        &mut output as *mut *mut c_void as *mut c_void,
        &mut total_i32 as *mut i32 as *mut c_void,
    ];
    let block = 256u32;
    let grid = (total as u32).div_ceil(block);
    function.launch((grid, 1, 1), (block, 1, 1), 0, &mut params)
}

fn launch_argmax(
    function: &HipFunction,
    input: &DeviceBuffer<f32>,
    output: &DeviceBuffer<i32>,
    cols: usize,
) -> Result<()> {
    let mut input_ptr = input.as_ptr();
    let mut output_ptr = output.as_mut_ptr();
    let mut rows_i32 = 1i32;
    let mut cols_i32 = cols as i32;
    let mut params = [
        &mut input_ptr as *mut *const c_void as *mut c_void,
        &mut output_ptr as *mut *mut c_void as *mut c_void,
        &mut rows_i32 as *mut i32 as *mut c_void,
        &mut cols_i32 as *mut i32 as *mut c_void,
    ];
    let block = 256u32;
    let shared = block * (std::mem::size_of::<f32>() + std::mem::size_of::<i32>()) as u32;
    function.launch((1, 1, 1), (block, 1, 1), shared, &mut params)
}
