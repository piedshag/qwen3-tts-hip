use std::cell::RefCell;
use std::ffi::c_void;
use std::path::Path;

use safetensors::tensor::TensorView;

use crate::blas::RocblasHandle;
use crate::buffer::DeviceBuffer;
use crate::config::Qwen3TtsConfig;
use crate::decode::DecodeStepStack;
use crate::error::{Error, Result};
use crate::graph::{HipGraphExec, HipStream};
use crate::kernel::{HipFunction, HipModule};
use crate::kernels::{
    ARGMAX_F32_SOURCE, EMBEDDING_F32_SOURCE, RMSNORM_F32_SOURCE, SUPPRESSION_F32_SOURCE,
};
use crate::model::sampling::{SamplingConfig, next_f32, select_token};
use crate::runtime::HipRuntime;
use crate::weights::{TensorArchive, tensor_to_f32};
use std::time::Instant;

const CODE_PREDICTOR_PREFIX: &str = "talker.code_predictor";

#[derive(Clone, Debug)]
pub struct CodePredictorConfig {
    pub talker_hidden: usize,
    pub layer_count: usize,
    pub num_code_groups: usize,
    pub vocab_size: usize,
    pub epsilon: f32,
    pub theta: f32,
}

impl Default for CodePredictorConfig {
    fn default() -> Self {
        Self {
            talker_hidden: 1024,
            layer_count: 5,
            num_code_groups: 16,
            vocab_size: 2048,
            epsilon: 1e-6,
            theta: 1_000_000.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct CodePrediction {
    pub acoustic_tokens: Vec<i32>,
    pub embedding_sum: Vec<f32>,
}

#[derive(Clone, Debug, Default)]
pub struct CodePredictorProfile {
    pub prefix_projection_seconds: f64,
    pub stack_prefill_seconds: f64,
    pub first_logits_seconds: f64,
    pub first_token_seconds: f64,
    pub remaining_projection_seconds: f64,
    pub remaining_stack_seconds: f64,
    pub remaining_logits_seconds: f64,
    pub remaining_token_seconds: f64,
    pub output_copy_seconds: f64,
}

impl CodePredictorProfile {
    pub fn total_seconds(&self) -> f64 {
        self.prefix_projection_seconds
            + self.stack_prefill_seconds
            + self.first_logits_seconds
            + self.first_token_seconds
            + self.remaining_projection_seconds
            + self.remaining_stack_seconds
            + self.remaining_logits_seconds
            + self.remaining_token_seconds
            + self.output_copy_seconds
    }
}

pub struct HipCodePredictor {
    config: CodePredictorConfig,
    hidden: usize,
    talker_hidden: usize,
    stack: DecodeStepStack,
    blas: RocblasHandle,
    kernels: CodePredictorKernels,
    projection_weight: Option<DeviceBuffer<f32>>,
    projection_bias: Option<DeviceBuffer<f32>>,
    norm_gamma: DeviceBuffer<f32>,
    lm_heads: Vec<DeviceBuffer<f32>>,
    codec_embeddings: Vec<DeviceBuffer<f32>>,
    prefill_hidden: DeviceBuffer<f32>,
    step_hidden: DeviceBuffer<f32>,
    projected_prefix: DeviceBuffer<f32>,
    projected_embedding: DeviceBuffer<f32>,
    normed: DeviceBuffer<f32>,
    logits: DeviceBuffer<f32>,
    token: DeviceBuffer<i32>,
    tokens: DeviceBuffer<i32>,
    current_embedding: DeviceBuffer<f32>,
    embedding_sum: DeviceBuffer<f32>,
    sample_random_values: DeviceBuffer<f32>,
    graph_stream: HipStream,
    remaining_groups_graph: RefCell<Option<HipGraphExec>>,
    sampled_remaining_groups_graph: RefCell<Option<SampledGroupsGraph>>,
}

struct SampledGroupsGraph {
    sampling: SamplingConfig,
    graph: HipGraphExec,
}

struct CodePredictorKernels {
    _rms_module: HipModule,
    _argmax_module: HipModule,
    _sampling_module: HipModule,
    _embedding_module: HipModule,
    rmsnorm: HipFunction,
    argmax: HipFunction,
    sample_topk: HipFunction,
    sample_topk_random_buffer: HipFunction,
    embedding_lookup: HipFunction,
    store_token: HipFunction,
    residual_add: HipFunction,
    bias_add: HipFunction,
    _elementwise_module: HipModule,
}

impl HipCodePredictor {
    pub fn load(runtime: &HipRuntime, model_dir: &Path) -> Result<Self> {
        let config = Qwen3TtsConfig::load(model_dir)?;
        Self::load_with_config(
            runtime,
            model_dir,
            CodePredictorConfig {
                talker_hidden: config.talker.hidden,
                layer_count: config.code_predictor.layers,
                num_code_groups: config.code_predictor.num_code_groups,
                vocab_size: config.code_predictor.vocab,
                epsilon: config.code_predictor.rms_eps,
                theta: config.code_predictor.rope_theta,
            },
        )
    }

    pub fn load_with_config(
        runtime: &HipRuntime,
        model_dir: &Path,
        config: CodePredictorConfig,
    ) -> Result<Self> {
        if config.num_code_groups < 2 {
            return Err(Error::InvalidInput(
                "CodePredictor needs at least one semantic and one acoustic group".to_string(),
            ));
        }
        let num_acoustic = config.num_code_groups - 1;
        let vocab_size = config.vocab_size;
        let talker_hidden = config.talker_hidden;
        let stack = DecodeStepStack::load_with_prefix(
            runtime,
            model_dir,
            &format!("{CODE_PREDICTOR_PREFIX}.model.layers"),
            config.layer_count,
            config.num_code_groups,
            config.epsilon,
            config.theta,
        )?;
        let hidden = stack.dims().hidden;
        if hidden == 0 || config.talker_hidden == 0 {
            return Err(Error::InvalidInput(
                "CodePredictor hidden sizes must be non-zero".to_string(),
            ));
        }
        let archive = TensorArchive::open(&model_dir.join("model.safetensors"))?;
        let norm_gamma =
            archive.vector_f32(&format!("{CODE_PREDICTOR_PREFIX}.model.norm.weight"))?;
        if norm_gamma.len() != hidden {
            return Err(Error::InvalidInput(format!(
                "CodePredictor norm width {} does not match stack hidden {hidden}",
                norm_gamma.len()
            )));
        }

        let (projection_weight, projection_bias) = if config.talker_hidden != hidden {
            let (weight, in_dim, out_dim) = archive.linear_weight_transposed_f32(&format!(
                "{CODE_PREDICTOR_PREFIX}.small_to_mtp_projection.weight"
            ))?;
            if in_dim != config.talker_hidden || out_dim != hidden {
                return Err(Error::InvalidInput(format!(
                    "small_to_mtp_projection.weight shape out={out_dim}, in={in_dim}; expected out={hidden}, in={}",
                    config.talker_hidden
                )));
            }
            let bias = archive.vector_f32(&format!(
                "{CODE_PREDICTOR_PREFIX}.small_to_mtp_projection.bias"
            ))?;
            if bias.len() != hidden {
                return Err(Error::InvalidInput(format!(
                    "small_to_mtp_projection.bias len {} does not match hidden {hidden}",
                    bias.len()
                )));
            }
            (
                Some(runtime.buffer_from_slice(&weight)?),
                Some(runtime.buffer_from_slice(&bias)?),
            )
        } else {
            (None, None)
        };

        let mut lm_heads = Vec::with_capacity(num_acoustic);
        let mut codec_embeddings = Vec::with_capacity(num_acoustic);
        for group in 0..num_acoustic {
            let (head, head_in, vocab) = archive.linear_weight_transposed_f32(&format!(
                "{CODE_PREDICTOR_PREFIX}.lm_head.{group}.weight"
            ))?;
            if head_in != hidden || vocab != vocab_size {
                return Err(Error::InvalidInput(format!(
                    "lm_head.{group} shape hidden={head_in}, vocab={vocab}; expected hidden={hidden}, vocab={}",
                    vocab_size
                )));
            }
            lm_heads.push(runtime.buffer_from_slice(&head)?);

            let embedding = embedding_table_f32(
                archive.tensor(&format!(
                    "{CODE_PREDICTOR_PREFIX}.model.codec_embedding.{group}.weight"
                ))?,
                vocab_size,
                talker_hidden,
                group,
            )?;
            codec_embeddings.push(runtime.buffer_from_slice(&embedding)?);
        }

        Ok(Self {
            config,
            hidden,
            talker_hidden,
            stack,
            blas: runtime.create_blas_handle()?,
            kernels: CodePredictorKernels::compile(runtime)?,
            projection_weight,
            projection_bias,
            norm_gamma: runtime.buffer_from_slice(&norm_gamma)?,
            lm_heads,
            codec_embeddings,
            prefill_hidden: runtime.empty_buffer::<f32>(hidden)?,
            step_hidden: runtime.empty_buffer::<f32>(hidden)?,
            projected_prefix: runtime.empty_buffer::<f32>(2 * hidden)?,
            projected_embedding: runtime.empty_buffer::<f32>(hidden)?,
            normed: runtime.empty_buffer::<f32>(hidden)?,
            logits: runtime.empty_buffer::<f32>(vocab_size)?,
            token: runtime.empty_buffer::<i32>(1)?,
            tokens: runtime.empty_buffer::<i32>(num_acoustic)?,
            current_embedding: runtime.empty_buffer::<f32>(talker_hidden)?,
            embedding_sum: runtime.empty_buffer::<f32>(talker_hidden)?,
            sample_random_values: runtime.empty_buffer::<f32>(num_acoustic)?,
            graph_stream: runtime.create_stream()?,
            remaining_groups_graph: RefCell::new(None),
            sampled_remaining_groups_graph: RefCell::new(None),
        })
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }

    pub fn talker_hidden(&self) -> usize {
        self.talker_hidden
    }

    pub fn num_acoustic_groups(&self) -> usize {
        self.config.num_code_groups - 1
    }

    pub fn generate_from_host(
        &self,
        talker_hidden: &[f32],
        semantic_embed: &[f32],
    ) -> Result<CodePrediction> {
        if talker_hidden.len() != self.talker_hidden || semantic_embed.len() != self.talker_hidden {
            return Err(Error::InvalidInput(format!(
                "CodePredictor inputs must have length {}, got talker_hidden={}, semantic_embed={}",
                self.talker_hidden,
                talker_hidden.len(),
                semantic_embed.len()
            )));
        }
        let mut prefix = Vec::with_capacity(2 * self.hidden);
        prefix.extend_from_slice(talker_hidden);
        prefix.extend_from_slice(semantic_embed);
        let prefix = self.stack_device_from_prefix(&prefix)?;
        self.generate(&prefix)
    }

    pub fn generate(&self, prefix: &DeviceBuffer<f32>) -> Result<CodePrediction> {
        self.generate_inner(prefix)?;
        Ok(CodePrediction {
            acoustic_tokens: self.tokens.copy_to_host()?,
            embedding_sum: self.embedding_sum.copy_to_host()?,
        })
    }

    pub fn generate_to_buffer(
        &self,
        prefix: &DeviceBuffer<f32>,
        embedding_sum: &DeviceBuffer<f32>,
    ) -> Result<Vec<i32>> {
        if embedding_sum.len() != self.talker_hidden {
            return Err(Error::InvalidInput(format!(
                "embedding_sum output length must be {}, got {}",
                self.talker_hidden,
                embedding_sum.len()
            )));
        }
        self.generate_inner(prefix)?;
        embedding_sum.copy_from_device(&self.embedding_sum)?;
        self.tokens.copy_to_host()
    }

    pub(crate) fn generate_to_buffer_with_options(
        &self,
        prefix: &DeviceBuffer<f32>,
        embedding_sum: &DeviceBuffer<f32>,
        sampling: SamplingConfig,
        rng_state: &mut u64,
    ) -> Result<Vec<i32>> {
        if embedding_sum.len() != self.talker_hidden {
            return Err(Error::InvalidInput(format!(
                "embedding_sum output length must be {}, got {}",
                self.talker_hidden,
                embedding_sum.len()
            )));
        }
        if sampling.do_sample {
            self.generate_inner_sampled(prefix, sampling, rng_state)?;
        } else {
            self.generate_inner(prefix)?;
        }
        embedding_sum.copy_from_device(&self.embedding_sum)?;
        self.tokens.copy_to_host()
    }

    pub fn generate_to_buffer_profiled(
        &self,
        runtime: &HipRuntime,
        prefix: &DeviceBuffer<f32>,
        embedding_sum: &DeviceBuffer<f32>,
    ) -> Result<(Vec<i32>, CodePredictorProfile)> {
        if embedding_sum.len() != self.talker_hidden {
            return Err(Error::InvalidInput(format!(
                "embedding_sum output length must be {}, got {}",
                self.talker_hidden,
                embedding_sum.len()
            )));
        }
        let mut profile = self.generate_inner_profiled(runtime, prefix)?;
        let start = Instant::now();
        embedding_sum.copy_from_device(&self.embedding_sum)?;
        let tokens = self.tokens.copy_to_host()?;
        profile.output_copy_seconds += start.elapsed().as_secs_f64();
        Ok((tokens, profile))
    }

    fn generate_inner(&self, prefix: &DeviceBuffer<f32>) -> Result<()> {
        if prefix.len() != 2 * self.talker_hidden {
            return Err(Error::InvalidInput(format!(
                "CodePredictor prefix length must be {}, got {}",
                2 * self.talker_hidden,
                prefix.len()
            )));
        }
        let num_acoustic = self.num_acoustic_groups();
        let stack_prefix = self.project_to_stack(prefix, &self.projected_prefix, 2)?;
        self.stack.prefill(stack_prefix, 2)?;
        self.stack.copy_prefill_step_to(&self.prefill_hidden, 1)?;
        self.logits_for_hidden(&self.prefill_hidden, 0)?;
        launch_argmax(
            &self.kernels.argmax,
            &self.logits,
            &self.token,
            self.config.vocab_size,
        )?;
        launch_store_token(&self.kernels.store_token, &self.token, &self.tokens, 0)?;
        launch_embedding_lookup(
            &self.kernels.embedding_lookup,
            &self.codec_embeddings[0],
            &self.token,
            &self.current_embedding,
            self.talker_hidden,
        )?;
        self.embedding_sum
            .copy_from_device(&self.current_embedding)?;

        if num_acoustic > 1 {
            let _ = self.token.copy_to_host()?;
            self.generate_remaining_groups_graph()?;
        }
        Ok(())
    }

    fn generate_inner_sampled(
        &self,
        prefix: &DeviceBuffer<f32>,
        sampling: SamplingConfig,
        rng_state: &mut u64,
    ) -> Result<()> {
        if prefix.len() != 2 * self.talker_hidden {
            return Err(Error::InvalidInput(format!(
                "CodePredictor prefix length must be {}, got {}",
                2 * self.talker_hidden,
                prefix.len()
            )));
        }
        let num_acoustic = self.num_acoustic_groups();
        let stack_prefix = self.project_to_stack(prefix, &self.projected_prefix, 2)?;
        self.stack.prefill(stack_prefix, 2)?;
        self.stack.copy_prefill_step_to(&self.prefill_hidden, 1)?;
        self.logits_for_hidden(&self.prefill_hidden, 0)?;
        self.sample_current_token_on_device(0, sampling, rng_state)?;
        launch_embedding_lookup(
            &self.kernels.embedding_lookup,
            &self.codec_embeddings[0],
            &self.token,
            &self.current_embedding,
            self.talker_hidden,
        )?;
        self.embedding_sum
            .copy_from_device(&self.current_embedding)?;

        if num_acoustic > 1 {
            if sampling.supports_device_sampling() {
                let mut random_values = vec![0.0f32; num_acoustic];
                for value in random_values.iter_mut().skip(1) {
                    *value = next_f32(rng_state);
                }
                self.sample_random_values.copy_from_host(&random_values)?;
                let _ = self.token.copy_to_host()?;
                self.generate_remaining_groups_sampled_graph(sampling)?;
            } else {
                for group in 1..num_acoustic {
                    let stack_embedding = self.project_to_stack(
                        &self.current_embedding,
                        &self.projected_embedding,
                        1,
                    )?;
                    self.stack
                        .decode_step(stack_embedding, &self.step_hidden, group + 1)?;
                    self.logits_for_hidden(&self.step_hidden, group)?;
                    self.sample_current_token_on_device(group, sampling, rng_state)?;
                    launch_embedding_lookup(
                        &self.kernels.embedding_lookup,
                        &self.codec_embeddings[group],
                        &self.token,
                        &self.current_embedding,
                        self.talker_hidden,
                    )?;
                    launch_residual_add_on_stream(
                        &self.kernels.residual_add,
                        self.embedding_sum.as_ptr(),
                        self.current_embedding.as_ptr(),
                        self.embedding_sum.as_mut_ptr(),
                        self.talker_hidden,
                        None,
                    )?;
                }
            }
        }
        Ok(())
    }

    fn sample_current_token(
        &self,
        group: usize,
        sampling: SamplingConfig,
        rng_state: &mut u64,
    ) -> Result<()> {
        let logits = self.logits.copy_to_host()?;
        let token = select_token(&logits, sampling, rng_state)?;
        self.token.copy_from_host(&[token])?;
        launch_store_token(&self.kernels.store_token, &self.token, &self.tokens, group)
    }

    fn sample_current_token_on_device(
        &self,
        group: usize,
        sampling: SamplingConfig,
        rng_state: &mut u64,
    ) -> Result<()> {
        if sampling.supports_device_sampling() {
            let random_value = next_f32(rng_state);
            launch_sample_topk(
                &self.kernels.sample_topk,
                &self.logits,
                &self.token,
                self.config.vocab_size,
                sampling.top_k,
                sampling.temperature,
                random_value,
            )?;
            launch_store_token(&self.kernels.store_token, &self.token, &self.tokens, group)
        } else {
            self.sample_current_token(group, sampling, rng_state)
        }
    }

    fn generate_inner_profiled(
        &self,
        runtime: &HipRuntime,
        prefix: &DeviceBuffer<f32>,
    ) -> Result<CodePredictorProfile> {
        if prefix.len() != 2 * self.talker_hidden {
            return Err(Error::InvalidInput(format!(
                "CodePredictor prefix length must be {}, got {}",
                2 * self.talker_hidden,
                prefix.len()
            )));
        }
        let num_acoustic = self.num_acoustic_groups();
        let mut profile = CodePredictorProfile::default();

        let start = Instant::now();
        let stack_prefix = self.project_to_stack(prefix, &self.projected_prefix, 2)?;
        runtime.synchronize()?;
        profile.prefix_projection_seconds += start.elapsed().as_secs_f64();

        let start = Instant::now();
        self.stack.prefill(stack_prefix, 2)?;
        self.stack.copy_prefill_step_to(&self.prefill_hidden, 1)?;
        runtime.synchronize()?;
        profile.stack_prefill_seconds += start.elapsed().as_secs_f64();

        let start = Instant::now();
        self.logits_for_hidden(&self.prefill_hidden, 0)?;
        runtime.synchronize()?;
        profile.first_logits_seconds += start.elapsed().as_secs_f64();

        let start = Instant::now();
        launch_argmax(
            &self.kernels.argmax,
            &self.logits,
            &self.token,
            self.config.vocab_size,
        )?;
        launch_store_token(&self.kernels.store_token, &self.token, &self.tokens, 0)?;
        launch_embedding_lookup(
            &self.kernels.embedding_lookup,
            &self.codec_embeddings[0],
            &self.token,
            &self.current_embedding,
            self.talker_hidden,
        )?;
        self.embedding_sum
            .copy_from_device(&self.current_embedding)?;
        runtime.synchronize()?;
        profile.first_token_seconds += start.elapsed().as_secs_f64();

        for group in 1..num_acoustic {
            let start = Instant::now();
            let stack_embedding =
                self.project_to_stack(&self.current_embedding, &self.projected_embedding, 1)?;
            runtime.synchronize()?;
            profile.remaining_projection_seconds += start.elapsed().as_secs_f64();

            let start = Instant::now();
            self.stack
                .decode_step(stack_embedding, &self.step_hidden, group + 1)?;
            runtime.synchronize()?;
            profile.remaining_stack_seconds += start.elapsed().as_secs_f64();

            let start = Instant::now();
            self.logits_for_hidden(&self.step_hidden, group)?;
            runtime.synchronize()?;
            profile.remaining_logits_seconds += start.elapsed().as_secs_f64();

            let start = Instant::now();
            launch_argmax(
                &self.kernels.argmax,
                &self.logits,
                &self.token,
                self.config.vocab_size,
            )?;
            launch_store_token(&self.kernels.store_token, &self.token, &self.tokens, group)?;
            launch_embedding_lookup(
                &self.kernels.embedding_lookup,
                &self.codec_embeddings[group],
                &self.token,
                &self.current_embedding,
                self.talker_hidden,
            )?;
            launch_residual_add_on_stream(
                &self.kernels.residual_add,
                self.embedding_sum.as_ptr(),
                self.current_embedding.as_ptr(),
                self.embedding_sum.as_mut_ptr(),
                self.talker_hidden,
                None,
            )?;
            runtime.synchronize()?;
            profile.remaining_token_seconds += start.elapsed().as_secs_f64();
        }
        Ok(profile)
    }

    fn generate_remaining_groups_graph(&self) -> Result<()> {
        {
            let mut graph = self.remaining_groups_graph.borrow_mut();
            if graph.is_none() {
                self.graph_stream.begin_capture()?;
                self.enqueue_remaining_groups_on_stream(&self.graph_stream)?;
                let captured = self.graph_stream.end_capture()?;
                *graph = Some(captured.instantiate()?);
            }
            graph
                .as_ref()
                .expect("graph was just initialized")
                .launch(&self.graph_stream)?;
        }
        self.graph_stream.synchronize()
    }

    fn generate_remaining_groups_sampled_graph(&self, sampling: SamplingConfig) -> Result<()> {
        {
            let mut graph = self.sampled_remaining_groups_graph.borrow_mut();
            let needs_capture = graph
                .as_ref()
                .map(|existing| existing.sampling != sampling)
                .unwrap_or(true);
            if needs_capture {
                self.graph_stream.begin_capture()?;
                self.enqueue_remaining_groups_sampled_on_stream(&self.graph_stream, sampling)?;
                let captured = self.graph_stream.end_capture()?;
                *graph = Some(SampledGroupsGraph {
                    sampling,
                    graph: captured.instantiate()?,
                });
            }
            graph
                .as_ref()
                .expect("graph was just initialized")
                .graph
                .launch(&self.graph_stream)?;
        }
        self.graph_stream.synchronize()
    }

    fn enqueue_remaining_groups_on_stream(&self, stream: &HipStream) -> Result<()> {
        for group in 1..self.num_acoustic_groups() {
            let stack_embedding = self.project_to_stack_on_stream(
                &self.current_embedding,
                &self.projected_embedding,
                1,
                stream,
            )?;
            self.stack.decode_step_on_stream(
                stack_embedding,
                &self.step_hidden,
                group + 1,
                stream,
            )?;
            self.logits_for_hidden_on_stream(&self.step_hidden, group, stream)?;
            launch_argmax_on_stream(
                &self.kernels.argmax,
                &self.logits,
                &self.token,
                self.config.vocab_size,
                Some(stream),
            )?;
            launch_store_token_on_stream(
                &self.kernels.store_token,
                &self.token,
                &self.tokens,
                group,
                Some(stream),
            )?;
            launch_embedding_lookup_on_stream(
                &self.kernels.embedding_lookup,
                &self.codec_embeddings[group],
                &self.token,
                &self.current_embedding,
                self.talker_hidden,
                Some(stream),
            )?;
            launch_residual_add_on_stream(
                &self.kernels.residual_add,
                self.embedding_sum.as_ptr(),
                self.current_embedding.as_ptr(),
                self.embedding_sum.as_mut_ptr(),
                self.talker_hidden,
                Some(stream),
            )?;
        }
        Ok(())
    }

    fn enqueue_remaining_groups_sampled_on_stream(
        &self,
        stream: &HipStream,
        sampling: SamplingConfig,
    ) -> Result<()> {
        for group in 1..self.num_acoustic_groups() {
            let stack_embedding = self.project_to_stack_on_stream(
                &self.current_embedding,
                &self.projected_embedding,
                1,
                stream,
            )?;
            self.stack.decode_step_on_stream(
                stack_embedding,
                &self.step_hidden,
                group + 1,
                stream,
            )?;
            self.logits_for_hidden_on_stream(&self.step_hidden, group, stream)?;
            launch_sample_topk_random_buffer_on_stream(
                &self.kernels.sample_topk_random_buffer,
                &self.logits,
                &self.token,
                &self.sample_random_values,
                group,
                self.config.vocab_size,
                sampling.top_k,
                sampling.temperature,
                Some(stream),
            )?;
            launch_store_token_on_stream(
                &self.kernels.store_token,
                &self.token,
                &self.tokens,
                group,
                Some(stream),
            )?;
            launch_embedding_lookup_on_stream(
                &self.kernels.embedding_lookup,
                &self.codec_embeddings[group],
                &self.token,
                &self.current_embedding,
                self.talker_hidden,
                Some(stream),
            )?;
            launch_residual_add_on_stream(
                &self.kernels.residual_add,
                self.embedding_sum.as_ptr(),
                self.current_embedding.as_ptr(),
                self.embedding_sum.as_mut_ptr(),
                self.talker_hidden,
                Some(stream),
            )?;
        }
        Ok(())
    }

    fn project_to_stack<'a>(
        &'a self,
        input: &'a DeviceBuffer<f32>,
        output: &'a DeviceBuffer<f32>,
        rows: usize,
    ) -> Result<&'a DeviceBuffer<f32>> {
        if self.talker_hidden == self.hidden {
            return Ok(input);
        }
        let weight = self.projection_weight.as_ref().ok_or_else(|| {
            Error::InvalidInput("missing CodePredictor projection weight".to_string())
        })?;
        let bias = self.projection_bias.as_ref().ok_or_else(|| {
            Error::InvalidInput("missing CodePredictor projection bias".to_string())
        })?;
        self.blas
            .sgemm_row_major(input, weight, output, rows, self.hidden, self.talker_hidden)?;
        launch_bias_add(
            &self.kernels.bias_add,
            output,
            bias,
            output,
            rows,
            self.hidden,
        )?;
        Ok(output)
    }

    fn project_to_stack_on_stream<'a>(
        &'a self,
        input: &'a DeviceBuffer<f32>,
        output: &'a DeviceBuffer<f32>,
        rows: usize,
        stream: &HipStream,
    ) -> Result<&'a DeviceBuffer<f32>> {
        if self.talker_hidden == self.hidden {
            return Ok(input);
        }
        let weight = self.projection_weight.as_ref().ok_or_else(|| {
            Error::InvalidInput("missing CodePredictor projection weight".to_string())
        })?;
        let bias = self.projection_bias.as_ref().ok_or_else(|| {
            Error::InvalidInput("missing CodePredictor projection bias".to_string())
        })?;
        self.blas.sgemm_row_major_on_stream(
            input,
            weight,
            output,
            rows,
            self.hidden,
            self.talker_hidden,
            stream,
        )?;
        launch_bias_add_on_stream(
            &self.kernels.bias_add,
            output,
            bias,
            output,
            rows,
            self.hidden,
            Some(stream),
        )?;
        Ok(output)
    }

    fn logits_for_hidden(&self, hidden: &DeviceBuffer<f32>, group: usize) -> Result<()> {
        launch_rmsnorm(
            &self.kernels.rmsnorm,
            hidden.as_ptr(),
            self.norm_gamma.as_ptr(),
            self.normed.as_mut_ptr(),
            self.hidden,
            self.config.epsilon,
        )?;
        self.blas.sgemm_row_major(
            &self.normed,
            &self.lm_heads[group],
            &self.logits,
            1,
            self.config.vocab_size,
            self.hidden,
        )
    }

    fn logits_for_hidden_on_stream(
        &self,
        hidden: &DeviceBuffer<f32>,
        group: usize,
        stream: &HipStream,
    ) -> Result<()> {
        launch_rmsnorm_on_stream(
            &self.kernels.rmsnorm,
            hidden.as_ptr(),
            self.norm_gamma.as_ptr(),
            self.normed.as_mut_ptr(),
            self.hidden,
            self.config.epsilon,
            Some(stream),
        )?;
        self.blas.sgemm_row_major_on_stream(
            &self.normed,
            &self.lm_heads[group],
            &self.logits,
            1,
            self.config.vocab_size,
            self.hidden,
            stream,
        )
    }

    fn stack_device_from_prefix(&self, prefix: &[f32]) -> Result<DeviceBuffer<f32>> {
        self.prefill_hidden.from_same_context(prefix)
    }
}

impl CodePredictorKernels {
    fn compile(runtime: &HipRuntime) -> Result<Self> {
        let rms_module =
            runtime.compile_module("code_predictor_rmsnorm_f32.cpp", RMSNORM_F32_SOURCE)?;
        let argmax_module =
            runtime.compile_module("code_predictor_argmax_f32.cpp", ARGMAX_F32_SOURCE)?;
        let sampling_module =
            runtime.compile_module("code_predictor_sampling_f32.cpp", SUPPRESSION_F32_SOURCE)?;
        let embedding_module =
            runtime.compile_module("code_predictor_embedding_f32.cpp", EMBEDDING_F32_SOURCE)?;
        let elementwise_module = runtime.compile_module(
            "code_predictor_elementwise_f32.cpp",
            crate::kernels::ELEMENTWISE_F32_SOURCE,
        )?;
        Ok(Self {
            rmsnorm: rms_module.function("rmsnorm_f32")?,
            argmax: argmax_module.function("argmax_rows_f32")?,
            sample_topk: sampling_module.function("sample_topk_f32")?,
            sample_topk_random_buffer: sampling_module.function("sample_topk_random_buffer_f32")?,
            embedding_lookup: embedding_module.function("embedding_lookup_f32")?,
            store_token: embedding_module.function("store_token_i32")?,
            residual_add: elementwise_module.function("residual_add_f32")?,
            bias_add: elementwise_module.function("bias_add_f32")?,
            _rms_module: rms_module,
            _argmax_module: argmax_module,
            _sampling_module: sampling_module,
            _embedding_module: embedding_module,
            _elementwise_module: elementwise_module,
        })
    }
}

fn embedding_table_f32(
    tensor: TensorView<'_>,
    vocab_size: usize,
    hidden: usize,
    group: usize,
) -> Result<Vec<f32>> {
    let shape = tensor.shape();
    if shape.len() != 2 || shape[0] != vocab_size || shape[1] != hidden {
        return Err(Error::InvalidInput(format!(
            "codec_embedding.{group}.weight shape {shape:?}; expected [{vocab_size}, {hidden}]"
        )));
    }
    tensor_to_f32(
        &format!("{CODE_PREDICTOR_PREFIX}.model.codec_embedding.{group}.weight"),
        tensor.dtype(),
        tensor.data(),
        vocab_size * hidden,
    )
}

fn launch_bias_add(
    function: &HipFunction,
    input: &DeviceBuffer<f32>,
    bias: &DeviceBuffer<f32>,
    output: &DeviceBuffer<f32>,
    rows: usize,
    cols: usize,
) -> Result<()> {
    launch_bias_add_on_stream(function, input, bias, output, rows, cols, None)
}

fn launch_bias_add_on_stream(
    function: &HipFunction,
    input: &DeviceBuffer<f32>,
    bias: &DeviceBuffer<f32>,
    output: &DeviceBuffer<f32>,
    rows: usize,
    cols: usize,
    stream: Option<&HipStream>,
) -> Result<()> {
    let total = rows * cols;
    let mut input_ptr = input.as_ptr();
    let mut bias_ptr = bias.as_ptr();
    let mut output_ptr = output.as_mut_ptr();
    let mut rows_i32 = rows as i32;
    let mut cols_i32 = cols as i32;
    let mut params = [
        &mut input_ptr as *mut *const c_void as *mut c_void,
        &mut bias_ptr as *mut *const c_void as *mut c_void,
        &mut output_ptr as *mut *mut c_void as *mut c_void,
        &mut rows_i32 as *mut i32 as *mut c_void,
        &mut cols_i32 as *mut i32 as *mut c_void,
    ];
    let block = 256u32;
    let grid = (total as u32).div_ceil(block);
    function.launch_on_stream((grid, 1, 1), (block, 1, 1), 0, &mut params, stream)
}

fn launch_rmsnorm(
    function: &HipFunction,
    input: *const c_void,
    gamma: *const c_void,
    output: *mut c_void,
    cols: usize,
    epsilon: f32,
) -> Result<()> {
    launch_rmsnorm_on_stream(function, input, gamma, output, cols, epsilon, None)
}

fn launch_rmsnorm_on_stream(
    function: &HipFunction,
    input: *const c_void,
    gamma: *const c_void,
    output: *mut c_void,
    cols: usize,
    epsilon: f32,
    stream: Option<&HipStream>,
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
    function.launch_on_stream((1, 1, 1), (block, 1, 1), block * 4, &mut params, stream)
}

fn launch_argmax(
    function: &HipFunction,
    input: &DeviceBuffer<f32>,
    output: &DeviceBuffer<i32>,
    cols: usize,
) -> Result<()> {
    launch_argmax_on_stream(function, input, output, cols, None)
}

fn launch_argmax_on_stream(
    function: &HipFunction,
    input: &DeviceBuffer<f32>,
    output: &DeviceBuffer<i32>,
    cols: usize,
    stream: Option<&HipStream>,
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
    function.launch_on_stream((1, 1, 1), (block, 1, 1), shared, &mut params, stream)
}

fn launch_sample_topk(
    function: &HipFunction,
    logits: &DeviceBuffer<f32>,
    token: &DeviceBuffer<i32>,
    vocab_size: usize,
    top_k: usize,
    temperature: f32,
    random_value: f32,
) -> Result<()> {
    let mut logits_ptr = logits.as_ptr();
    let mut token_ptr = token.as_mut_ptr();
    let mut vocab_size_i32 = vocab_size as i32;
    let sort_size = vocab_size.next_power_of_two();
    let mut sort_size_i32 = sort_size as i32;
    let mut top_k_i32 = top_k as i32;
    let mut temperature = temperature;
    let mut random_value = random_value;
    let mut params = [
        &mut logits_ptr as *mut *const c_void as *mut c_void,
        &mut token_ptr as *mut *mut c_void as *mut c_void,
        &mut vocab_size_i32 as *mut i32 as *mut c_void,
        &mut sort_size_i32 as *mut i32 as *mut c_void,
        &mut top_k_i32 as *mut i32 as *mut c_void,
        &mut temperature as *mut f32 as *mut c_void,
        &mut random_value as *mut f32 as *mut c_void,
    ];
    let shared = (sort_size * (std::mem::size_of::<f32>() + std::mem::size_of::<i32>())) as u32;
    function.launch((1, 1, 1), (1024, 1, 1), shared, &mut params)
}

#[allow(clippy::too_many_arguments)]
fn launch_sample_topk_random_buffer_on_stream(
    function: &HipFunction,
    logits: &DeviceBuffer<f32>,
    token: &DeviceBuffer<i32>,
    random_values: &DeviceBuffer<f32>,
    random_offset: usize,
    vocab_size: usize,
    top_k: usize,
    temperature: f32,
    stream: Option<&HipStream>,
) -> Result<()> {
    let mut logits_ptr = logits.as_ptr();
    let mut token_ptr = token.as_mut_ptr();
    let mut random_values_ptr = random_values.as_ptr();
    let mut random_offset_i32 = random_offset as i32;
    let mut vocab_size_i32 = vocab_size as i32;
    let sort_size = vocab_size.next_power_of_two();
    let mut sort_size_i32 = sort_size as i32;
    let mut top_k_i32 = top_k as i32;
    let mut temperature = temperature;
    let mut params = [
        &mut logits_ptr as *mut *const c_void as *mut c_void,
        &mut token_ptr as *mut *mut c_void as *mut c_void,
        &mut random_values_ptr as *mut *const c_void as *mut c_void,
        &mut random_offset_i32 as *mut i32 as *mut c_void,
        &mut vocab_size_i32 as *mut i32 as *mut c_void,
        &mut sort_size_i32 as *mut i32 as *mut c_void,
        &mut top_k_i32 as *mut i32 as *mut c_void,
        &mut temperature as *mut f32 as *mut c_void,
    ];
    let shared = (sort_size * (std::mem::size_of::<f32>() + std::mem::size_of::<i32>())) as u32;
    function.launch_on_stream((1, 1, 1), (1024, 1, 1), shared, &mut params, stream)
}

fn launch_embedding_lookup(
    function: &HipFunction,
    table: &DeviceBuffer<f32>,
    token: &DeviceBuffer<i32>,
    output: &DeviceBuffer<f32>,
    cols: usize,
) -> Result<()> {
    launch_embedding_lookup_on_stream(function, table, token, output, cols, None)
}

fn launch_embedding_lookup_on_stream(
    function: &HipFunction,
    table: &DeviceBuffer<f32>,
    token: &DeviceBuffer<i32>,
    output: &DeviceBuffer<f32>,
    cols: usize,
    stream: Option<&HipStream>,
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
    function.launch_on_stream((grid, 1, 1), (block, 1, 1), 0, &mut params, stream)
}

fn launch_store_token(
    function: &HipFunction,
    token: &DeviceBuffer<i32>,
    output: &DeviceBuffer<i32>,
    offset: usize,
) -> Result<()> {
    launch_store_token_on_stream(function, token, output, offset, None)
}

fn launch_store_token_on_stream(
    function: &HipFunction,
    token: &DeviceBuffer<i32>,
    output: &DeviceBuffer<i32>,
    offset: usize,
    stream: Option<&HipStream>,
) -> Result<()> {
    let mut token_ptr = token.as_ptr();
    let mut output_ptr = output.as_mut_ptr();
    let mut offset_i32 = offset as i32;
    let mut params = [
        &mut token_ptr as *mut *const c_void as *mut c_void,
        &mut output_ptr as *mut *mut c_void as *mut c_void,
        &mut offset_i32 as *mut i32 as *mut c_void,
    ];
    function.launch_on_stream((1, 1, 1), (1, 1, 1), 0, &mut params, stream)
}

fn launch_residual_add_on_stream(
    function: &HipFunction,
    residual: *const c_void,
    update: *const c_void,
    output: *mut c_void,
    total: usize,
    stream: Option<&HipStream>,
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
    function.launch_on_stream((grid, 1, 1), (block, 1, 1), 0, &mut params, stream)
}
