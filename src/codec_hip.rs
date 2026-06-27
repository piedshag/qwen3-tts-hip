use std::ffi::c_void;
use std::path::Path;

use safetensors::tensor::TensorView;

use crate::blas::RocblasHandle;
use crate::buffer::DeviceBuffer;
use crate::error::{Error, Result};
use crate::kernel::{HipFunction, HipModule};
use crate::kernels::{
    ATTENTION_F32_SOURCE, CODEC_INITIAL_F32_SOURCE, ELEMENTWISE_F32_SOURCE, LAYOUT_F32_SOURCE,
    RMSNORM_F32_SOURCE, ROPE_BHSD_F32_SOURCE, SOFTMAX_F32_SOURCE,
};
use crate::runtime::HipRuntime;
use crate::weights::{TensorArchive, tensor_to_f32};

const CODE_GROUPS: usize = 16;
const CODEBOOK_SIZE: usize = 2048;
const CODEBOOK_DIM: usize = 256;
const PROJECTED_DIM: usize = 512;
const PRE_CONV_OUT: usize = 1024;
const PRE_CONV_KERNEL: usize = 3;
const PRE_CONV_PADDING: usize = 2;
const TRANSFORMER_HIDDEN: usize = 512;
const TRANSFORMER_LAYERS: usize = 8;
const TRANSFORMER_HEADS: usize = 16;
const TRANSFORMER_HEAD_DIM: usize = 64;
const TRANSFORMER_Q_OUT: usize = TRANSFORMER_HEADS * TRANSFORMER_HEAD_DIM;
const TRANSFORMER_INTERMEDIATE: usize = 1024;
const TRANSFORMER_EPSILON: f32 = 1e-5;
const TRANSFORMER_THETA: f32 = 10_000.0;
const UPSAMPLE_CHANNELS: usize = 1024;
const UPSAMPLE_RATIO: usize = 2;
const CONVNEXT_KERNEL: usize = 7;
const CONVNEXT_PADDING: usize = 6;
const CONVNEXT_INTERMEDIATE: usize = 4096;
const LAYERNORM_EPSILON: f32 = 1e-6;
const DECODER_BLOCKS: usize = 4;
const DECODER_INIT_CHANNELS: usize = 1536;
const DECODER_FINAL_CHANNELS: usize = 96;
const DECODER_INIT_KERNEL: usize = 7;
const DECODER_FINAL_KERNEL: usize = 7;
const RESIDUAL_KERNEL: usize = 7;
const RESIDUAL_POINTWISE_KERNEL: usize = 1;

pub struct HipCodecInitial {
    kernels: CodecInitialKernels,
    blas: RocblasHandle,
    first_codebook: DeviceBuffer<f32>,
    rest_codebooks: DeviceBuffer<f32>,
    first_weight: DeviceBuffer<f32>,
    rest_weight: DeviceBuffer<f32>,
    pre_conv_weight: DeviceBuffer<f32>,
    pre_conv_bias: DeviceBuffer<f32>,
    input_proj: DeviceBuffer<f32>,
    input_bias: DeviceBuffer<f32>,
    output_proj: DeviceBuffer<f32>,
    output_bias: DeviceBuffer<f32>,
    final_norm: DeviceBuffer<f32>,
    layers: Vec<CodecTransformerLayerWeights>,
    upsample_stages: Vec<UpsampleStageWeights>,
    decoder_init: CausalConvWeights,
    decoder_blocks: Vec<DecoderBlockWeights>,
    final_snake: SnakeWeights,
    final_conv: CausalConvWeights,
}

struct CodecInitialKernels {
    _module: HipModule,
    _rms_module: HipModule,
    _rope_module: HipModule,
    _layout_module: HipModule,
    _softmax_module: HipModule,
    _attention_module: HipModule,
    _elementwise_module: HipModule,
    rvq_project: HipFunction,
    causal_conv1d: HipFunction,
    transpose_ct_to_tc: HipFunction,
    transpose_tc_to_ct: HipFunction,
    scaled_residual_add: HipFunction,
    bias_add: HipFunction,
    gelu: HipFunction,
    layernorm: HipFunction,
    transconv1d: HipFunction,
    depthwise_conv1d: HipFunction,
    convnext_residual: HipFunction,
    snake_beta: HipFunction,
    causal_conv1d_dilated: HipFunction,
    transconv1d_channels: HipFunction,
    clamp: HipFunction,
    rmsnorm: HipFunction,
    rope: HipFunction,
    permute: HipFunction,
    softmax: HipFunction,
    scores: HipFunction,
    apply_value: HipFunction,
    residual_add: HipFunction,
    swiglu: HipFunction,
}

struct CodecTransformerLayerWeights {
    input_gamma: DeviceBuffer<f32>,
    q_weight: DeviceBuffer<f32>,
    k_weight: DeviceBuffer<f32>,
    v_weight: DeviceBuffer<f32>,
    o_weight: DeviceBuffer<f32>,
    attn_scale: DeviceBuffer<f32>,
    post_gamma: DeviceBuffer<f32>,
    gate_weight: DeviceBuffer<f32>,
    up_weight: DeviceBuffer<f32>,
    down_weight: DeviceBuffer<f32>,
    mlp_scale: DeviceBuffer<f32>,
}

struct UpsampleStageWeights {
    trans_weight: DeviceBuffer<f32>,
    trans_bias: DeviceBuffer<f32>,
    dw_weight: DeviceBuffer<f32>,
    dw_bias: DeviceBuffer<f32>,
    norm_gamma: DeviceBuffer<f32>,
    norm_beta: DeviceBuffer<f32>,
    pw1_weight: DeviceBuffer<f32>,
    pw1_bias: DeviceBuffer<f32>,
    pw2_weight: DeviceBuffer<f32>,
    pw2_bias: DeviceBuffer<f32>,
    gamma: DeviceBuffer<f32>,
}

struct CausalConvWeights {
    weight: DeviceBuffer<f32>,
    bias: DeviceBuffer<f32>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    dilation: usize,
    causal_padding: usize,
}

struct SnakeWeights {
    alpha: DeviceBuffer<f32>,
    beta: DeviceBuffer<f32>,
    channels: usize,
}

struct ResidualUnitWeights {
    act1: SnakeWeights,
    conv1: CausalConvWeights,
    act2: SnakeWeights,
    conv2: CausalConvWeights,
}

struct DecoderBlockWeights {
    snake: SnakeWeights,
    trans_weight: DeviceBuffer<f32>,
    trans_bias: DeviceBuffer<f32>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    res1: ResidualUnitWeights,
    res2: ResidualUnitWeights,
    res3: ResidualUnitWeights,
}

pub struct HipCodecInitialOutput {
    pub quantized: DeviceBuffer<f32>,
    pub pre_conv: DeviceBuffer<f32>,
    pub frames: usize,
}

pub struct HipCodecUpsampleOutputs {
    pub upsample_0_0: DeviceBuffer<f32>,
    pub upsample_0_1: DeviceBuffer<f32>,
    pub upsample_1_0: DeviceBuffer<f32>,
    pub upsample_1_1: DeviceBuffer<f32>,
    pub frames_0: usize,
    pub frames_1: usize,
}

pub struct HipCodecDecoderOutputs {
    pub decoder_0: DeviceBuffer<f32>,
    pub decoder_1: DeviceBuffer<f32>,
    pub decoder_2: DeviceBuffer<f32>,
    pub decoder_3: DeviceBuffer<f32>,
    pub decoder_4: DeviceBuffer<f32>,
    pub decoder_5: DeviceBuffer<f32>,
    pub decoder_6: DeviceBuffer<f32>,
    pub waveform: DeviceBuffer<f32>,
    pub frames_0: usize,
    pub frames_1: usize,
    pub frames_2: usize,
    pub frames_3: usize,
    pub frames_4: usize,
}

impl HipCodecInitial {
    pub fn load(runtime: &HipRuntime, model_dir: &Path) -> Result<Self> {
        let archive = TensorArchive::open(&model_dir.join("speech_tokenizer/model.safetensors"))?;
        let first_codebook = load_normalized_codebook(
            &archive,
            "decoder.quantizer.rvq_first.vq.layers.0._codebook",
        )?;
        let mut rest_codebooks =
            Vec::with_capacity((CODE_GROUPS - 1) * CODEBOOK_SIZE * CODEBOOK_DIM);
        for index in 0..CODE_GROUPS - 1 {
            rest_codebooks.extend(load_normalized_codebook(
                &archive,
                &format!("decoder.quantizer.rvq_rest.vq.layers.{index}._codebook"),
            )?);
        }
        let first_weight =
            conv1x1_weight(&archive, "decoder.quantizer.rvq_first.output_proj.weight")?;
        let rest_weight =
            conv1x1_weight(&archive, "decoder.quantizer.rvq_rest.output_proj.weight")?;
        let pre_conv_weight = conv1d_weight(
            &archive,
            "decoder.pre_conv.conv.weight",
            PRE_CONV_OUT,
            PROJECTED_DIM,
            PRE_CONV_KERNEL,
        )?;
        let pre_conv_bias = vector_f32(&archive, "decoder.pre_conv.conv.bias", PRE_CONV_OUT)?;
        let (input_proj, input_in, input_out) =
            linear_weight_transposed(&archive, "decoder.pre_transformer.input_proj.weight")?;
        if input_in != PRE_CONV_OUT || input_out != TRANSFORMER_HIDDEN {
            return Err(Error::InvalidInput(format!(
                "invalid codec input_proj shape [{input_out}, {input_in}]"
            )));
        }
        let input_bias = vector_f32(
            &archive,
            "decoder.pre_transformer.input_proj.bias",
            TRANSFORMER_HIDDEN,
        )?;
        let (output_proj, output_in, output_out) =
            linear_weight_transposed(&archive, "decoder.pre_transformer.output_proj.weight")?;
        if output_in != TRANSFORMER_HIDDEN || output_out != PRE_CONV_OUT {
            return Err(Error::InvalidInput(format!(
                "invalid codec output_proj shape [{output_out}, {output_in}]"
            )));
        }
        let output_bias = vector_f32(
            &archive,
            "decoder.pre_transformer.output_proj.bias",
            PRE_CONV_OUT,
        )?;
        let final_norm = vector_f32(
            &archive,
            "decoder.pre_transformer.norm.weight",
            TRANSFORMER_HIDDEN,
        )?;
        let mut layers = Vec::with_capacity(TRANSFORMER_LAYERS);
        for index in 0..TRANSFORMER_LAYERS {
            layers.push(load_transformer_layer(runtime, &archive, index)?);
        }
        let mut upsample_stages = Vec::with_capacity(2);
        for index in 0..2 {
            upsample_stages.push(load_upsample_stage(runtime, &archive, index)?);
        }
        let decoder_init = load_causal_conv(
            runtime,
            &archive,
            "decoder.decoder.0.conv",
            PRE_CONV_OUT,
            DECODER_INIT_CHANNELS,
            DECODER_INIT_KERNEL,
            1,
        )?;
        let mut decoder_blocks = Vec::with_capacity(DECODER_BLOCKS);
        let decoder_channels = [(1536, 768, 8), (768, 384, 5), (384, 192, 4), (192, 96, 3)];
        for (index, (in_channels, out_channels, stride)) in decoder_channels.into_iter().enumerate()
        {
            decoder_blocks.push(load_decoder_block(
                runtime,
                &archive,
                index,
                in_channels,
                out_channels,
                stride,
            )?);
        }
        let final_snake = load_snake(
            runtime,
            &archive,
            "decoder.decoder.5",
            DECODER_FINAL_CHANNELS,
        )?;
        let final_conv = load_causal_conv(
            runtime,
            &archive,
            "decoder.decoder.6.conv",
            DECODER_FINAL_CHANNELS,
            1,
            DECODER_FINAL_KERNEL,
            1,
        )?;

        Ok(Self {
            kernels: CodecInitialKernels::compile(runtime)?,
            blas: runtime.create_blas_handle()?,
            first_codebook: runtime.buffer_from_slice(&first_codebook)?,
            rest_codebooks: runtime.buffer_from_slice(&rest_codebooks)?,
            first_weight: runtime.buffer_from_slice(&first_weight)?,
            rest_weight: runtime.buffer_from_slice(&rest_weight)?,
            pre_conv_weight: runtime.buffer_from_slice(&pre_conv_weight)?,
            pre_conv_bias: runtime.buffer_from_slice(&pre_conv_bias)?,
            input_proj: runtime.buffer_from_slice(&input_proj)?,
            input_bias: runtime.buffer_from_slice(&input_bias)?,
            output_proj: runtime.buffer_from_slice(&output_proj)?,
            output_bias: runtime.buffer_from_slice(&output_bias)?,
            final_norm: runtime.buffer_from_slice(&final_norm)?,
            layers,
            upsample_stages,
            decoder_init,
            decoder_blocks,
            final_snake,
            final_conv,
        })
    }

    pub fn run(
        &self,
        runtime: &HipRuntime,
        codes: &[i32],
        frames: usize,
    ) -> Result<HipCodecInitialOutput> {
        if frames == 0 || codes.len() != frames * CODE_GROUPS {
            return Err(Error::InvalidInput(format!(
                "codes length {} does not match frames*groups {}",
                codes.len(),
                frames * CODE_GROUPS
            )));
        }
        let codes = runtime.buffer_from_slice(codes)?;
        let quantized = runtime.empty_buffer::<f32>(PROJECTED_DIM * frames)?;
        let pre_conv = runtime.empty_buffer::<f32>(PRE_CONV_OUT * frames)?;
        launch_rvq_project(
            &self.kernels.rvq_project,
            &codes,
            &self.first_codebook,
            &self.rest_codebooks,
            &self.first_weight,
            &self.rest_weight,
            &quantized,
            frames,
        )?;
        launch_causal_conv1d(
            &self.kernels.causal_conv1d,
            &quantized,
            &self.pre_conv_weight,
            &self.pre_conv_bias,
            &pre_conv,
            frames,
        )?;
        Ok(HipCodecInitialOutput {
            quantized,
            pre_conv,
            frames,
        })
    }

    pub fn run_pre_transformer(
        &self,
        runtime: &HipRuntime,
        pre_conv: &DeviceBuffer<f32>,
        frames: usize,
    ) -> Result<DeviceBuffer<f32>> {
        if pre_conv.len() != PRE_CONV_OUT * frames {
            return Err(Error::InvalidInput(format!(
                "pre_conv length {} does not match {}",
                pre_conv.len(),
                PRE_CONV_OUT * frames
            )));
        }
        let input_tc = runtime.empty_buffer::<f32>(frames * PRE_CONV_OUT)?;
        let hidden_a = runtime.empty_buffer::<f32>(frames * TRANSFORMER_HIDDEN)?;
        let hidden_b = runtime.empty_buffer::<f32>(frames * TRANSFORMER_HIDDEN)?;
        let normed = runtime.empty_buffer::<f32>(frames * TRANSFORMER_HIDDEN)?;
        let q_proj = runtime.empty_buffer::<f32>(frames * TRANSFORMER_Q_OUT)?;
        let k_proj = runtime.empty_buffer::<f32>(frames * TRANSFORMER_Q_OUT)?;
        let v_proj = runtime.empty_buffer::<f32>(frames * TRANSFORMER_Q_OUT)?;
        let q_bhsd = runtime.empty_buffer::<f32>(frames * TRANSFORMER_Q_OUT)?;
        let k_bhsd = runtime.empty_buffer::<f32>(frames * TRANSFORMER_Q_OUT)?;
        let v_bhsd = runtime.empty_buffer::<f32>(frames * TRANSFORMER_Q_OUT)?;
        let q_rope = runtime.empty_buffer::<f32>(frames * TRANSFORMER_Q_OUT)?;
        let k_rope = runtime.empty_buffer::<f32>(frames * TRANSFORMER_Q_OUT)?;
        let scores = runtime.empty_buffer::<f32>(TRANSFORMER_HEADS * frames * frames)?;
        let probs = runtime.empty_buffer::<f32>(TRANSFORMER_HEADS * frames * frames)?;
        let attended = runtime.empty_buffer::<f32>(frames * TRANSFORMER_Q_OUT)?;
        let projected = runtime.empty_buffer::<f32>(frames * TRANSFORMER_HIDDEN)?;
        let attn_output = runtime.empty_buffer::<f32>(frames * TRANSFORMER_HIDDEN)?;
        let post_norm = runtime.empty_buffer::<f32>(frames * TRANSFORMER_HIDDEN)?;
        let gate = runtime.empty_buffer::<f32>(frames * TRANSFORMER_INTERMEDIATE)?;
        let up = runtime.empty_buffer::<f32>(frames * TRANSFORMER_INTERMEDIATE)?;
        let swiglu = runtime.empty_buffer::<f32>(frames * TRANSFORMER_INTERMEDIATE)?;
        let mlp_down = runtime.empty_buffer::<f32>(frames * TRANSFORMER_HIDDEN)?;
        let output_tc = runtime.empty_buffer::<f32>(frames * PRE_CONV_OUT)?;
        let output_ct = runtime.empty_buffer::<f32>(PRE_CONV_OUT * frames)?;

        launch_transpose_ct_to_tc(
            &self.kernels.transpose_ct_to_tc,
            pre_conv,
            &input_tc,
            PRE_CONV_OUT,
            frames,
        )?;
        self.blas.sgemm_row_major(
            &input_tc,
            &self.input_proj,
            &hidden_a,
            frames,
            TRANSFORMER_HIDDEN,
            PRE_CONV_OUT,
        )?;
        launch_bias_add(
            &self.kernels.bias_add,
            &hidden_a,
            &self.input_bias,
            &hidden_a,
            frames,
            TRANSFORMER_HIDDEN,
        )?;

        let mut current_a = true;
        for layer in &self.layers {
            let (input, output) = if current_a {
                (&hidden_a, &hidden_b)
            } else {
                (&hidden_b, &hidden_a)
            };
            launch_rmsnorm(
                &self.kernels.rmsnorm,
                input.as_ptr(),
                layer.input_gamma.as_ptr(),
                normed.as_mut_ptr(),
                frames,
                TRANSFORMER_HIDDEN,
                TRANSFORMER_EPSILON,
            )?;
            self.blas.sgemm_row_major(
                &normed,
                &layer.q_weight,
                &q_proj,
                frames,
                TRANSFORMER_Q_OUT,
                TRANSFORMER_HIDDEN,
            )?;
            self.blas.sgemm_row_major(
                &normed,
                &layer.k_weight,
                &k_proj,
                frames,
                TRANSFORMER_Q_OUT,
                TRANSFORMER_HIDDEN,
            )?;
            self.blas.sgemm_row_major(
                &normed,
                &layer.v_weight,
                &v_proj,
                frames,
                TRANSFORMER_Q_OUT,
                TRANSFORMER_HIDDEN,
            )?;
            launch_permute(
                &self.kernels.permute,
                q_proj.as_ptr(),
                q_bhsd.as_mut_ptr(),
                frames,
                TRANSFORMER_HEADS,
                TRANSFORMER_HEAD_DIM,
            )?;
            launch_permute(
                &self.kernels.permute,
                k_proj.as_ptr(),
                k_bhsd.as_mut_ptr(),
                frames,
                TRANSFORMER_HEADS,
                TRANSFORMER_HEAD_DIM,
            )?;
            launch_permute(
                &self.kernels.permute,
                v_proj.as_ptr(),
                v_bhsd.as_mut_ptr(),
                frames,
                TRANSFORMER_HEADS,
                TRANSFORMER_HEAD_DIM,
            )?;
            launch_rope(
                &self.kernels.rope,
                q_bhsd.as_ptr(),
                q_rope.as_mut_ptr(),
                TRANSFORMER_HEADS * frames * TRANSFORMER_HEAD_DIM,
                TRANSFORMER_HEADS,
                frames,
                TRANSFORMER_HEAD_DIM,
                TRANSFORMER_THETA,
            )?;
            launch_rope(
                &self.kernels.rope,
                k_bhsd.as_ptr(),
                k_rope.as_mut_ptr(),
                TRANSFORMER_HEADS * frames * TRANSFORMER_HEAD_DIM,
                TRANSFORMER_HEADS,
                frames,
                TRANSFORMER_HEAD_DIM,
                TRANSFORMER_THETA,
            )?;
            launch_attention_scores(
                &self.kernels.scores,
                q_rope.as_ptr(),
                k_rope.as_ptr(),
                scores.as_mut_ptr(),
                frames,
                TRANSFORMER_HEAD_DIM,
            )?;
            launch_softmax(
                &self.kernels.softmax,
                scores.as_ptr(),
                probs.as_mut_ptr(),
                TRANSFORMER_HEADS * frames,
                frames,
            )?;
            launch_apply_value(
                &self.kernels.apply_value,
                probs.as_ptr(),
                v_bhsd.as_ptr(),
                attended.as_mut_ptr(),
                frames,
                TRANSFORMER_HEAD_DIM,
            )?;
            self.blas.sgemm_row_major(
                &attended,
                &layer.o_weight,
                &projected,
                frames,
                TRANSFORMER_HIDDEN,
                TRANSFORMER_Q_OUT,
            )?;
            launch_scaled_add(
                &self.kernels.scaled_residual_add,
                input,
                &projected,
                &layer.attn_scale,
                &attn_output,
                frames,
                TRANSFORMER_HIDDEN,
            )?;
            launch_rmsnorm(
                &self.kernels.rmsnorm,
                attn_output.as_ptr(),
                layer.post_gamma.as_ptr(),
                post_norm.as_mut_ptr(),
                frames,
                TRANSFORMER_HIDDEN,
                TRANSFORMER_EPSILON,
            )?;
            self.blas.sgemm_row_major(
                &post_norm,
                &layer.gate_weight,
                &gate,
                frames,
                TRANSFORMER_INTERMEDIATE,
                TRANSFORMER_HIDDEN,
            )?;
            self.blas.sgemm_row_major(
                &post_norm,
                &layer.up_weight,
                &up,
                frames,
                TRANSFORMER_INTERMEDIATE,
                TRANSFORMER_HIDDEN,
            )?;
            launch_ternary(
                &self.kernels.swiglu,
                gate.as_ptr(),
                up.as_ptr(),
                swiglu.as_mut_ptr(),
                frames * TRANSFORMER_INTERMEDIATE,
            )?;
            self.blas.sgemm_row_major(
                &swiglu,
                &layer.down_weight,
                &mlp_down,
                frames,
                TRANSFORMER_HIDDEN,
                TRANSFORMER_INTERMEDIATE,
            )?;
            launch_scaled_add(
                &self.kernels.scaled_residual_add,
                &attn_output,
                &mlp_down,
                &layer.mlp_scale,
                output,
                frames,
                TRANSFORMER_HIDDEN,
            )?;
            current_a = !current_a;
        }

        let final_hidden = if current_a { &hidden_a } else { &hidden_b };
        launch_rmsnorm(
            &self.kernels.rmsnorm,
            final_hidden.as_ptr(),
            self.final_norm.as_ptr(),
            normed.as_mut_ptr(),
            frames,
            TRANSFORMER_HIDDEN,
            TRANSFORMER_EPSILON,
        )?;
        self.blas.sgemm_row_major(
            &normed,
            &self.output_proj,
            &output_tc,
            frames,
            PRE_CONV_OUT,
            TRANSFORMER_HIDDEN,
        )?;
        launch_bias_add(
            &self.kernels.bias_add,
            &output_tc,
            &self.output_bias,
            &output_tc,
            frames,
            PRE_CONV_OUT,
        )?;
        launch_transpose_tc_to_ct(
            &self.kernels.transpose_tc_to_ct,
            &output_tc,
            &output_ct,
            frames,
            PRE_CONV_OUT,
        )?;
        Ok(output_ct)
    }

    pub fn run_upsample_stages(
        &self,
        runtime: &HipRuntime,
        input: &DeviceBuffer<f32>,
        frames: usize,
    ) -> Result<HipCodecUpsampleOutputs> {
        if self.upsample_stages.len() != 2 {
            return Err(Error::InvalidInput(
                "expected 2 upsample stages".to_string(),
            ));
        }
        let frames_0 = frames * UPSAMPLE_RATIO;
        let upsample_0_0 = runtime.empty_buffer::<f32>(UPSAMPLE_CHANNELS * frames_0)?;
        let upsample_0_1 = runtime.empty_buffer::<f32>(UPSAMPLE_CHANNELS * frames_0)?;
        self.run_upsample_stage(
            runtime,
            input,
            frames,
            &self.upsample_stages[0],
            &upsample_0_0,
            &upsample_0_1,
        )?;

        let frames_1 = frames_0 * UPSAMPLE_RATIO;
        let upsample_1_0 = runtime.empty_buffer::<f32>(UPSAMPLE_CHANNELS * frames_1)?;
        let upsample_1_1 = runtime.empty_buffer::<f32>(UPSAMPLE_CHANNELS * frames_1)?;
        self.run_upsample_stage(
            runtime,
            &upsample_0_1,
            frames_0,
            &self.upsample_stages[1],
            &upsample_1_0,
            &upsample_1_1,
        )?;

        Ok(HipCodecUpsampleOutputs {
            upsample_0_0,
            upsample_0_1,
            upsample_1_0,
            upsample_1_1,
            frames_0,
            frames_1,
        })
    }

    fn run_upsample_stage(
        &self,
        runtime: &HipRuntime,
        input: &DeviceBuffer<f32>,
        in_frames: usize,
        weights: &UpsampleStageWeights,
        trans_output: &DeviceBuffer<f32>,
        block_output: &DeviceBuffer<f32>,
    ) -> Result<()> {
        let out_frames = in_frames * UPSAMPLE_RATIO;
        launch_transconv1d(
            &self.kernels.transconv1d,
            input,
            &weights.trans_weight,
            &weights.trans_bias,
            trans_output,
            in_frames,
            out_frames,
        )?;

        let dw = runtime.empty_buffer::<f32>(UPSAMPLE_CHANNELS * out_frames)?;
        let dw_tc = runtime.empty_buffer::<f32>(out_frames * UPSAMPLE_CHANNELS)?;
        let norm = runtime.empty_buffer::<f32>(out_frames * UPSAMPLE_CHANNELS)?;
        let pw1 = runtime.empty_buffer::<f32>(out_frames * CONVNEXT_INTERMEDIATE)?;
        let gelu = runtime.empty_buffer::<f32>(out_frames * CONVNEXT_INTERMEDIATE)?;
        let pw2 = runtime.empty_buffer::<f32>(out_frames * UPSAMPLE_CHANNELS)?;

        launch_depthwise_conv1d(
            &self.kernels.depthwise_conv1d,
            trans_output,
            &weights.dw_weight,
            &weights.dw_bias,
            &dw,
            out_frames,
        )?;
        launch_transpose_ct_to_tc(
            &self.kernels.transpose_ct_to_tc,
            &dw,
            &dw_tc,
            UPSAMPLE_CHANNELS,
            out_frames,
        )?;
        launch_layernorm(
            &self.kernels.layernorm,
            &dw_tc,
            &weights.norm_gamma,
            &weights.norm_beta,
            &norm,
            out_frames,
            UPSAMPLE_CHANNELS,
        )?;
        self.blas.sgemm_row_major(
            &norm,
            &weights.pw1_weight,
            &pw1,
            out_frames,
            CONVNEXT_INTERMEDIATE,
            UPSAMPLE_CHANNELS,
        )?;
        launch_bias_add(
            &self.kernels.bias_add,
            &pw1,
            &weights.pw1_bias,
            &pw1,
            out_frames,
            CONVNEXT_INTERMEDIATE,
        )?;
        launch_gelu(
            &self.kernels.gelu,
            &pw1,
            &gelu,
            out_frames * CONVNEXT_INTERMEDIATE,
        )?;
        self.blas.sgemm_row_major(
            &gelu,
            &weights.pw2_weight,
            &pw2,
            out_frames,
            UPSAMPLE_CHANNELS,
            CONVNEXT_INTERMEDIATE,
        )?;
        launch_bias_add(
            &self.kernels.bias_add,
            &pw2,
            &weights.pw2_bias,
            &pw2,
            out_frames,
            UPSAMPLE_CHANNELS,
        )?;
        launch_convnext_residual(
            &self.kernels.convnext_residual,
            trans_output,
            &pw2,
            &weights.gamma,
            block_output,
            out_frames,
        )
    }

    pub fn run_decoder_stages(
        &self,
        runtime: &HipRuntime,
        input: &DeviceBuffer<f32>,
        frames: usize,
    ) -> Result<HipCodecDecoderOutputs> {
        let decoder_0 = runtime.empty_buffer::<f32>(DECODER_INIT_CHANNELS * frames)?;
        self.run_causal_conv(input, &self.decoder_init, &decoder_0, frames)?;

        let frames_1 = frames * self.decoder_blocks[0].stride;
        let decoder_1 =
            runtime.empty_buffer::<f32>(self.decoder_blocks[0].out_channels * frames_1)?;
        self.run_decoder_block(
            runtime,
            &decoder_0,
            frames,
            &self.decoder_blocks[0],
            &decoder_1,
        )?;

        let frames_2 = frames_1 * self.decoder_blocks[1].stride;
        let decoder_2 =
            runtime.empty_buffer::<f32>(self.decoder_blocks[1].out_channels * frames_2)?;
        self.run_decoder_block(
            runtime,
            &decoder_1,
            frames_1,
            &self.decoder_blocks[1],
            &decoder_2,
        )?;

        let frames_3 = frames_2 * self.decoder_blocks[2].stride;
        let decoder_3 =
            runtime.empty_buffer::<f32>(self.decoder_blocks[2].out_channels * frames_3)?;
        self.run_decoder_block(
            runtime,
            &decoder_2,
            frames_2,
            &self.decoder_blocks[2],
            &decoder_3,
        )?;

        let frames_4 = frames_3 * self.decoder_blocks[3].stride;
        let decoder_4 =
            runtime.empty_buffer::<f32>(self.decoder_blocks[3].out_channels * frames_4)?;
        self.run_decoder_block(
            runtime,
            &decoder_3,
            frames_3,
            &self.decoder_blocks[3],
            &decoder_4,
        )?;

        let decoder_5 = runtime.empty_buffer::<f32>(DECODER_FINAL_CHANNELS * frames_4)?;
        launch_snake(
            &self.kernels.snake_beta,
            &decoder_4,
            &self.final_snake,
            &decoder_5,
            frames_4,
        )?;

        let decoder_6 = runtime.empty_buffer::<f32>(frames_4)?;
        self.run_causal_conv(&decoder_5, &self.final_conv, &decoder_6, frames_4)?;
        let waveform = runtime.empty_buffer::<f32>(frames_4)?;
        launch_clamp(
            &self.kernels.clamp,
            &decoder_6,
            &waveform,
            frames_4,
            -1.0,
            1.0,
        )?;

        Ok(HipCodecDecoderOutputs {
            decoder_0,
            decoder_1,
            decoder_2,
            decoder_3,
            decoder_4,
            decoder_5,
            decoder_6,
            waveform,
            frames_0: frames,
            frames_1,
            frames_2,
            frames_3,
            frames_4,
        })
    }

    fn run_decoder_block(
        &self,
        runtime: &HipRuntime,
        input: &DeviceBuffer<f32>,
        in_frames: usize,
        block: &DecoderBlockWeights,
        output: &DeviceBuffer<f32>,
    ) -> Result<()> {
        let snake = runtime.empty_buffer::<f32>(block.in_channels * in_frames)?;
        launch_snake(
            &self.kernels.snake_beta,
            input,
            &block.snake,
            &snake,
            in_frames,
        )?;
        let out_frames = in_frames * block.stride;
        let trans = runtime.empty_buffer::<f32>(block.out_channels * out_frames)?;
        launch_transconv1d_channels(
            &self.kernels.transconv1d_channels,
            &snake,
            &block.trans_weight,
            &block.trans_bias,
            &trans,
            in_frames,
            out_frames,
            block.in_channels,
            block.out_channels,
            block.kernel_size,
            block.stride,
        )?;
        let res1 = runtime.empty_buffer::<f32>(block.out_channels * out_frames)?;
        self.run_residual_unit(runtime, &trans, out_frames, &block.res1, &res1)?;
        let res2 = runtime.empty_buffer::<f32>(block.out_channels * out_frames)?;
        self.run_residual_unit(runtime, &res1, out_frames, &block.res2, &res2)?;
        self.run_residual_unit(runtime, &res2, out_frames, &block.res3, output)
    }

    fn run_residual_unit(
        &self,
        runtime: &HipRuntime,
        input: &DeviceBuffer<f32>,
        frames: usize,
        unit: &ResidualUnitWeights,
        output: &DeviceBuffer<f32>,
    ) -> Result<()> {
        let act1 = runtime.empty_buffer::<f32>(unit.conv1.in_channels * frames)?;
        let conv1 = runtime.empty_buffer::<f32>(unit.conv1.out_channels * frames)?;
        let act2 = runtime.empty_buffer::<f32>(unit.conv2.in_channels * frames)?;
        let conv2 = runtime.empty_buffer::<f32>(unit.conv2.out_channels * frames)?;
        launch_snake(&self.kernels.snake_beta, input, &unit.act1, &act1, frames)?;
        self.run_causal_conv(&act1, &unit.conv1, &conv1, frames)?;
        launch_snake(&self.kernels.snake_beta, &conv1, &unit.act2, &act2, frames)?;
        self.run_causal_conv(&act2, &unit.conv2, &conv2, frames)?;
        launch_ternary(
            &self.kernels.residual_add,
            input.as_ptr(),
            conv2.as_ptr(),
            output.as_mut_ptr(),
            unit.conv2.out_channels * frames,
        )
    }

    fn run_causal_conv(
        &self,
        input: &DeviceBuffer<f32>,
        conv: &CausalConvWeights,
        output: &DeviceBuffer<f32>,
        frames: usize,
    ) -> Result<()> {
        launch_causal_conv1d_dilated(
            &self.kernels.causal_conv1d_dilated,
            input,
            &conv.weight,
            &conv.bias,
            output,
            frames,
            conv.in_channels,
            conv.out_channels,
            conv.kernel_size,
            conv.dilation,
            conv.causal_padding,
        )
    }
}

impl CodecInitialKernels {
    fn compile(runtime: &HipRuntime) -> Result<Self> {
        let module = runtime.compile_module("codec_initial_f32.cpp", CODEC_INITIAL_F32_SOURCE)?;
        let rms_module = runtime.compile_module("codec_rmsnorm_f32.cpp", RMSNORM_F32_SOURCE)?;
        let rope_module = runtime.compile_module("codec_rope_f32.cpp", ROPE_BHSD_F32_SOURCE)?;
        let layout_module = runtime.compile_module("codec_layout_f32.cpp", LAYOUT_F32_SOURCE)?;
        let softmax_module = runtime.compile_module("codec_softmax_f32.cpp", SOFTMAX_F32_SOURCE)?;
        let attention_module =
            runtime.compile_module("codec_attention_f32.cpp", ATTENTION_F32_SOURCE)?;
        let elementwise_module =
            runtime.compile_module("codec_elementwise_f32.cpp", ELEMENTWISE_F32_SOURCE)?;
        Ok(Self {
            rvq_project: module.function("codec_rvq_project_f32")?,
            causal_conv1d: module.function("codec_causal_conv1d_f32")?,
            transpose_ct_to_tc: module.function("codec_transpose_ct_to_tc_f32")?,
            transpose_tc_to_ct: module.function("codec_transpose_tc_to_ct_f32")?,
            scaled_residual_add: module.function("codec_scaled_residual_add_f32")?,
            bias_add: module.function("codec_bias_add_f32")?,
            gelu: module.function("codec_gelu_f32")?,
            layernorm: module.function("codec_layernorm_f32")?,
            transconv1d: module.function("codec_transconv1d_f32")?,
            depthwise_conv1d: module.function("codec_depthwise_causal_conv1d_f32")?,
            convnext_residual: module.function("codec_convnext_residual_f32")?,
            snake_beta: module.function("codec_snake_beta_f32")?,
            causal_conv1d_dilated: module.function("codec_causal_conv1d_dilated_f32")?,
            transconv1d_channels: module.function("codec_transconv1d_channels_f32")?,
            clamp: module.function("codec_clamp_f32")?,
            rmsnorm: rms_module.function("rmsnorm_f32")?,
            rope: rope_module.function("rope_bhsd_f32")?,
            permute: layout_module.function("permute_bshd_to_bhsd_f32")?,
            softmax: softmax_module.function("masked_softmax_f32")?,
            scores: attention_module.function("attention_scores_causal_f32")?,
            apply_value: attention_module.function("attention_apply_value_f32")?,
            residual_add: elementwise_module.function("residual_add_f32")?,
            swiglu: elementwise_module.function("swiglu_f32")?,
            _module: module,
            _rms_module: rms_module,
            _rope_module: rope_module,
            _layout_module: layout_module,
            _softmax_module: softmax_module,
            _attention_module: attention_module,
            _elementwise_module: elementwise_module,
        })
    }
}

fn load_transformer_layer(
    runtime: &HipRuntime,
    archive: &TensorArchive,
    index: usize,
) -> Result<CodecTransformerLayerWeights> {
    let prefix = format!("decoder.pre_transformer.layers.{index}");
    Ok(CodecTransformerLayerWeights {
        input_gamma: runtime.buffer_from_slice(&vector_f32(
            archive,
            &format!("{prefix}.input_layernorm.weight"),
            TRANSFORMER_HIDDEN,
        )?)?,
        q_weight: runtime.buffer_from_slice(
            &linear_weight_transposed(archive, &format!("{prefix}.self_attn.q_proj.weight"))?.0,
        )?,
        k_weight: runtime.buffer_from_slice(
            &linear_weight_transposed(archive, &format!("{prefix}.self_attn.k_proj.weight"))?.0,
        )?,
        v_weight: runtime.buffer_from_slice(
            &linear_weight_transposed(archive, &format!("{prefix}.self_attn.v_proj.weight"))?.0,
        )?,
        o_weight: runtime.buffer_from_slice(
            &linear_weight_transposed(archive, &format!("{prefix}.self_attn.o_proj.weight"))?.0,
        )?,
        attn_scale: runtime.buffer_from_slice(&vector_f32(
            archive,
            &format!("{prefix}.self_attn_layer_scale.scale"),
            TRANSFORMER_HIDDEN,
        )?)?,
        post_gamma: runtime.buffer_from_slice(&vector_f32(
            archive,
            &format!("{prefix}.post_attention_layernorm.weight"),
            TRANSFORMER_HIDDEN,
        )?)?,
        gate_weight: runtime.buffer_from_slice(
            &linear_weight_transposed(archive, &format!("{prefix}.mlp.gate_proj.weight"))?.0,
        )?,
        up_weight: runtime.buffer_from_slice(
            &linear_weight_transposed(archive, &format!("{prefix}.mlp.up_proj.weight"))?.0,
        )?,
        down_weight: runtime.buffer_from_slice(
            &linear_weight_transposed(archive, &format!("{prefix}.mlp.down_proj.weight"))?.0,
        )?,
        mlp_scale: runtime.buffer_from_slice(&vector_f32(
            archive,
            &format!("{prefix}.mlp_layer_scale.scale"),
            TRANSFORMER_HIDDEN,
        )?)?,
    })
}

fn load_upsample_stage(
    runtime: &HipRuntime,
    archive: &TensorArchive,
    index: usize,
) -> Result<UpsampleStageWeights> {
    let prefix = format!("decoder.upsample.{index}");
    let trans_weight = tensor_f32(
        archive.tensor(&format!("{prefix}.0.conv.weight"))?,
        &format!("{prefix}.0.conv.weight"),
        &[UPSAMPLE_CHANNELS, UPSAMPLE_CHANNELS, UPSAMPLE_RATIO],
    )?;
    let trans_bias = vector_f32(archive, &format!("{prefix}.0.conv.bias"), UPSAMPLE_CHANNELS)?;
    let dw_weight_raw = tensor_f32(
        archive.tensor(&format!("{prefix}.1.dwconv.conv.weight"))?,
        &format!("{prefix}.1.dwconv.conv.weight"),
        &[UPSAMPLE_CHANNELS, 1, CONVNEXT_KERNEL],
    )?;
    let mut dw_weight = vec![0.0; UPSAMPLE_CHANNELS * CONVNEXT_KERNEL];
    for channel in 0..UPSAMPLE_CHANNELS {
        for k in 0..CONVNEXT_KERNEL {
            dw_weight[channel * CONVNEXT_KERNEL + k] = dw_weight_raw[channel * CONVNEXT_KERNEL + k];
        }
    }
    let dw_bias = vector_f32(
        archive,
        &format!("{prefix}.1.dwconv.conv.bias"),
        UPSAMPLE_CHANNELS,
    )?;
    let norm_gamma = vector_f32(
        archive,
        &format!("{prefix}.1.norm.weight"),
        UPSAMPLE_CHANNELS,
    )?;
    let norm_beta = vector_f32(archive, &format!("{prefix}.1.norm.bias"), UPSAMPLE_CHANNELS)?;
    let pw1_weight = linear_weight_transposed(archive, &format!("{prefix}.1.pwconv1.weight"))?.0;
    let pw1_bias = vector_f32(
        archive,
        &format!("{prefix}.1.pwconv1.bias"),
        CONVNEXT_INTERMEDIATE,
    )?;
    let pw2_weight = linear_weight_transposed(archive, &format!("{prefix}.1.pwconv2.weight"))?.0;
    let pw2_bias = vector_f32(
        archive,
        &format!("{prefix}.1.pwconv2.bias"),
        UPSAMPLE_CHANNELS,
    )?;
    let gamma = vector_f32(archive, &format!("{prefix}.1.gamma"), UPSAMPLE_CHANNELS)?;

    Ok(UpsampleStageWeights {
        trans_weight: runtime.buffer_from_slice(&trans_weight)?,
        trans_bias: runtime.buffer_from_slice(&trans_bias)?,
        dw_weight: runtime.buffer_from_slice(&dw_weight)?,
        dw_bias: runtime.buffer_from_slice(&dw_bias)?,
        norm_gamma: runtime.buffer_from_slice(&norm_gamma)?,
        norm_beta: runtime.buffer_from_slice(&norm_beta)?,
        pw1_weight: runtime.buffer_from_slice(&pw1_weight)?,
        pw1_bias: runtime.buffer_from_slice(&pw1_bias)?,
        pw2_weight: runtime.buffer_from_slice(&pw2_weight)?,
        pw2_bias: runtime.buffer_from_slice(&pw2_bias)?,
        gamma: runtime.buffer_from_slice(&gamma)?,
    })
}

fn load_decoder_block(
    runtime: &HipRuntime,
    archive: &TensorArchive,
    index: usize,
    in_channels: usize,
    out_channels: usize,
    stride: usize,
) -> Result<DecoderBlockWeights> {
    let prefix = format!("decoder.decoder.{}.block", index + 1);
    let kernel_size = stride * 2;
    let trans_weight = tensor_f32(
        archive.tensor(&format!("{prefix}.1.conv.weight"))?,
        &format!("{prefix}.1.conv.weight"),
        &[in_channels, out_channels, kernel_size],
    )?;
    let trans_bias = vector_f32(archive, &format!("{prefix}.1.conv.bias"), out_channels)?;
    Ok(DecoderBlockWeights {
        snake: load_snake(runtime, archive, &format!("{prefix}.0"), in_channels)?,
        trans_weight: runtime.buffer_from_slice(&trans_weight)?,
        trans_bias: runtime.buffer_from_slice(&trans_bias)?,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        res1: load_residual_unit(runtime, archive, &format!("{prefix}.2"), out_channels, 1)?,
        res2: load_residual_unit(runtime, archive, &format!("{prefix}.3"), out_channels, 3)?,
        res3: load_residual_unit(runtime, archive, &format!("{prefix}.4"), out_channels, 9)?,
    })
}

fn load_residual_unit(
    runtime: &HipRuntime,
    archive: &TensorArchive,
    prefix: &str,
    channels: usize,
    dilation: usize,
) -> Result<ResidualUnitWeights> {
    Ok(ResidualUnitWeights {
        act1: load_snake(runtime, archive, &format!("{prefix}.act1"), channels)?,
        conv1: load_causal_conv(
            runtime,
            archive,
            &format!("{prefix}.conv1.conv"),
            channels,
            channels,
            RESIDUAL_KERNEL,
            dilation,
        )?,
        act2: load_snake(runtime, archive, &format!("{prefix}.act2"), channels)?,
        conv2: load_causal_conv(
            runtime,
            archive,
            &format!("{prefix}.conv2.conv"),
            channels,
            channels,
            RESIDUAL_POINTWISE_KERNEL,
            1,
        )?,
    })
}

fn load_snake(
    runtime: &HipRuntime,
    archive: &TensorArchive,
    prefix: &str,
    channels: usize,
) -> Result<SnakeWeights> {
    let alpha = vector_f32(archive, &format!("{prefix}.alpha"), channels)?;
    let beta = vector_f32(archive, &format!("{prefix}.beta"), channels)?;
    Ok(SnakeWeights {
        alpha: runtime.buffer_from_slice(&alpha)?,
        beta: runtime.buffer_from_slice(&beta)?,
        channels,
    })
}

fn load_causal_conv(
    runtime: &HipRuntime,
    archive: &TensorArchive,
    prefix: &str,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    dilation: usize,
) -> Result<CausalConvWeights> {
    let weight = tensor_f32(
        archive.tensor(&format!("{prefix}.weight"))?,
        &format!("{prefix}.weight"),
        &[out_channels, in_channels, kernel_size],
    )?;
    let bias = vector_f32(archive, &format!("{prefix}.bias"), out_channels)?;
    Ok(CausalConvWeights {
        weight: runtime.buffer_from_slice(&weight)?,
        bias: runtime.buffer_from_slice(&bias)?,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
        causal_padding: dilation * kernel_size.saturating_sub(1),
    })
}

fn launch_rvq_project(
    kernel: &HipFunction,
    codes: &DeviceBuffer<i32>,
    first_codebook: &DeviceBuffer<f32>,
    rest_codebooks: &DeviceBuffer<f32>,
    first_weight: &DeviceBuffer<f32>,
    rest_weight: &DeviceBuffer<f32>,
    output: &DeviceBuffer<f32>,
    frames: usize,
) -> Result<()> {
    let total = frames * PROJECTED_DIM;
    let mut frames_i = frames as i32;
    let mut groups_i = CODE_GROUPS as i32;
    let mut codebook_size_i = CODEBOOK_SIZE as i32;
    let mut codebook_dim_i = CODEBOOK_DIM as i32;
    let mut projected_dim_i = PROJECTED_DIM as i32;
    let mut params = [
        &mut (codes.as_ptr() as *mut c_void) as *mut _ as *mut c_void,
        &mut (first_codebook.as_ptr() as *mut c_void) as *mut _ as *mut c_void,
        &mut (rest_codebooks.as_ptr() as *mut c_void) as *mut _ as *mut c_void,
        &mut (first_weight.as_ptr() as *mut c_void) as *mut _ as *mut c_void,
        &mut (rest_weight.as_ptr() as *mut c_void) as *mut _ as *mut c_void,
        &mut (output.as_mut_ptr()) as *mut _ as *mut c_void,
        &mut frames_i as *mut _ as *mut c_void,
        &mut groups_i as *mut _ as *mut c_void,
        &mut codebook_size_i as *mut _ as *mut c_void,
        &mut codebook_dim_i as *mut _ as *mut c_void,
        &mut projected_dim_i as *mut _ as *mut c_void,
    ];
    let block = 256u32;
    let grid = (total as u32).div_ceil(block);
    kernel.launch((grid, 1, 1), (block, 1, 1), 0, &mut params)
}

fn launch_causal_conv1d(
    kernel: &HipFunction,
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<f32>,
    bias: &DeviceBuffer<f32>,
    output: &DeviceBuffer<f32>,
    frames: usize,
) -> Result<()> {
    let total = frames * PRE_CONV_OUT;
    let mut frames_i = frames as i32;
    let mut in_channels_i = PROJECTED_DIM as i32;
    let mut out_channels_i = PRE_CONV_OUT as i32;
    let mut kernel_size_i = PRE_CONV_KERNEL as i32;
    let mut padding_i = PRE_CONV_PADDING as i32;
    let mut params = [
        &mut (input.as_ptr() as *mut c_void) as *mut _ as *mut c_void,
        &mut (weight.as_ptr() as *mut c_void) as *mut _ as *mut c_void,
        &mut (bias.as_ptr() as *mut c_void) as *mut _ as *mut c_void,
        &mut (output.as_mut_ptr()) as *mut _ as *mut c_void,
        &mut frames_i as *mut _ as *mut c_void,
        &mut in_channels_i as *mut _ as *mut c_void,
        &mut out_channels_i as *mut _ as *mut c_void,
        &mut kernel_size_i as *mut _ as *mut c_void,
        &mut padding_i as *mut _ as *mut c_void,
    ];
    let block = 256u32;
    let grid = (total as u32).div_ceil(block);
    kernel.launch((grid, 1, 1), (block, 1, 1), 0, &mut params)
}

fn launch_transpose_ct_to_tc(
    kernel: &HipFunction,
    input: &DeviceBuffer<f32>,
    output: &DeviceBuffer<f32>,
    channels: usize,
    frames: usize,
) -> Result<()> {
    let total = channels * frames;
    let mut channels_i = channels as i32;
    let mut frames_i = frames as i32;
    let mut total_i = total as i32;
    let mut params = [
        &mut (input.as_ptr() as *mut c_void) as *mut _ as *mut c_void,
        &mut (output.as_mut_ptr()) as *mut _ as *mut c_void,
        &mut channels_i as *mut _ as *mut c_void,
        &mut frames_i as *mut _ as *mut c_void,
        &mut total_i as *mut _ as *mut c_void,
    ];
    let block = 256u32;
    let grid = (total as u32).div_ceil(block);
    kernel.launch((grid, 1, 1), (block, 1, 1), 0, &mut params)
}

fn launch_transpose_tc_to_ct(
    kernel: &HipFunction,
    input: &DeviceBuffer<f32>,
    output: &DeviceBuffer<f32>,
    frames: usize,
    channels: usize,
) -> Result<()> {
    let total = channels * frames;
    let mut frames_i = frames as i32;
    let mut channels_i = channels as i32;
    let mut total_i = total as i32;
    let mut params = [
        &mut (input.as_ptr() as *mut c_void) as *mut _ as *mut c_void,
        &mut (output.as_mut_ptr()) as *mut _ as *mut c_void,
        &mut frames_i as *mut _ as *mut c_void,
        &mut channels_i as *mut _ as *mut c_void,
        &mut total_i as *mut _ as *mut c_void,
    ];
    let block = 256u32;
    let grid = (total as u32).div_ceil(block);
    kernel.launch((grid, 1, 1), (block, 1, 1), 0, &mut params)
}

fn launch_rmsnorm(
    kernel: &HipFunction,
    input: *const c_void,
    gamma: *const c_void,
    mut output: *mut c_void,
    rows: usize,
    cols: usize,
    epsilon: f32,
) -> Result<()> {
    let mut rows_i = rows as i32;
    let mut cols_i = cols as i32;
    let mut epsilon_f = epsilon;
    let mut params = [
        &mut (input as *mut c_void) as *mut _ as *mut c_void,
        &mut (gamma as *mut c_void) as *mut _ as *mut c_void,
        &mut output as *mut _ as *mut c_void,
        &mut rows_i as *mut _ as *mut c_void,
        &mut cols_i as *mut _ as *mut c_void,
        &mut epsilon_f as *mut _ as *mut c_void,
    ];
    let block = 256u32;
    kernel.launch((rows as u32, 1, 1), (block, 1, 1), block * 4, &mut params)
}

fn launch_permute(
    kernel: &HipFunction,
    input: *const c_void,
    mut output: *mut c_void,
    steps: usize,
    heads: usize,
    head_dim: usize,
) -> Result<()> {
    let total = steps * heads * head_dim;
    let mut batch_i = 1i32;
    let mut steps_i = steps as i32;
    let mut heads_i = heads as i32;
    let mut head_dim_i = head_dim as i32;
    let mut total_i = total as i32;
    let mut params = [
        &mut (input as *mut c_void) as *mut _ as *mut c_void,
        &mut output as *mut _ as *mut c_void,
        &mut batch_i as *mut _ as *mut c_void,
        &mut steps_i as *mut _ as *mut c_void,
        &mut heads_i as *mut _ as *mut c_void,
        &mut head_dim_i as *mut _ as *mut c_void,
        &mut total_i as *mut _ as *mut c_void,
    ];
    let block = 256u32;
    let grid = (total as u32).div_ceil(block);
    kernel.launch((grid, 1, 1), (block, 1, 1), 0, &mut params)
}

fn launch_rope(
    kernel: &HipFunction,
    input: *const c_void,
    mut output: *mut c_void,
    total: usize,
    heads: usize,
    steps: usize,
    head_dim: usize,
    theta: f32,
) -> Result<()> {
    let mut total_i = total as i32;
    let mut heads_i = heads as i32;
    let mut steps_i = steps as i32;
    let mut head_dim_i = head_dim as i32;
    let mut offset_i = 0i32;
    let mut theta_f = theta;
    let mut params = [
        &mut (input as *mut c_void) as *mut _ as *mut c_void,
        &mut output as *mut _ as *mut c_void,
        &mut total_i as *mut _ as *mut c_void,
        &mut heads_i as *mut _ as *mut c_void,
        &mut steps_i as *mut _ as *mut c_void,
        &mut head_dim_i as *mut _ as *mut c_void,
        &mut offset_i as *mut _ as *mut c_void,
        &mut theta_f as *mut _ as *mut c_void,
    ];
    let block = 256u32;
    let grid = (total as u32).div_ceil(block);
    kernel.launch((grid, 1, 1), (block, 1, 1), 0, &mut params)
}

fn launch_attention_scores(
    kernel: &HipFunction,
    q: *const c_void,
    k: *const c_void,
    mut output: *mut c_void,
    steps: usize,
    head_dim: usize,
) -> Result<()> {
    let total = TRANSFORMER_HEADS * steps * steps;
    let mut batch_i = 1i32;
    let mut heads_i = TRANSFORMER_HEADS as i32;
    let mut query_steps_i = steps as i32;
    let mut key_steps_i = steps as i32;
    let mut head_dim_i = head_dim as i32;
    let mut offset_i = 0i32;
    let mut scale_f = 1.0f32 / (head_dim as f32).sqrt();
    let mut total_i = total as i32;
    let mut params = [
        &mut (q as *mut c_void) as *mut _ as *mut c_void,
        &mut (k as *mut c_void) as *mut _ as *mut c_void,
        &mut output as *mut _ as *mut c_void,
        &mut batch_i as *mut _ as *mut c_void,
        &mut heads_i as *mut _ as *mut c_void,
        &mut heads_i as *mut _ as *mut c_void,
        &mut query_steps_i as *mut _ as *mut c_void,
        &mut key_steps_i as *mut _ as *mut c_void,
        &mut head_dim_i as *mut _ as *mut c_void,
        &mut offset_i as *mut _ as *mut c_void,
        &mut scale_f as *mut _ as *mut c_void,
        &mut total_i as *mut _ as *mut c_void,
    ];
    let block = 256u32;
    let grid = (total as u32).div_ceil(block);
    kernel.launch((grid, 1, 1), (block, 1, 1), 0, &mut params)
}

fn launch_softmax(
    kernel: &HipFunction,
    input: *const c_void,
    mut output: *mut c_void,
    rows: usize,
    cols: usize,
) -> Result<()> {
    let mut rows_i = rows as i32;
    let mut cols_i = cols as i32;
    let mut active_cols_i = cols as i32;
    let mut params = [
        &mut (input as *mut c_void) as *mut _ as *mut c_void,
        &mut output as *mut _ as *mut c_void,
        &mut rows_i as *mut _ as *mut c_void,
        &mut cols_i as *mut _ as *mut c_void,
        &mut active_cols_i as *mut _ as *mut c_void,
    ];
    let block = 256u32;
    kernel.launch((rows as u32, 1, 1), (block, 1, 1), block * 4, &mut params)
}

fn launch_apply_value(
    kernel: &HipFunction,
    probs: *const c_void,
    v: *const c_void,
    mut output: *mut c_void,
    steps: usize,
    head_dim: usize,
) -> Result<()> {
    let total = steps * TRANSFORMER_HEADS * head_dim;
    let mut batch_i = 1i32;
    let mut heads_i = TRANSFORMER_HEADS as i32;
    let mut query_steps_i = steps as i32;
    let mut key_steps_i = steps as i32;
    let mut head_dim_i = head_dim as i32;
    let mut total_i = total as i32;
    let mut params = [
        &mut (probs as *mut c_void) as *mut _ as *mut c_void,
        &mut (v as *mut c_void) as *mut _ as *mut c_void,
        &mut output as *mut _ as *mut c_void,
        &mut batch_i as *mut _ as *mut c_void,
        &mut heads_i as *mut _ as *mut c_void,
        &mut heads_i as *mut _ as *mut c_void,
        &mut query_steps_i as *mut _ as *mut c_void,
        &mut key_steps_i as *mut _ as *mut c_void,
        &mut head_dim_i as *mut _ as *mut c_void,
        &mut total_i as *mut _ as *mut c_void,
    ];
    let block = 256u32;
    let grid = (total as u32).div_ceil(block);
    kernel.launch((grid, 1, 1), (block, 1, 1), 0, &mut params)
}

fn launch_ternary(
    kernel: &HipFunction,
    a: *const c_void,
    b: *const c_void,
    mut output: *mut c_void,
    total: usize,
) -> Result<()> {
    let mut total_i = total as i32;
    let mut params = [
        &mut (a as *mut c_void) as *mut _ as *mut c_void,
        &mut (b as *mut c_void) as *mut _ as *mut c_void,
        &mut output as *mut _ as *mut c_void,
        &mut total_i as *mut _ as *mut c_void,
    ];
    let block = 256u32;
    let grid = (total as u32).div_ceil(block);
    kernel.launch((grid, 1, 1), (block, 1, 1), 0, &mut params)
}

fn launch_scaled_add(
    kernel: &HipFunction,
    residual: &DeviceBuffer<f32>,
    update: &DeviceBuffer<f32>,
    scale: &DeviceBuffer<f32>,
    output: &DeviceBuffer<f32>,
    rows: usize,
    cols: usize,
) -> Result<()> {
    let total = rows * cols;
    let mut rows_i = rows as i32;
    let mut cols_i = cols as i32;
    let mut params = [
        &mut (residual.as_ptr() as *mut c_void) as *mut _ as *mut c_void,
        &mut (update.as_ptr() as *mut c_void) as *mut _ as *mut c_void,
        &mut (scale.as_ptr() as *mut c_void) as *mut _ as *mut c_void,
        &mut (output.as_mut_ptr()) as *mut _ as *mut c_void,
        &mut rows_i as *mut _ as *mut c_void,
        &mut cols_i as *mut _ as *mut c_void,
    ];
    let block = 256u32;
    let grid = (total as u32).div_ceil(block);
    kernel.launch((grid, 1, 1), (block, 1, 1), 0, &mut params)
}

fn launch_bias_add(
    kernel: &HipFunction,
    input: &DeviceBuffer<f32>,
    bias: &DeviceBuffer<f32>,
    output: &DeviceBuffer<f32>,
    rows: usize,
    cols: usize,
) -> Result<()> {
    let total = rows * cols;
    let mut rows_i = rows as i32;
    let mut cols_i = cols as i32;
    let mut params = [
        &mut (input.as_ptr() as *mut c_void) as *mut _ as *mut c_void,
        &mut (bias.as_ptr() as *mut c_void) as *mut _ as *mut c_void,
        &mut (output.as_mut_ptr()) as *mut _ as *mut c_void,
        &mut rows_i as *mut _ as *mut c_void,
        &mut cols_i as *mut _ as *mut c_void,
    ];
    let block = 256u32;
    let grid = (total as u32).div_ceil(block);
    kernel.launch((grid, 1, 1), (block, 1, 1), 0, &mut params)
}

fn launch_gelu(
    kernel: &HipFunction,
    input: &DeviceBuffer<f32>,
    output: &DeviceBuffer<f32>,
    total: usize,
) -> Result<()> {
    let mut total_i = total as i32;
    let mut params = [
        &mut (input.as_ptr() as *mut c_void) as *mut _ as *mut c_void,
        &mut (output.as_mut_ptr()) as *mut _ as *mut c_void,
        &mut total_i as *mut _ as *mut c_void,
    ];
    let block = 256u32;
    let grid = (total as u32).div_ceil(block);
    kernel.launch((grid, 1, 1), (block, 1, 1), 0, &mut params)
}

fn launch_layernorm(
    kernel: &HipFunction,
    input: &DeviceBuffer<f32>,
    gamma: &DeviceBuffer<f32>,
    beta: &DeviceBuffer<f32>,
    output: &DeviceBuffer<f32>,
    rows: usize,
    cols: usize,
) -> Result<()> {
    let mut rows_i = rows as i32;
    let mut cols_i = cols as i32;
    let mut epsilon = LAYERNORM_EPSILON;
    let mut params = [
        &mut (input.as_ptr() as *mut c_void) as *mut _ as *mut c_void,
        &mut (gamma.as_ptr() as *mut c_void) as *mut _ as *mut c_void,
        &mut (beta.as_ptr() as *mut c_void) as *mut _ as *mut c_void,
        &mut (output.as_mut_ptr()) as *mut _ as *mut c_void,
        &mut rows_i as *mut _ as *mut c_void,
        &mut cols_i as *mut _ as *mut c_void,
        &mut epsilon as *mut _ as *mut c_void,
    ];
    let block = 256u32;
    kernel.launch((rows as u32, 1, 1), (block, 1, 1), block * 8, &mut params)
}

fn launch_transconv1d(
    kernel: &HipFunction,
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<f32>,
    bias: &DeviceBuffer<f32>,
    output: &DeviceBuffer<f32>,
    in_frames: usize,
    out_frames: usize,
) -> Result<()> {
    let total = UPSAMPLE_CHANNELS * out_frames;
    let mut in_frames_i = in_frames as i32;
    let mut out_frames_i = out_frames as i32;
    let mut channels_i = UPSAMPLE_CHANNELS as i32;
    let mut kernel_size_i = UPSAMPLE_RATIO as i32;
    let mut stride_i = UPSAMPLE_RATIO as i32;
    let mut params = [
        &mut (input.as_ptr() as *mut c_void) as *mut _ as *mut c_void,
        &mut (weight.as_ptr() as *mut c_void) as *mut _ as *mut c_void,
        &mut (bias.as_ptr() as *mut c_void) as *mut _ as *mut c_void,
        &mut (output.as_mut_ptr()) as *mut _ as *mut c_void,
        &mut in_frames_i as *mut _ as *mut c_void,
        &mut out_frames_i as *mut _ as *mut c_void,
        &mut channels_i as *mut _ as *mut c_void,
        &mut kernel_size_i as *mut _ as *mut c_void,
        &mut stride_i as *mut _ as *mut c_void,
    ];
    let block = 256u32;
    let grid = (total as u32).div_ceil(block);
    kernel.launch((grid, 1, 1), (block, 1, 1), 0, &mut params)
}

fn launch_depthwise_conv1d(
    kernel: &HipFunction,
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<f32>,
    bias: &DeviceBuffer<f32>,
    output: &DeviceBuffer<f32>,
    frames: usize,
) -> Result<()> {
    let total = UPSAMPLE_CHANNELS * frames;
    let mut frames_i = frames as i32;
    let mut channels_i = UPSAMPLE_CHANNELS as i32;
    let mut kernel_size_i = CONVNEXT_KERNEL as i32;
    let mut padding_i = CONVNEXT_PADDING as i32;
    let mut params = [
        &mut (input.as_ptr() as *mut c_void) as *mut _ as *mut c_void,
        &mut (weight.as_ptr() as *mut c_void) as *mut _ as *mut c_void,
        &mut (bias.as_ptr() as *mut c_void) as *mut _ as *mut c_void,
        &mut (output.as_mut_ptr()) as *mut _ as *mut c_void,
        &mut frames_i as *mut _ as *mut c_void,
        &mut channels_i as *mut _ as *mut c_void,
        &mut kernel_size_i as *mut _ as *mut c_void,
        &mut padding_i as *mut _ as *mut c_void,
    ];
    let block = 256u32;
    let grid = (total as u32).div_ceil(block);
    kernel.launch((grid, 1, 1), (block, 1, 1), 0, &mut params)
}

fn launch_convnext_residual(
    kernel: &HipFunction,
    residual: &DeviceBuffer<f32>,
    update_tc: &DeviceBuffer<f32>,
    gamma: &DeviceBuffer<f32>,
    output: &DeviceBuffer<f32>,
    frames: usize,
) -> Result<()> {
    let total = UPSAMPLE_CHANNELS * frames;
    let mut channels_i = UPSAMPLE_CHANNELS as i32;
    let mut frames_i = frames as i32;
    let mut total_i = total as i32;
    let mut params = [
        &mut (residual.as_ptr() as *mut c_void) as *mut _ as *mut c_void,
        &mut (update_tc.as_ptr() as *mut c_void) as *mut _ as *mut c_void,
        &mut (gamma.as_ptr() as *mut c_void) as *mut _ as *mut c_void,
        &mut (output.as_mut_ptr()) as *mut _ as *mut c_void,
        &mut channels_i as *mut _ as *mut c_void,
        &mut frames_i as *mut _ as *mut c_void,
        &mut total_i as *mut _ as *mut c_void,
    ];
    let block = 256u32;
    let grid = (total as u32).div_ceil(block);
    kernel.launch((grid, 1, 1), (block, 1, 1), 0, &mut params)
}

fn launch_snake(
    kernel: &HipFunction,
    input: &DeviceBuffer<f32>,
    weights: &SnakeWeights,
    output: &DeviceBuffer<f32>,
    frames: usize,
) -> Result<()> {
    let total = weights.channels * frames;
    let mut channels_i = weights.channels as i32;
    let mut frames_i = frames as i32;
    let mut total_i = total as i32;
    let mut params = [
        &mut (input.as_ptr() as *mut c_void) as *mut _ as *mut c_void,
        &mut (weights.alpha.as_ptr() as *mut c_void) as *mut _ as *mut c_void,
        &mut (weights.beta.as_ptr() as *mut c_void) as *mut _ as *mut c_void,
        &mut (output.as_mut_ptr()) as *mut _ as *mut c_void,
        &mut channels_i as *mut _ as *mut c_void,
        &mut frames_i as *mut _ as *mut c_void,
        &mut total_i as *mut _ as *mut c_void,
    ];
    let block = 256u32;
    let grid = (total as u32).div_ceil(block);
    kernel.launch((grid, 1, 1), (block, 1, 1), 0, &mut params)
}

fn launch_causal_conv1d_dilated(
    kernel: &HipFunction,
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<f32>,
    bias: &DeviceBuffer<f32>,
    output: &DeviceBuffer<f32>,
    frames: usize,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    dilation: usize,
    causal_padding: usize,
) -> Result<()> {
    let total = frames * out_channels;
    let mut frames_i = frames as i32;
    let mut in_channels_i = in_channels as i32;
    let mut out_channels_i = out_channels as i32;
    let mut kernel_size_i = kernel_size as i32;
    let mut dilation_i = dilation as i32;
    let mut causal_padding_i = causal_padding as i32;
    let mut params = [
        &mut (input.as_ptr() as *mut c_void) as *mut _ as *mut c_void,
        &mut (weight.as_ptr() as *mut c_void) as *mut _ as *mut c_void,
        &mut (bias.as_ptr() as *mut c_void) as *mut _ as *mut c_void,
        &mut (output.as_mut_ptr()) as *mut _ as *mut c_void,
        &mut frames_i as *mut _ as *mut c_void,
        &mut in_channels_i as *mut _ as *mut c_void,
        &mut out_channels_i as *mut _ as *mut c_void,
        &mut kernel_size_i as *mut _ as *mut c_void,
        &mut dilation_i as *mut _ as *mut c_void,
        &mut causal_padding_i as *mut _ as *mut c_void,
    ];
    let block = 256u32;
    let grid = (total as u32).div_ceil(block);
    kernel.launch((grid, 1, 1), (block, 1, 1), 0, &mut params)
}

fn launch_transconv1d_channels(
    kernel: &HipFunction,
    input: &DeviceBuffer<f32>,
    weight: &DeviceBuffer<f32>,
    bias: &DeviceBuffer<f32>,
    output: &DeviceBuffer<f32>,
    in_frames: usize,
    out_frames: usize,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
) -> Result<()> {
    let total = out_frames * out_channels;
    let mut in_frames_i = in_frames as i32;
    let mut out_frames_i = out_frames as i32;
    let mut in_channels_i = in_channels as i32;
    let mut out_channels_i = out_channels as i32;
    let mut kernel_size_i = kernel_size as i32;
    let mut stride_i = stride as i32;
    let mut params = [
        &mut (input.as_ptr() as *mut c_void) as *mut _ as *mut c_void,
        &mut (weight.as_ptr() as *mut c_void) as *mut _ as *mut c_void,
        &mut (bias.as_ptr() as *mut c_void) as *mut _ as *mut c_void,
        &mut (output.as_mut_ptr()) as *mut _ as *mut c_void,
        &mut in_frames_i as *mut _ as *mut c_void,
        &mut out_frames_i as *mut _ as *mut c_void,
        &mut in_channels_i as *mut _ as *mut c_void,
        &mut out_channels_i as *mut _ as *mut c_void,
        &mut kernel_size_i as *mut _ as *mut c_void,
        &mut stride_i as *mut _ as *mut c_void,
    ];
    let block = 256u32;
    let grid = (total as u32).div_ceil(block);
    kernel.launch((grid, 1, 1), (block, 1, 1), 0, &mut params)
}

fn launch_clamp(
    kernel: &HipFunction,
    input: &DeviceBuffer<f32>,
    output: &DeviceBuffer<f32>,
    total: usize,
    min_value: f32,
    max_value: f32,
) -> Result<()> {
    let mut total_i = total as i32;
    let mut min_value_f = min_value;
    let mut max_value_f = max_value;
    let mut params = [
        &mut (input.as_ptr() as *mut c_void) as *mut _ as *mut c_void,
        &mut (output.as_mut_ptr()) as *mut _ as *mut c_void,
        &mut total_i as *mut _ as *mut c_void,
        &mut min_value_f as *mut _ as *mut c_void,
        &mut max_value_f as *mut _ as *mut c_void,
    ];
    let block = 256u32;
    let grid = (total as u32).div_ceil(block);
    kernel.launch((grid, 1, 1), (block, 1, 1), 0, &mut params)
}

fn load_normalized_codebook(archive: &TensorArchive, prefix: &str) -> Result<Vec<f32>> {
    let embedding = tensor_f32(
        archive.tensor(&format!("{prefix}.embedding_sum"))?,
        &format!("{prefix}.embedding_sum"),
        &[CODEBOOK_SIZE, CODEBOOK_DIM],
    )?;
    let usage = tensor_f32(
        archive.tensor(&format!("{prefix}.cluster_usage"))?,
        &format!("{prefix}.cluster_usage"),
        &[CODEBOOK_SIZE],
    )?;
    let mut normalized = embedding;
    for code in 0..CODEBOOK_SIZE {
        let scale = usage[code].max(1e-7);
        for dim in 0..CODEBOOK_DIM {
            normalized[code * CODEBOOK_DIM + dim] /= scale;
        }
    }
    Ok(normalized)
}

fn conv1x1_weight(archive: &TensorArchive, name: &str) -> Result<Vec<f32>> {
    let tensor = archive.tensor(name)?;
    let shape = tensor.shape();
    if shape != [PROJECTED_DIM, CODEBOOK_DIM, 1] {
        return Err(Error::InvalidInput(format!(
            "{name} shape {shape:?}; expected [{PROJECTED_DIM}, {CODEBOOK_DIM}, 1]"
        )));
    }
    tensor_to_f32(
        name,
        tensor.dtype(),
        tensor.data(),
        PROJECTED_DIM * CODEBOOK_DIM,
    )
}

fn linear_weight_transposed(
    archive: &TensorArchive,
    name: &str,
) -> Result<(Vec<f32>, usize, usize)> {
    let tensor = archive.tensor(name)?;
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
    Ok((transposed, in_dim, out_dim))
}

fn conv1d_weight(
    archive: &TensorArchive,
    name: &str,
    out_channels: usize,
    in_channels: usize,
    kernel_size: usize,
) -> Result<Vec<f32>> {
    tensor_f32(
        archive.tensor(name)?,
        name,
        &[out_channels, in_channels, kernel_size],
    )
}

fn vector_f32(archive: &TensorArchive, name: &str, len: usize) -> Result<Vec<f32>> {
    tensor_f32(archive.tensor(name)?, name, &[len])
}

fn tensor_f32(tensor: TensorView<'_>, name: &str, expected_shape: &[usize]) -> Result<Vec<f32>> {
    let shape = tensor.shape();
    if shape != expected_shape {
        return Err(Error::InvalidInput(format!(
            "{name} shape {shape:?}; expected {expected_shape:?}"
        )));
    }
    tensor_to_f32(
        name,
        tensor.dtype(),
        tensor.data(),
        expected_shape.iter().product(),
    )
}
