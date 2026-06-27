use std::path::Path;

use burn::backend::Flex;
use burn::module::Param;
use burn::nn::conv::{Conv1d, Conv1dConfig, ConvTranspose1d, ConvTranspose1dConfig};
use burn::nn::{
    LayerNorm, LayerNormConfig, Linear, LinearConfig, PaddingConfig1d, RmsNorm as BurnRmsNorm,
    RmsNormConfig,
};
use burn::tensor::activation::{gelu, silu, softmax};
use burn::tensor::backend::Backend;
use burn::tensor::ops::PadMode;
use burn::tensor::{Bool, Int, Tensor, TensorData};
use safetensors::SafeTensors;
use safetensors::tensor::{Dtype, TensorView};
use serde::Deserialize;

use crate::error::{Error, Result};

pub type CodecBackend = Flex<f32>;

#[derive(Clone, Debug)]
pub struct Decoder12HzConfig {
    pub codebook_dim: usize,
    pub projected_dim: usize,
    pub latent_dim: usize,
    pub transformer_dim: usize,
    pub transformer_layers: usize,
    pub transformer_heads: usize,
    pub transformer_head_dim: usize,
    pub transformer_intermediate: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub upsampling_ratios: Vec<usize>,
    pub decoder_dim: usize,
    pub decoder_upsample_rates: Vec<usize>,
    pub num_quantizers: usize,
    pub codebook_size: usize,
}

impl Decoder12HzConfig {
    pub fn from_speech_tokenizer_dir(path: &Path) -> Result<Self> {
        let config_path = path.join("config.json");
        let bytes = std::fs::read(&config_path).map_err(|err| {
            Error::InvalidInput(format!("failed to read {}: {err}", config_path.display()))
        })?;
        let config: SpeechTokenizerConfig = serde_json::from_slice(&bytes).map_err(|err| {
            Error::InvalidInput(format!("failed to parse {}: {err}", config_path.display()))
        })?;
        let decoder = config.decoder_config;
        Ok(Self {
            // The 12 Hz tokenizer config reports decoder_config.codebook_dim=512,
            // but the shipped RVQ codebooks and output projection inputs are 256.
            codebook_dim: 256,
            projected_dim: decoder.vector_quantization_hidden_dimension,
            latent_dim: decoder.latent_dim,
            transformer_dim: decoder.hidden_size,
            transformer_layers: decoder.num_hidden_layers,
            transformer_heads: decoder.num_attention_heads,
            transformer_head_dim: decoder.head_dim,
            transformer_intermediate: decoder.intermediate_size,
            rms_norm_eps: decoder.rms_norm_eps,
            rope_theta: decoder.rope_theta,
            upsampling_ratios: decoder.upsampling_ratios,
            decoder_dim: decoder.decoder_dim,
            decoder_upsample_rates: decoder.upsample_rates,
            num_quantizers: decoder.num_quantizers,
            codebook_size: decoder.codebook_size,
        })
    }
}

#[derive(Clone, Debug)]
pub struct Decoder12Hz<B: Backend> {
    pub config: Decoder12HzConfig,
    pub first_codebook: Param<Tensor<B, 2>>,
    pub rest_codebooks: Vec<Param<Tensor<B, 2>>>,
    pub first_output_proj: Conv1d<B>,
    pub rest_output_proj: Conv1d<B>,
    pub pre_conv: CausalConv1d<B>,
    pub input_proj: Linear<B>,
    pub pre_transformer_layers: Vec<CodecTransformerLayer<B>>,
    pub pre_transformer_norm: RmsNorm<B>,
    pub output_proj: Linear<B>,
    pub upsample_stages: Vec<UpsampleStage<B>>,
    pub decoder_init_conv: CausalConv1d<B>,
    pub decoder_blocks: Vec<DecoderBlock<B>>,
    pub final_snake: SnakeBeta<B>,
    pub final_conv: CausalConv1d<B>,
}

impl<B: Backend> Decoder12Hz<B> {
    pub fn new(config: Decoder12HzConfig, device: &B::Device) -> Self {
        let conv_1x1 = |in_channels, out_channels| {
            Conv1dConfig::new(in_channels, out_channels, 1)
                .with_bias(false)
                .init(device)
        };

        Self {
            first_codebook: Param::from_tensor(Tensor::zeros(
                [config.codebook_size, config.codebook_dim],
                device,
            )),
            rest_codebooks: (0..config.num_quantizers - 1)
                .map(|_| {
                    Param::from_tensor(Tensor::zeros(
                        [config.codebook_size, config.codebook_dim],
                        device,
                    ))
                })
                .collect(),
            first_output_proj: conv_1x1(config.codebook_dim, config.projected_dim),
            rest_output_proj: conv_1x1(config.codebook_dim, config.projected_dim),
            pre_conv: CausalConv1d::new(config.projected_dim, config.latent_dim, 3, 1, 1, device),
            input_proj: LinearConfig::new(config.latent_dim, config.transformer_dim).init(device),
            pre_transformer_layers: (0..config.transformer_layers)
                .map(|_| CodecTransformerLayer::new(&config, device))
                .collect(),
            pre_transformer_norm: RmsNorm::new(config.transformer_dim, config.rms_norm_eps, device),
            output_proj: LinearConfig::new(config.transformer_dim, config.latent_dim).init(device),
            upsample_stages: config
                .upsampling_ratios
                .iter()
                .copied()
                .map(|ratio| UpsampleStage::new(config.latent_dim, ratio, device))
                .collect(),
            decoder_init_conv: CausalConv1d::new(
                config.latent_dim,
                config.decoder_dim,
                7,
                1,
                1,
                device,
            ),
            decoder_blocks: decoder_block_channels(&config)
                .into_iter()
                .zip(config.decoder_upsample_rates.iter().copied())
                .map(|((in_channels, out_channels), rate)| {
                    DecoderBlock::new(in_channels, out_channels, rate, device)
                })
                .collect(),
            final_snake: SnakeBeta::new(final_decoder_channels(&config), device),
            final_conv: CausalConv1d::new(final_decoder_channels(&config), 1, 7, 1, 1, device),
            config,
        }
    }

    pub fn decode_waveform(&self, codes: &[Vec<u32>], device: &B::Device) -> Tensor<B, 3> {
        let mut hidden = self.decode_upsampled(codes, device);
        hidden = self.decoder_init_conv.forward(hidden);
        for block in &self.decoder_blocks {
            hidden = block.forward(hidden);
        }
        hidden = self.final_snake.forward(hidden);
        self.final_conv.forward(hidden).clamp(-1.0, 1.0)
    }

    fn decode_quantized(&self, codes: &[Vec<u32>], device: &B::Device) -> Tensor<B, 3> {
        let code_tensor = frame_codes_to_tensor::<B>(codes, self.config.num_quantizers, device);
        let [batch, quantizers, frames] = code_tensor.dims();
        assert_eq!(quantizers, self.config.num_quantizers);

        let first_codes = code_tensor.clone().narrow(1, 0, 1).squeeze_dim::<2>(1);
        let first_embed = embedding_lookup(self.first_codebook.val(), first_codes);
        let first = self
            .first_output_proj
            .forward(first_embed.permute([0, 2, 1]));

        let mut rest_embed =
            Tensor::<B, 3>::zeros([batch, frames, self.config.codebook_dim], device);
        for index in 0..self.config.num_quantizers - 1 {
            let q_codes = code_tensor
                .clone()
                .narrow(1, index + 1, 1)
                .squeeze_dim::<2>(1);
            rest_embed = rest_embed + embedding_lookup(self.rest_codebooks[index].val(), q_codes);
        }
        let rest = self.rest_output_proj.forward(rest_embed.permute([0, 2, 1]));
        first + rest
    }

    fn decode_upsampled(&self, codes: &[Vec<u32>], device: &B::Device) -> Tensor<B, 3> {
        let mut hidden = self.pre_conv.forward(self.decode_quantized(codes, device));
        hidden = self.run_pre_transformer(hidden);
        for stage in &self.upsample_stages {
            hidden = stage.forward(hidden);
        }
        hidden
    }

    fn run_pre_transformer(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch, _channels, frames] = hidden.dims();
        let mut hidden = linear_3d(&self.input_proj, hidden.permute([0, 2, 1]));
        let mask = causal_mask::<B>(batch, frames, &hidden.device());
        for layer in &self.pre_transformer_layers {
            hidden = layer.forward(hidden, Some(mask.clone()));
        }
        let hidden = self.pre_transformer_norm.forward_3d(hidden);
        linear_3d(&self.output_proj, hidden).permute([0, 2, 1])
    }
}

pub fn load_decoder_from_model_dir(model_dir: &Path) -> Result<Decoder12Hz<CodecBackend>> {
    let speech_dir = model_dir.join("speech_tokenizer");
    let device = Default::default();
    let config = Decoder12HzConfig::from_speech_tokenizer_dir(&speech_dir)?;
    let mut decoder = Decoder12Hz::<CodecBackend>::new(config, &device);
    let archive = TensorArchive::open(&speech_dir.join("model.safetensors"))?;
    load_decoder_12hz(&mut decoder, &archive, &device)?;
    Ok(decoder)
}

pub fn decode_codes_to_waveform(model_dir: &Path, codes: &[Vec<u32>]) -> Result<Vec<f32>> {
    let decoder = load_decoder_from_model_dir(model_dir)?;
    let device = Default::default();
    Ok(decoder
        .decode_waveform(codes, &device)
        .into_data()
        .to_vec::<f32>()
        .map_err(|err| Error::InvalidInput(format!("failed to read waveform tensor: {err}")))?)
}

pub fn write_wav(path: &Path, samples: &[f32], sample_rate: u32, gain: f32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(path, spec).map_err(|err| {
        Error::InvalidInput(format!("failed to create {}: {err}", path.display()))
    })?;
    for &sample in samples {
        let value = (sample * gain).clamp(-1.0, 1.0);
        writer
            .write_sample((value * i16::MAX as f32) as i16)
            .map_err(|err| Error::InvalidInput(format!("failed to write wav sample: {err}")))?;
    }
    writer.finalize().map_err(|err| {
        Error::InvalidInput(format!("failed to finalize {}: {err}", path.display()))
    })?;
    Ok(())
}

#[derive(Clone, Debug)]
pub struct CausalConv1d<B: Backend> {
    pub conv: Conv1d<B>,
    pub causal_padding: usize,
}

impl<B: Backend> CausalConv1d<B> {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        dilation: usize,
        groups: usize,
        device: &B::Device,
    ) -> Self {
        let conv = Conv1dConfig::new(in_channels, out_channels, kernel_size)
            .with_dilation(dilation)
            .with_groups(groups)
            .with_padding(PaddingConfig1d::Valid)
            .init(device);
        Self {
            conv,
            causal_padding: dilation * kernel_size.saturating_sub(1),
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let input = if self.causal_padding > 0 {
            input.pad(
                [(0, 0), (0, 0), (self.causal_padding, 0)],
                PadMode::Constant(0.0),
            )
        } else {
            input
        };
        self.conv.forward(input)
    }
}

#[derive(Clone, Debug)]
pub struct CausalTransConv1d<B: Backend> {
    pub conv: ConvTranspose1d<B>,
    pub right_trim: usize,
}

impl<B: Backend> CausalTransConv1d<B> {
    pub fn new_channels(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        device: &B::Device,
    ) -> Self {
        let conv = ConvTranspose1dConfig::new([in_channels, out_channels], kernel_size)
            .with_stride(stride)
            .init(device);
        Self {
            conv,
            right_trim: kernel_size.saturating_sub(stride),
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let output = self.conv.forward(input);
        if self.right_trim == 0 {
            return output;
        }
        let length = output.dims()[2].saturating_sub(self.right_trim);
        output.narrow(2, 0, length)
    }
}

#[derive(Clone, Debug)]
pub struct SnakeBeta<B: Backend> {
    pub alpha: Param<Tensor<B, 1>>,
    pub beta: Param<Tensor<B, 1>>,
}

impl<B: Backend> SnakeBeta<B> {
    pub fn new(channels: usize, device: &B::Device) -> Self {
        Self {
            alpha: Param::from_tensor(Tensor::zeros([channels], device)),
            beta: Param::from_tensor(Tensor::zeros([channels], device)),
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let alpha = self.alpha.val().exp().reshape([1, self.alpha.dims()[0], 1]);
        let beta = self.beta.val().exp().reshape([1, self.beta.dims()[0], 1]);
        let sin = (input.clone() * alpha).sin().powf_scalar(2.0);
        input + sin / (beta + 1e-9)
    }
}

#[derive(Clone, Debug)]
pub struct ConvNeXtBlock<B: Backend> {
    pub dwconv: CausalConv1d<B>,
    pub norm: LayerNorm<B>,
    pub pwconv1: Linear<B>,
    pub pwconv2: Linear<B>,
    pub gamma: Param<Tensor<B, 1>>,
}

impl<B: Backend> ConvNeXtBlock<B> {
    pub fn new(dim: usize, device: &B::Device) -> Self {
        Self {
            dwconv: CausalConv1d::new(dim, dim, 7, 1, dim, device),
            norm: LayerNormConfig::new(dim).with_epsilon(1e-6).init(device),
            pwconv1: LinearConfig::new(dim, 4 * dim).init(device),
            pwconv2: LinearConfig::new(4 * dim, dim).init(device),
            gamma: Param::from_tensor(Tensor::zeros([dim], device)),
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = input.clone();
        let hidden = self.dwconv.forward(input).permute([0, 2, 1]);
        let hidden = self.norm.forward(hidden);
        let hidden = linear_3d(&self.pwconv2, gelu(linear_3d(&self.pwconv1, hidden)));
        let dim = self.gamma.dims()[0];
        residual + (hidden * self.gamma.val().reshape([1, 1, dim])).permute([0, 2, 1])
    }
}

#[derive(Clone, Debug)]
pub struct UpsampleStage<B: Backend> {
    pub trans_conv: CausalTransConv1d<B>,
    pub convnext: ConvNeXtBlock<B>,
}

impl<B: Backend> UpsampleStage<B> {
    pub fn new(channels: usize, ratio: usize, device: &B::Device) -> Self {
        Self {
            trans_conv: CausalTransConv1d::new_channels(channels, channels, ratio, ratio, device),
            convnext: ConvNeXtBlock::new(channels, device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        self.convnext.forward(self.trans_conv.forward(input))
    }
}

#[derive(Clone, Debug)]
pub struct ResidualUnit<B: Backend> {
    pub act1: SnakeBeta<B>,
    pub conv1: CausalConv1d<B>,
    pub act2: SnakeBeta<B>,
    pub conv2: CausalConv1d<B>,
}

impl<B: Backend> ResidualUnit<B> {
    pub fn new(channels: usize, dilation: usize, device: &B::Device) -> Self {
        Self {
            act1: SnakeBeta::new(channels, device),
            conv1: CausalConv1d::new(channels, channels, 7, dilation, 1, device),
            act2: SnakeBeta::new(channels, device),
            conv2: CausalConv1d::new(channels, channels, 1, 1, 1, device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let residual = input.clone();
        let hidden = self.act1.forward(input);
        let hidden = self.conv1.forward(hidden);
        let hidden = self.act2.forward(hidden);
        residual + self.conv2.forward(hidden)
    }
}

#[derive(Clone, Debug)]
pub struct DecoderBlock<B: Backend> {
    pub snake: SnakeBeta<B>,
    pub upsample: CausalTransConv1d<B>,
    pub res1: ResidualUnit<B>,
    pub res2: ResidualUnit<B>,
    pub res3: ResidualUnit<B>,
}

impl<B: Backend> DecoderBlock<B> {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        upsample_rate: usize,
        device: &B::Device,
    ) -> Self {
        Self {
            snake: SnakeBeta::new(in_channels, device),
            upsample: CausalTransConv1d::new_channels(
                in_channels,
                out_channels,
                2 * upsample_rate,
                upsample_rate,
                device,
            ),
            res1: ResidualUnit::new(out_channels, 1, device),
            res2: ResidualUnit::new(out_channels, 3, device),
            res3: ResidualUnit::new(out_channels, 9, device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let hidden = self.snake.forward(input);
        let hidden = self.upsample.forward(hidden);
        let hidden = self.res1.forward(hidden);
        let hidden = self.res2.forward(hidden);
        self.res3.forward(hidden)
    }
}

#[derive(Clone, Debug)]
pub struct CodecTransformerLayer<B: Backend> {
    pub input_layernorm: RmsNorm<B>,
    pub self_attn: CodecAttention<B>,
    pub self_attn_layer_scale: Param<Tensor<B, 1>>,
    pub post_attention_layernorm: RmsNorm<B>,
    pub gate_proj: Linear<B>,
    pub up_proj: Linear<B>,
    pub down_proj: Linear<B>,
    pub mlp_layer_scale: Param<Tensor<B, 1>>,
}

impl<B: Backend> CodecTransformerLayer<B> {
    pub fn new(config: &Decoder12HzConfig, device: &B::Device) -> Self {
        Self {
            input_layernorm: RmsNorm::new(config.transformer_dim, config.rms_norm_eps, device),
            self_attn: CodecAttention::new(config, device),
            self_attn_layer_scale: Param::from_tensor(Tensor::from_data(
                TensorData::new(vec![0.01; config.transformer_dim], [config.transformer_dim]),
                device,
            )),
            post_attention_layernorm: RmsNorm::new(
                config.transformer_dim,
                config.rms_norm_eps,
                device,
            ),
            gate_proj: LinearConfig::new(config.transformer_dim, config.transformer_intermediate)
                .with_bias(false)
                .init(device),
            up_proj: LinearConfig::new(config.transformer_dim, config.transformer_intermediate)
                .with_bias(false)
                .init(device),
            down_proj: LinearConfig::new(config.transformer_intermediate, config.transformer_dim)
                .with_bias(false)
                .init(device),
            mlp_layer_scale: Param::from_tensor(Tensor::from_data(
                TensorData::new(vec![0.01; config.transformer_dim], [config.transformer_dim]),
                device,
            )),
        }
    }

    pub fn forward(&self, hidden: Tensor<B, 3>, mask: Option<Tensor<B, 4, Bool>>) -> Tensor<B, 3> {
        let residual = hidden.clone();
        let attn = self
            .self_attn
            .forward(self.input_layernorm.forward_3d(hidden), mask);
        let dim = self.self_attn_layer_scale.val().dims()[0];
        let hidden = residual + attn * self.self_attn_layer_scale.val().reshape([1, 1, dim]);
        let residual = hidden.clone();
        let normed = self.post_attention_layernorm.forward_3d(hidden);
        let mlp = linear_3d(
            &self.down_proj,
            silu(linear_3d(&self.gate_proj, normed.clone())) * linear_3d(&self.up_proj, normed),
        );
        let dim = self.mlp_layer_scale.val().dims()[0];
        residual + mlp * self.mlp_layer_scale.val().reshape([1, 1, dim])
    }
}

#[derive(Clone, Debug)]
pub struct CodecAttention<B: Backend> {
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub o_proj: Linear<B>,
    pub heads: usize,
    pub head_dim: usize,
    pub rope: RotaryEmbedding,
}

impl<B: Backend> CodecAttention<B> {
    pub fn new(config: &Decoder12HzConfig, device: &B::Device) -> Self {
        let projected = config.transformer_heads * config.transformer_head_dim;
        Self {
            q_proj: LinearConfig::new(config.transformer_dim, projected)
                .with_bias(false)
                .init(device),
            k_proj: LinearConfig::new(config.transformer_dim, projected)
                .with_bias(false)
                .init(device),
            v_proj: LinearConfig::new(config.transformer_dim, projected)
                .with_bias(false)
                .init(device),
            o_proj: LinearConfig::new(projected, config.transformer_dim)
                .with_bias(false)
                .init(device),
            heads: config.transformer_heads,
            head_dim: config.transformer_head_dim,
            rope: RotaryEmbedding::new(config.transformer_head_dim, config.rope_theta),
        }
    }

    pub fn forward(&self, hidden: Tensor<B, 3>, mask: Option<Tensor<B, 4, Bool>>) -> Tensor<B, 3> {
        let [batch, frames, _] = hidden.dims();
        let q = linear_3d(&self.q_proj, hidden.clone())
            .reshape([batch, frames, self.heads, self.head_dim])
            .permute([0, 2, 1, 3]);
        let k = linear_3d(&self.k_proj, hidden.clone())
            .reshape([batch, frames, self.heads, self.head_dim])
            .permute([0, 2, 1, 3]);
        let v = linear_3d(&self.v_proj, hidden)
            .reshape([batch, frames, self.heads, self.head_dim])
            .permute([0, 2, 1, 3]);
        let q = self.rope.apply_bhsd(q, 0);
        let k = self.rope.apply_bhsd(k, 0);
        let mut scores = q
            .matmul(k.transpose())
            .div_scalar((self.head_dim as f32).sqrt());
        if let Some(mask) = mask {
            scores = scores.mask_fill(mask, f32::NEG_INFINITY);
        }
        let attended = softmax(scores, 3).matmul(v);
        linear_3d(
            &self.o_proj,
            attended
                .permute([0, 2, 1, 3])
                .reshape([batch, frames, self.heads * self.head_dim]),
        )
    }
}

#[derive(Clone, Debug)]
pub struct RmsNorm<B: Backend> {
    pub inner: BurnRmsNorm<B>,
}

impl<B: Backend> RmsNorm<B> {
    pub fn new(features: usize, epsilon: f64, device: &B::Device) -> Self {
        Self {
            inner: RmsNormConfig::new(features)
                .with_epsilon(epsilon)
                .init(device),
        }
    }

    pub fn forward_3d(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        self.inner.forward(input)
    }
}

#[derive(Clone, Debug)]
pub struct RotaryEmbedding {
    inv_freq: Vec<f32>,
}

impl RotaryEmbedding {
    pub fn new(head_dim: usize, theta: f64) -> Self {
        let half = head_dim / 2;
        let inv_freq = (0..half)
            .map(|idx| 1.0 / theta.powf((idx as f64) * 2.0 / head_dim as f64) as f32)
            .collect();
        Self { inv_freq }
    }

    pub fn apply_bhsd<B: Backend>(&self, input: Tensor<B, 4>, offset: usize) -> Tensor<B, 4> {
        let device = input.device();
        let [batch, heads, steps, head_dim] = input.dims();
        let data = input.into_data().to_vec::<f32>().unwrap();
        let mut output = vec![0.0; data.len()];
        let half = head_dim / 2;
        for b in 0..batch {
            for h in 0..heads {
                for s in 0..steps {
                    let pos = (offset + s) as f32;
                    let base = ((b * heads + h) * steps + s) * head_dim;
                    for i in 0..half {
                        let angle = pos * self.inv_freq[i];
                        let (sin, cos) = angle.sin_cos();
                        let x1 = data[base + i];
                        let x2 = data[base + i + half];
                        output[base + i] = x1 * cos - x2 * sin;
                        output[base + i + half] = x2 * cos + x1 * sin;
                    }
                }
            }
        }
        Tensor::from_data(
            TensorData::new(output, [batch, heads, steps, head_dim]),
            &device,
        )
    }
}

fn decoder_block_channels(config: &Decoder12HzConfig) -> Vec<(usize, usize)> {
    let mut channels = config.decoder_dim;
    config
        .decoder_upsample_rates
        .iter()
        .map(|_| {
            let in_channels = channels;
            channels /= 2;
            (in_channels, channels)
        })
        .collect()
}

fn final_decoder_channels(config: &Decoder12HzConfig) -> usize {
    decoder_block_channels(config)
        .last()
        .map(|(_, out)| *out)
        .unwrap_or(config.decoder_dim)
}

fn linear_3d<B: Backend>(linear: &Linear<B>, input: Tensor<B, 3>) -> Tensor<B, 3> {
    let [batch, steps, in_dim] = input.dims();
    let out_dim = linear.weight.dims()[1];
    let output = linear.forward(input.reshape([batch * steps, in_dim]));
    output.reshape([batch, steps, out_dim])
}

fn causal_mask<B: Backend>(batch: usize, frames: usize, device: &B::Device) -> Tensor<B, 4, Bool> {
    let mut data = Vec::with_capacity(batch * frames * frames);
    for _ in 0..batch {
        for q in 0..frames {
            for k in 0..frames {
                data.push(k > q);
            }
        }
    }
    Tensor::from_bool(TensorData::new(data, [batch, 1, frames, frames]), device)
}

fn frame_codes_to_tensor<B: Backend>(
    codes: &[Vec<u32>],
    num_quantizers: usize,
    device: &B::Device,
) -> Tensor<B, 3, Int> {
    let frames = codes.len();
    let mut data = Vec::with_capacity(num_quantizers * frames);
    for q in 0..num_quantizers {
        for frame in codes {
            assert_eq!(frame.len(), num_quantizers);
            data.push((frame[q] % 2048) as i64);
        }
    }
    Tensor::from_data(TensorData::new(data, [1, num_quantizers, frames]), device)
}

fn embedding_lookup<B: Backend>(codebook: Tensor<B, 2>, codes: Tensor<B, 2, Int>) -> Tensor<B, 3> {
    let device = codebook.device();
    let [batch, frames] = codes.dims();
    let dim = codebook.dims()[1];
    let indices = codes.into_data().to_vec::<i32>().unwrap();
    let codebook_values = codebook.into_data().to_vec::<f32>().unwrap();
    let mut values = Vec::with_capacity(batch * frames * dim);
    for index in indices {
        let index = index.max(0) as usize;
        let start = index * dim;
        values.extend_from_slice(&codebook_values[start..start + dim]);
    }
    Tensor::from_data(TensorData::new(values, [batch, frames, dim]), &device)
}

fn load_decoder_12hz<B: Backend>(
    decoder: &mut Decoder12Hz<B>,
    archive: &TensorArchive,
    device: &B::Device,
) -> Result<()> {
    decoder.first_codebook = Param::from_tensor(load_normalized_codebook(
        archive,
        "decoder.quantizer.rvq_first.vq.layers.0._codebook",
        device,
    )?);
    for (index, codebook) in decoder.rest_codebooks.iter_mut().enumerate() {
        *codebook = Param::from_tensor(load_normalized_codebook(
            archive,
            &format!("decoder.quantizer.rvq_rest.vq.layers.{index}._codebook"),
            device,
        )?);
    }
    load_conv1d(
        archive,
        "decoder.quantizer.rvq_first.output_proj",
        &mut decoder.first_output_proj,
        device,
    )?;
    load_conv1d(
        archive,
        "decoder.quantizer.rvq_rest.output_proj",
        &mut decoder.rest_output_proj,
        device,
    )?;
    load_causal_conv1d(archive, "decoder.pre_conv", &mut decoder.pre_conv, device)?;
    load_linear(
        archive,
        "decoder.pre_transformer.input_proj",
        &mut decoder.input_proj,
        device,
    )?;
    for (index, layer) in decoder.pre_transformer_layers.iter_mut().enumerate() {
        load_codec_transformer_layer(
            archive,
            &format!("decoder.pre_transformer.layers.{index}"),
            layer,
            device,
        )?;
    }
    load_rms_norm(
        archive,
        "decoder.pre_transformer.norm",
        &mut decoder.pre_transformer_norm,
        device,
    )?;
    load_linear(
        archive,
        "decoder.pre_transformer.output_proj",
        &mut decoder.output_proj,
        device,
    )?;
    for (index, stage) in decoder.upsample_stages.iter_mut().enumerate() {
        load_upsample_stage(archive, &format!("decoder.upsample.{index}"), stage, device)?;
    }
    load_causal_conv1d(
        archive,
        "decoder.decoder.0",
        &mut decoder.decoder_init_conv,
        device,
    )?;
    for (index, block) in decoder.decoder_blocks.iter_mut().enumerate() {
        load_decoder_block(
            archive,
            &format!("decoder.decoder.{}.block", index + 1),
            block,
            device,
        )?;
    }
    load_snake_beta(
        archive,
        "decoder.decoder.5",
        &mut decoder.final_snake,
        device,
    )?;
    load_causal_conv1d(
        archive,
        "decoder.decoder.6",
        &mut decoder.final_conv,
        device,
    )?;
    Ok(())
}

fn load_codec_transformer_layer<B: Backend>(
    archive: &TensorArchive,
    prefix: &str,
    layer: &mut CodecTransformerLayer<B>,
    device: &B::Device,
) -> Result<()> {
    load_rms_norm(
        archive,
        &format!("{prefix}.input_layernorm"),
        &mut layer.input_layernorm,
        device,
    )?;
    load_rms_norm(
        archive,
        &format!("{prefix}.post_attention_layernorm"),
        &mut layer.post_attention_layernorm,
        device,
    )?;
    let attn_prefix = format!("{prefix}.self_attn");
    load_linear(
        archive,
        &format!("{attn_prefix}.q_proj"),
        &mut layer.self_attn.q_proj,
        device,
    )?;
    load_linear(
        archive,
        &format!("{attn_prefix}.k_proj"),
        &mut layer.self_attn.k_proj,
        device,
    )?;
    load_linear(
        archive,
        &format!("{attn_prefix}.v_proj"),
        &mut layer.self_attn.v_proj,
        device,
    )?;
    load_linear(
        archive,
        &format!("{attn_prefix}.o_proj"),
        &mut layer.self_attn.o_proj,
        device,
    )?;
    layer.self_attn_layer_scale = Param::from_tensor(
        archive.load_f32_tensor(&format!("{prefix}.self_attn_layer_scale.scale"), device)?,
    );
    let mlp_prefix = format!("{prefix}.mlp");
    load_linear(
        archive,
        &format!("{mlp_prefix}.gate_proj"),
        &mut layer.gate_proj,
        device,
    )?;
    load_linear(
        archive,
        &format!("{mlp_prefix}.up_proj"),
        &mut layer.up_proj,
        device,
    )?;
    load_linear(
        archive,
        &format!("{mlp_prefix}.down_proj"),
        &mut layer.down_proj,
        device,
    )?;
    layer.mlp_layer_scale = Param::from_tensor(
        archive.load_f32_tensor(&format!("{prefix}.mlp_layer_scale.scale"), device)?,
    );
    Ok(())
}

fn load_upsample_stage<B: Backend>(
    archive: &TensorArchive,
    prefix: &str,
    stage: &mut UpsampleStage<B>,
    device: &B::Device,
) -> Result<()> {
    load_conv_transpose1d(
        archive,
        &format!("{prefix}.0"),
        &mut stage.trans_conv,
        device,
    )?;
    load_convnext_block(archive, &format!("{prefix}.1"), &mut stage.convnext, device)
}

fn load_convnext_block<B: Backend>(
    archive: &TensorArchive,
    prefix: &str,
    block: &mut ConvNeXtBlock<B>,
    device: &B::Device,
) -> Result<()> {
    load_causal_conv1d(
        archive,
        &format!("{prefix}.dwconv"),
        &mut block.dwconv,
        device,
    )?;
    load_layer_norm(archive, &format!("{prefix}.norm"), &mut block.norm, device)?;
    load_linear(
        archive,
        &format!("{prefix}.pwconv1"),
        &mut block.pwconv1,
        device,
    )?;
    load_linear(
        archive,
        &format!("{prefix}.pwconv2"),
        &mut block.pwconv2,
        device,
    )?;
    block.gamma = Param::from_tensor(archive.load_f32_tensor(&format!("{prefix}.gamma"), device)?);
    Ok(())
}

fn load_decoder_block<B: Backend>(
    archive: &TensorArchive,
    prefix: &str,
    block: &mut DecoderBlock<B>,
    device: &B::Device,
) -> Result<()> {
    load_snake_beta(archive, &format!("{prefix}.0"), &mut block.snake, device)?;
    load_conv_transpose1d(archive, &format!("{prefix}.1"), &mut block.upsample, device)?;
    load_residual_unit(archive, &format!("{prefix}.2"), &mut block.res1, device)?;
    load_residual_unit(archive, &format!("{prefix}.3"), &mut block.res2, device)?;
    load_residual_unit(archive, &format!("{prefix}.4"), &mut block.res3, device)
}

fn load_residual_unit<B: Backend>(
    archive: &TensorArchive,
    prefix: &str,
    unit: &mut ResidualUnit<B>,
    device: &B::Device,
) -> Result<()> {
    load_snake_beta(archive, &format!("{prefix}.act1"), &mut unit.act1, device)?;
    load_causal_conv1d(archive, &format!("{prefix}.conv1"), &mut unit.conv1, device)?;
    load_snake_beta(archive, &format!("{prefix}.act2"), &mut unit.act2, device)?;
    load_causal_conv1d(archive, &format!("{prefix}.conv2"), &mut unit.conv2, device)
}

fn load_snake_beta<B: Backend>(
    archive: &TensorArchive,
    prefix: &str,
    snake: &mut SnakeBeta<B>,
    device: &B::Device,
) -> Result<()> {
    snake.alpha = Param::from_tensor(archive.load_f32_tensor(&format!("{prefix}.alpha"), device)?);
    snake.beta = Param::from_tensor(archive.load_f32_tensor(&format!("{prefix}.beta"), device)?);
    Ok(())
}

fn load_rms_norm<B: Backend>(
    archive: &TensorArchive,
    prefix: &str,
    norm: &mut RmsNorm<B>,
    device: &B::Device,
) -> Result<()> {
    norm.inner.gamma =
        Param::from_tensor(archive.load_f32_tensor(&format!("{prefix}.weight"), device)?);
    Ok(())
}

fn load_layer_norm<B: Backend>(
    archive: &TensorArchive,
    prefix: &str,
    norm: &mut LayerNorm<B>,
    device: &B::Device,
) -> Result<()> {
    norm.gamma = Param::from_tensor(archive.load_f32_tensor(&format!("{prefix}.weight"), device)?);
    if norm.beta.is_some() {
        norm.beta = Some(Param::from_tensor(
            archive.load_f32_tensor(&format!("{prefix}.bias"), device)?,
        ));
    }
    Ok(())
}

fn load_linear<B: Backend>(
    archive: &TensorArchive,
    prefix: &str,
    linear: &mut Linear<B>,
    device: &B::Device,
) -> Result<()> {
    let weight: Tensor<B, 2> = archive.load_f32_tensor(&format!("{prefix}.weight"), device)?;
    linear.weight = Param::from_tensor(weight.transpose());
    if linear.bias.is_some() && archive.has_tensor(&format!("{prefix}.bias"))? {
        linear.bias = Some(Param::from_tensor(
            archive.load_f32_tensor(&format!("{prefix}.bias"), device)?,
        ));
    }
    Ok(())
}

fn load_causal_conv1d<B: Backend>(
    archive: &TensorArchive,
    prefix: &str,
    conv: &mut CausalConv1d<B>,
    device: &B::Device,
) -> Result<()> {
    load_conv1d(archive, &format!("{prefix}.conv"), &mut conv.conv, device)?;
    let kernel = conv.conv.kernel_size;
    let dilation = conv.conv.dilation;
    conv.causal_padding = dilation * kernel.saturating_sub(1);
    Ok(())
}

fn load_conv1d<B: Backend>(
    archive: &TensorArchive,
    prefix: &str,
    conv: &mut Conv1d<B>,
    device: &B::Device,
) -> Result<()> {
    conv.weight = Param::from_tensor(archive.load_f32_tensor(&format!("{prefix}.weight"), device)?);
    if conv.bias.is_some() && archive.has_tensor(&format!("{prefix}.bias"))? {
        conv.bias = Some(Param::from_tensor(
            archive.load_f32_tensor(&format!("{prefix}.bias"), device)?,
        ));
    }
    Ok(())
}

fn load_conv_transpose1d<B: Backend>(
    archive: &TensorArchive,
    prefix: &str,
    conv: &mut CausalTransConv1d<B>,
    device: &B::Device,
) -> Result<()> {
    let weight: Tensor<B, 3> = archive.load_f32_tensor(&format!("{prefix}.conv.weight"), device)?;
    let [in_channels, out_channels_per_group, kernel_size] = weight.dims();
    conv.conv.weight = Param::from_tensor(weight);
    conv.conv.kernel_size = kernel_size;
    conv.conv.channels = [in_channels, out_channels_per_group * conv.conv.groups];
    if conv.conv.bias.is_some() && archive.has_tensor(&format!("{prefix}.conv.bias"))? {
        conv.conv.bias = Some(Param::from_tensor(
            archive.load_f32_tensor(&format!("{prefix}.conv.bias"), device)?,
        ));
    }
    let stride = conv.conv.stride;
    conv.right_trim = kernel_size.saturating_sub(stride);
    Ok(())
}

fn load_normalized_codebook<B: Backend>(
    archive: &TensorArchive,
    prefix: &str,
    device: &B::Device,
) -> Result<Tensor<B, 2>> {
    let embedding_sum: Tensor<B, 2> =
        archive.load_f32_tensor(&format!("{prefix}.embedding_sum"), device)?;
    let cluster_usage: Tensor<B, 1> =
        archive.load_f32_tensor(&format!("{prefix}.cluster_usage"), device)?;
    Ok(embedding_sum / cluster_usage.clamp_min(1e-7).unsqueeze_dim::<2>(1))
}

struct TensorArchive {
    bytes: Vec<u8>,
}

impl TensorArchive {
    fn open(path: &Path) -> Result<Self> {
        let bytes = std::fs::read(path)
            .map_err(|err| Error::InvalidInput(format!("failed to read {path:?}: {err}")))?;
        Ok(Self { bytes })
    }

    fn has_tensor(&self, name: &str) -> Result<bool> {
        let tensors = SafeTensors::deserialize(&self.bytes)
            .map_err(|err| Error::InvalidInput(format!("failed to parse safetensors: {err}")))?;
        Ok(tensors.names().iter().any(|tensor| *tensor == name))
    }

    fn tensor(&self, name: &str) -> Result<TensorView<'_>> {
        let tensors = SafeTensors::deserialize(&self.bytes)
            .map_err(|err| Error::InvalidInput(format!("failed to parse safetensors: {err}")))?;
        tensors
            .tensor(name)
            .map_err(|err| Error::InvalidInput(format!("failed to load {name}: {err}")))
    }

    fn load_f32_tensor<B: Backend, const D: usize>(
        &self,
        name: &str,
        device: &B::Device,
    ) -> Result<Tensor<B, D>> {
        let tensor = self.tensor(name)?;
        let shape = tensor.shape().to_vec();
        if shape.len() != D {
            return Err(Error::InvalidInput(format!(
                "{name} rank {}, expected {D}",
                shape.len()
            )));
        }
        let data = tensor_to_f32(name, tensor.dtype(), tensor.data(), shape.iter().product())?;
        Ok(Tensor::from_data(TensorData::new(data, shape), device))
    }
}

fn tensor_to_f32(name: &str, dtype: Dtype, data: &[u8], len: usize) -> Result<Vec<f32>> {
    match dtype {
        Dtype::F32 => Ok(data
            .chunks_exact(4)
            .take(len)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect()),
        Dtype::BF16 => Ok(data
            .chunks_exact(2)
            .take(len)
            .map(|chunk| {
                let bits = u16::from_le_bytes(chunk.try_into().unwrap());
                f32::from_bits((bits as u32) << 16)
            })
            .collect()),
        _ => Err(Error::InvalidInput(format!(
            "{name} has dtype {dtype:?}, expected F32 or BF16"
        ))),
    }
}

#[derive(Debug, Deserialize)]
struct SpeechTokenizerConfig {
    decoder_config: SpeechDecoderConfig,
}

#[derive(Debug, Deserialize)]
struct SpeechDecoderConfig {
    latent_dim: usize,
    codebook_size: usize,
    decoder_dim: usize,
    hidden_size: usize,
    intermediate_size: usize,
    head_dim: usize,
    num_attention_heads: usize,
    num_hidden_layers: usize,
    num_quantizers: usize,
    rms_norm_eps: f64,
    rope_theta: f64,
    upsample_rates: Vec<usize>,
    upsampling_ratios: Vec<usize>,
    vector_quantization_hidden_dimension: usize,
}
