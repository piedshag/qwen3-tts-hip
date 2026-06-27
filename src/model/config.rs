use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

use crate::error::{Error, Result};

#[derive(Clone, Debug)]
pub struct Qwen3TtsConfig {
    pub talker: TransformerConfig,
    pub code_predictor: TransformerConfig,
    pub tokens: TokenConfig,
}

#[derive(Clone, Debug)]
pub struct TransformerConfig {
    pub hidden: usize,
    pub intermediate: usize,
    pub layers: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub vocab: usize,
    pub num_code_groups: usize,
    pub rms_eps: f32,
    pub rope_theta: f32,
}

#[derive(Clone, Debug)]
pub struct TokenConfig {
    pub codec_bos: usize,
    pub codec_eos: usize,
    pub codec_pad: usize,
    pub codec_think: usize,
    pub codec_think_bos: usize,
    pub codec_think_eos: usize,
    pub language_ids: HashMap<String, usize>,
    pub speaker_ids: HashMap<String, usize>,
    pub tts_pad: usize,
    pub tts_bos: usize,
    pub tts_eos: usize,
    pub im_start: usize,
    pub assistant: usize,
}

impl Qwen3TtsConfig {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let path = model_dir.join("config.json");
        let bytes = std::fs::read(&path).map_err(|err| {
            Error::InvalidInput(format!("failed to read {}: {err}", path.display()))
        })?;
        let raw: RawConfig = serde_json::from_slice(&bytes).map_err(|err| {
            Error::InvalidInput(format!("failed to parse {}: {err}", path.display()))
        })?;
        Ok(Self::from_raw(raw))
    }

    fn from_raw(raw: RawConfig) -> Self {
        let talker = raw.talker_config;
        let code_predictor = talker.code_predictor_config.clone();
        Self {
            tokens: TokenConfig {
                codec_bos: talker.codec_bos_id,
                codec_eos: talker.codec_eos_token_id,
                codec_pad: talker.codec_pad_id,
                codec_think: talker.codec_think_id,
                codec_think_bos: talker.codec_think_bos_id,
                codec_think_eos: talker.codec_think_eos_id,
                language_ids: talker.codec_language_id,
                speaker_ids: talker.spk_id,
                tts_pad: raw.tts_pad_token_id,
                tts_bos: raw.tts_bos_token_id,
                tts_eos: raw.tts_eos_token_id,
                im_start: raw.im_start_token_id,
                assistant: raw.assistant_token_id,
            },
            talker: TransformerConfig {
                hidden: talker.hidden_size,
                intermediate: talker.intermediate_size,
                layers: talker.num_hidden_layers,
                q_heads: talker.num_attention_heads,
                kv_heads: talker.num_key_value_heads,
                head_dim: talker.head_dim,
                vocab: talker.vocab_size,
                num_code_groups: talker.num_code_groups,
                rms_eps: talker.rms_norm_eps,
                rope_theta: talker.rope_theta,
            },
            code_predictor: TransformerConfig {
                hidden: code_predictor.hidden_size,
                intermediate: code_predictor.intermediate_size,
                layers: code_predictor.num_hidden_layers,
                q_heads: code_predictor.num_attention_heads,
                kv_heads: code_predictor.num_key_value_heads,
                head_dim: code_predictor.head_dim,
                vocab: code_predictor.vocab_size,
                num_code_groups: code_predictor.num_code_groups,
                rms_eps: code_predictor.rms_norm_eps,
                rope_theta: code_predictor.rope_theta,
            },
        }
    }
}

#[derive(Debug, Deserialize)]
struct RawConfig {
    assistant_token_id: usize,
    im_start_token_id: usize,
    tts_bos_token_id: usize,
    tts_eos_token_id: usize,
    tts_pad_token_id: usize,
    talker_config: RawTalkerConfig,
}

#[derive(Debug, Deserialize)]
struct RawTalkerConfig {
    code_predictor_config: RawTransformerConfig,
    codec_bos_id: usize,
    codec_eos_token_id: usize,
    codec_language_id: HashMap<String, usize>,
    codec_pad_id: usize,
    codec_think_id: usize,
    codec_think_bos_id: usize,
    codec_think_eos_id: usize,
    spk_id: HashMap<String, usize>,
    head_dim: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_attention_heads: usize,
    num_code_groups: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    vocab_size: usize,
}

#[derive(Clone, Debug, Deserialize)]
struct RawTransformerConfig {
    head_dim: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_attention_heads: usize,
    num_code_groups: usize,
    num_hidden_layers: usize,
    num_key_value_heads: usize,
    rms_norm_eps: f32,
    rope_theta: f32,
    vocab_size: usize,
}
