use std::path::Path;
use std::str::FromStr;

use safetensors::tensor::Dtype;
use serde::Deserialize;
use tokenizers::models::bpe::BPE;
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{AddedToken, Tokenizer};

use crate::config::Qwen3TtsConfig;
use crate::error::{Error, Result};
use crate::weights::{TensorArchive, read_value, tensor_to_f32};

const NEWLINE: u32 = 198;

#[derive(Debug, Clone)]
pub struct CustomVoiceInputs {
    pub input_ids: Vec<u32>,
    pub content_ids: Vec<u32>,
    pub prefill: Vec<f32>,
    pub prefill_steps: usize,
    pub trailing_text: Vec<f32>,
    pub trailing_steps: usize,
    pub tts_pad_embed: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Language {
    Chinese,
    English,
    Japanese,
    Korean,
    German,
    French,
    Russian,
    Portuguese,
    Spanish,
    Italian,
}

impl Language {
    pub fn token_id(self) -> u32 {
        match self {
            Self::Chinese => 2055,
            Self::English => 2050,
            Self::Japanese => 2058,
            Self::Korean => 2064,
            Self::German => 2053,
            Self::French => 2061,
            Self::Russian => 2069,
            Self::Portuguese => 2071,
            Self::Spanish => 2054,
            Self::Italian => 2070,
        }
    }

    fn config_key(self) -> &'static str {
        match self {
            Self::Chinese => "chinese",
            Self::English => "english",
            Self::Japanese => "japanese",
            Self::Korean => "korean",
            Self::German => "german",
            Self::French => "french",
            Self::Russian => "russian",
            Self::Portuguese => "portuguese",
            Self::Spanish => "spanish",
            Self::Italian => "italian",
        }
    }
}

impl FromStr for Language {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        match value.to_lowercase().as_str() {
            "chinese" | "zh" => Ok(Self::Chinese),
            "english" | "en" => Ok(Self::English),
            "japanese" | "ja" => Ok(Self::Japanese),
            "korean" | "ko" => Ok(Self::Korean),
            "german" | "de" => Ok(Self::German),
            "french" | "fr" => Ok(Self::French),
            "russian" | "ru" => Ok(Self::Russian),
            "portuguese" | "pt" => Ok(Self::Portuguese),
            "spanish" | "es" => Ok(Self::Spanish),
            "italian" | "it" => Ok(Self::Italian),
            _ => Err(Error::InvalidInput(format!("unknown language {value}"))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Speaker {
    Serena,
    Vivian,
    UncleFu,
    Ryan,
    Aiden,
    OnoAnna,
    Sohee,
    Eric,
    Dylan,
}

impl Speaker {
    pub fn token_id(self) -> u32 {
        match self {
            Self::Serena => 3066,
            Self::Vivian => 3065,
            Self::UncleFu => 3010,
            Self::Ryan => 3061,
            Self::Aiden => 2861,
            Self::OnoAnna => 2873,
            Self::Sohee => 2864,
            Self::Eric => 2875,
            Self::Dylan => 2878,
        }
    }

    fn config_key(self) -> &'static str {
        match self {
            Self::Serena => "serena",
            Self::Vivian => "vivian",
            Self::UncleFu => "uncle_fu",
            Self::Ryan => "ryan",
            Self::Aiden => "aiden",
            Self::OnoAnna => "ono_anna",
            Self::Sohee => "sohee",
            Self::Eric => "eric",
            Self::Dylan => "dylan",
        }
    }
}

impl FromStr for Speaker {
    type Err = Error;

    fn from_str(value: &str) -> Result<Self> {
        match value.to_lowercase().as_str() {
            "serena" => Ok(Self::Serena),
            "vivian" => Ok(Self::Vivian),
            "uncle_fu" | "unclefu" => Ok(Self::UncleFu),
            "ryan" => Ok(Self::Ryan),
            "aiden" => Ok(Self::Aiden),
            "ono_anna" | "onoanna" => Ok(Self::OnoAnna),
            "sohee" => Ok(Self::Sohee),
            "eric" => Ok(Self::Eric),
            "dylan" => Ok(Self::Dylan),
            _ => Err(Error::InvalidInput(format!("unknown speaker {value}"))),
        }
    }
}

pub struct TextTokenizer {
    tokenizer: Tokenizer,
}

impl TextTokenizer {
    pub fn from_pretrained(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if path.is_dir() {
            let tokenizer_path = path.join("tokenizer.json");
            if tokenizer_path.exists() {
                return Self::from_file(tokenizer_path);
            }

            let vocab_path = path.join("vocab.json");
            let merges_path = path.join("merges.txt");
            if vocab_path.exists() && merges_path.exists() {
                return Self::from_bpe_files(path, &vocab_path, &merges_path);
            }

            return Err(Error::InvalidInput(format!(
                "tokenizer files not found under {}; expected tokenizer.json or vocab.json + merges.txt",
                path.display()
            )));
        }

        Self::from_file(path)
    }

    fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let tokenizer = Tokenizer::from_file(path).map_err(|err| {
            Error::InvalidInput(format!(
                "failed to load tokenizer from {}: {err}",
                path.display()
            ))
        })?;
        Ok(Self { tokenizer })
    }

    fn from_bpe_files(root: &Path, vocab_path: &Path, merges_path: &Path) -> Result<Self> {
        let bpe = BPE::from_file(
            vocab_path.to_string_lossy().as_ref(),
            merges_path.to_string_lossy().as_ref(),
        )
        .build()
        .map_err(|err| Error::InvalidInput(format!("failed to load BPE tokenizer: {err}")))?;
        let mut tokenizer = Tokenizer::new(bpe);
        tokenizer
            .with_pre_tokenizer(Some(ByteLevel::default().add_prefix_space(false)))
            .with_decoder(Some(ByteLevel::default()))
            .with_post_processor(Some(ByteLevel::default().trim_offsets(false)));

        let config_path = root.join("tokenizer_config.json");
        if config_path.exists() {
            let config_bytes = std::fs::read(&config_path).map_err(|err| {
                Error::InvalidInput(format!("failed to read {}: {err}", config_path.display()))
            })?;
            let config: TokenizerConfig = serde_json::from_slice(&config_bytes).map_err(|err| {
                Error::InvalidInput(format!("failed to parse {}: {err}", config_path.display()))
            })?;
            let mut added = config
                .added_tokens_decoder
                .into_iter()
                .map(|(id, token)| id.parse::<u32>().map(|id| (id, token)))
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|err| Error::InvalidInput(format!("invalid added token id: {err}")))?;
            added.sort_by_key(|(id, _)| *id);

            let added_tokens = added
                .into_iter()
                .map(|(_, token)| {
                    AddedToken::from(token.content, token.special)
                        .single_word(token.single_word)
                        .lstrip(token.lstrip)
                        .rstrip(token.rstrip)
                        .normalized(token.normalized)
                })
                .collect::<Vec<_>>();
            tokenizer.add_tokens(&added_tokens);
        }

        Ok(Self { tokenizer })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|err| Error::InvalidInput(format!("failed to encode text: {err}")))?;
        Ok(encoding.get_ids().to_vec())
    }
}

pub struct CustomVoiceTextPrep {
    config: Qwen3TtsConfig,
    tokenizer: TextTokenizer,
    text_embedding_dtype: Dtype,
    text_embedding_data: Vec<u8>,
    text_vocab: usize,
    text_embed_dim: usize,
    hidden: usize,
    text_proj_intermediate: usize,
    fc1_weight: Vec<f32>,
    fc1_bias: Vec<f32>,
    fc2_weight: Vec<f32>,
    fc2_bias: Vec<f32>,
    codec_embedding: Vec<f32>,
    codec_vocab: usize,
}

impl CustomVoiceTextPrep {
    pub fn load(model_dir: &Path) -> Result<Self> {
        let config = Qwen3TtsConfig::load(model_dir)?;
        let tokenizer = TextTokenizer::from_pretrained(model_dir)?;
        let archive = TensorArchive::open(&model_dir.join("model.safetensors"))?;
        archive.with_tensors(|tensors| {
            let text_embedding_tensor = tensors.tensor("talker.model.text_embedding.weight")
                .map_err(|err| Error::InvalidInput(format!("failed to load talker.model.text_embedding.weight: {err}")))?;
            let text_shape = text_embedding_tensor.shape();
            if text_shape.len() != 2 || text_shape[1] != config.talker.hidden && text_shape[1] != 2048 {
                return Err(Error::InvalidInput(format!(
                    "talker.model.text_embedding.weight shape {text_shape:?}; expected [vocab, text_hidden]"
                )));
            }
            let text_embed_dim = text_shape[1];
            if !matches!(text_embedding_tensor.dtype(), Dtype::F32 | Dtype::BF16) {
                return Err(Error::InvalidInput(format!(
                    "talker.model.text_embedding.weight has dtype {:?}, expected F32 or BF16",
                    text_embedding_tensor.dtype()
                )));
            }

            let codec_embedding_tensor = tensors.tensor("talker.model.codec_embedding.weight")
                .map_err(|err| Error::InvalidInput(format!("failed to load talker.model.codec_embedding.weight: {err}")))?;
            let codec_shape = codec_embedding_tensor.shape();
            if codec_shape.len() != 2 || codec_shape[1] != config.talker.hidden {
                return Err(Error::InvalidInput(format!(
                    "talker.model.codec_embedding.weight shape {codec_shape:?}; expected [vocab, {}]",
                    config.talker.hidden
                )));
            }
            let hidden = codec_shape[1];

            let (fc1_weight, fc1_bias, fc1_in, fc1_out) = load_linear(tensors, "talker.text_projection.linear_fc1")?;
            let (fc2_weight, fc2_bias, fc2_in, fc2_out) = load_linear(tensors, "talker.text_projection.linear_fc2")?;
            if fc1_in != text_embed_dim || fc1_out != fc2_in || fc2_out != hidden {
                return Err(Error::InvalidInput(format!(
                    "invalid text projection shapes: fc1=[{fc1_out},{fc1_in}], fc2=[{fc2_out},{fc2_in}]"
                )));
            }

            Ok(Self {
                config,
                tokenizer,
                text_embedding_dtype: text_embedding_tensor.dtype(),
                text_embedding_data: text_embedding_tensor.data().to_vec(),
                text_vocab: text_shape[0],
                text_embed_dim,
                hidden,
                text_proj_intermediate: fc1_out,
                fc1_weight,
                fc1_bias,
                fc2_weight,
                fc2_bias,
                codec_embedding: tensor_to_f32(
                    "talker.model.codec_embedding.weight",
                    codec_embedding_tensor.dtype(),
                    codec_embedding_tensor.data(),
                    codec_shape[0] * codec_shape[1],
                )?,
                codec_vocab: codec_shape[0],
            })
        })
    }

    pub fn prepare_custom_voice(
        &self,
        text: &str,
        speaker: Speaker,
        language: Language,
    ) -> Result<CustomVoiceInputs> {
        let assistant_text =
            format!("<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n");
        let input_ids = self.tokenizer.encode(&assistant_text)?;
        if input_ids.len() < 8 {
            return Err(Error::InvalidInput(format!(
                "formatted prompt is too short: {} tokens",
                input_ids.len()
            )));
        }
        let content_ids = input_ids[3..input_ids.len() - 5].to_vec();
        let prefill = self.build_custom_voice_prefill(&content_ids, speaker, language)?;
        let mut trailing = content_ids.get(1..).unwrap_or_default().to_vec();
        trailing.push(self.config.tokens.tts_eos as u32);
        let trailing_text = self.projected_text_embeddings(&trailing)?;
        let tts_pad_embed = self.projected_text_embeddings(&[self.config.tokens.tts_pad as u32])?;
        Ok(CustomVoiceInputs {
            input_ids,
            content_ids,
            prefill_steps: prefill.len() / self.hidden,
            prefill,
            trailing_steps: trailing.len(),
            trailing_text,
            tts_pad_embed,
        })
    }

    fn build_custom_voice_prefill(
        &self,
        text_tokens: &[u32],
        speaker: Speaker,
        language: Language,
    ) -> Result<Vec<f32>> {
        let language_id = self.token_id(
            "language",
            language.config_key(),
            &self.config.tokens.language_ids,
        )?;
        let speaker_id = self.token_id(
            "speaker",
            speaker.config_key(),
            &self.config.tokens.speaker_ids,
        )?;
        let role_prefix = self.projected_text_embeddings(&[
            self.config.tokens.im_start as u32,
            self.config.tokens.assistant as u32,
            NEWLINE,
        ])?;
        let codec = self.codec_embeddings(&[
            self.config.tokens.codec_think as u32,
            self.config.tokens.codec_think_bos as u32,
            language_id as u32,
            self.config.tokens.codec_think_eos as u32,
            speaker_id as u32,
            self.config.tokens.codec_pad as u32,
            self.config.tokens.codec_bos as u32,
        ])?;
        let tts_pad = self.config.tokens.tts_pad as u32;
        let tts_bos = self.config.tokens.tts_bos as u32;
        let tts = self
            .projected_text_embeddings(&[tts_pad, tts_pad, tts_pad, tts_pad, tts_pad, tts_bos])?;

        let mut hidden =
            Vec::with_capacity((9 + usize::from(!text_tokens.is_empty())) * self.hidden);
        hidden.extend(role_prefix);
        for idx in 0..6 {
            hidden.extend(add_rows(
                &tts[idx * self.hidden..(idx + 1) * self.hidden],
                &codec[idx * self.hidden..(idx + 1) * self.hidden],
            ));
        }
        if let Some(&first_text) = text_tokens.first() {
            let text = self.projected_text_embeddings(&[first_text])?;
            hidden.extend(add_rows(&text, &codec[6 * self.hidden..7 * self.hidden]));
        }
        Ok(hidden)
    }

    fn projected_text_embeddings(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let mut output = Vec::with_capacity(tokens.len() * self.hidden);
        let mut fc1 = vec![0.0; self.text_proj_intermediate];
        let mut projected = vec![0.0; self.hidden];
        for &token in tokens {
            let embedding = self.text_embedding_row(token)?;
            linear(
                &embedding,
                &self.fc1_weight,
                &self.fc1_bias,
                &mut fc1,
                self.text_embed_dim,
                self.text_proj_intermediate,
            );
            for value in &mut fc1 {
                *value = silu(*value);
            }
            linear(
                &fc1,
                &self.fc2_weight,
                &self.fc2_bias,
                &mut projected,
                self.text_proj_intermediate,
                self.hidden,
            );
            output.extend_from_slice(&projected);
        }
        Ok(output)
    }

    fn codec_embeddings(&self, tokens: &[u32]) -> Result<Vec<f32>> {
        let mut output = Vec::with_capacity(tokens.len() * self.hidden);
        for &token in tokens {
            let row = token as usize;
            if row >= self.codec_vocab {
                return Err(Error::InvalidInput(format!(
                    "codec token {token} is out of range for vocab {}",
                    self.codec_vocab
                )));
            }
            output.extend_from_slice(
                &self.codec_embedding[row * self.hidden..(row + 1) * self.hidden],
            );
        }
        Ok(output)
    }

    fn text_embedding_row(&self, token: u32) -> Result<Vec<f32>> {
        let row = token as usize;
        if row >= self.text_vocab {
            return Err(Error::InvalidInput(format!(
                "text token {token} is out of range for vocab {}",
                self.text_vocab
            )));
        }
        let offset = row * self.text_embed_dim;
        Ok((0..self.text_embed_dim)
            .map(|idx| {
                read_value(
                    self.text_embedding_dtype,
                    &self.text_embedding_data,
                    offset + idx,
                )
            })
            .collect())
    }

    fn token_id(
        &self,
        kind: &str,
        key: &str,
        tokens: &std::collections::HashMap<String, usize>,
    ) -> Result<usize> {
        tokens
            .get(key)
            .copied()
            .ok_or_else(|| Error::InvalidInput(format!("missing {kind} token id for {key}")))
    }
}

fn load_linear(
    tensors: &safetensors::SafeTensors<'_>,
    prefix: &str,
) -> Result<(Vec<f32>, Vec<f32>, usize, usize)> {
    let weight = tensors
        .tensor(&format!("{prefix}.weight"))
        .map_err(|err| Error::InvalidInput(format!("failed to load {prefix}.weight: {err}")))?;
    let bias = tensors
        .tensor(&format!("{prefix}.bias"))
        .map_err(|err| Error::InvalidInput(format!("failed to load {prefix}.bias: {err}")))?;
    let weight_shape = weight.shape();
    if weight_shape.len() != 2 {
        return Err(Error::InvalidInput(format!(
            "{prefix}.weight shape {weight_shape:?}; expected [out, in]"
        )));
    }
    let out_dim = weight_shape[0];
    let in_dim = weight_shape[1];
    let bias_shape = bias.shape();
    if bias_shape != [out_dim] {
        return Err(Error::InvalidInput(format!(
            "{prefix}.bias shape {bias_shape:?}; expected [{out_dim}]"
        )));
    }
    Ok((
        tensor_to_f32(prefix, weight.dtype(), weight.data(), out_dim * in_dim)?,
        tensor_to_f32(prefix, bias.dtype(), bias.data(), out_dim)?,
        in_dim,
        out_dim,
    ))
}

fn linear(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    output: &mut [f32],
    in_dim: usize,
    out_dim: usize,
) {
    for out_idx in 0..out_dim {
        let row = &weight[out_idx * in_dim..(out_idx + 1) * in_dim];
        let mut sum = bias[out_idx];
        for in_idx in 0..in_dim {
            sum = input[in_idx].mul_add(row[in_idx], sum);
        }
        output[out_idx] = sum;
    }
}

fn silu(value: f32) -> f32 {
    value / (1.0 + (-value).exp())
}

fn add_rows(left: &[f32], right: &[f32]) -> Vec<f32> {
    left.iter().zip(right).map(|(a, b)| a + b).collect()
}

#[derive(Debug, Deserialize)]
struct TokenizerConfig {
    #[serde(default)]
    added_tokens_decoder: std::collections::HashMap<String, AddedTokenConfig>,
}

#[derive(Debug, Deserialize)]
struct AddedTokenConfig {
    content: String,
    #[serde(default)]
    lstrip: bool,
    #[serde(default)]
    normalized: bool,
    #[serde(default)]
    rstrip: bool,
    #[serde(default)]
    single_word: bool,
    #[serde(default)]
    special: bool,
}
