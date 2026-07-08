use std::path::Path;

use serde::Deserialize;

use crate::error::{Error, Result};

#[derive(Debug, Clone)]
pub struct VoiceClonePrompt {
    pub speaker_embedding: Vec<f32>,
    pub x_vector_only_mode: bool,
    pub icl_mode: bool,
    pub ref_text: Option<String>,
    pub ref_codes: Option<Vec<i32>>,
}

#[derive(Debug, Deserialize)]
struct VoiceClonePromptJson {
    speaker_embedding: Vec<f32>,
    #[serde(default)]
    x_vector_only_mode: bool,
    #[serde(default)]
    icl_mode: bool,
    #[serde(default)]
    ref_text: Option<String>,
    #[serde(default)]
    ref_codes: Option<Vec<i32>>,
}

impl VoiceClonePrompt {
    pub fn from_json(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|err| {
            Error::InvalidInput(format!("failed to read {}: {err}", path.display()))
        })?;
        let raw: VoiceClonePromptJson = serde_json::from_slice(&bytes).map_err(|err| {
            Error::InvalidInput(format!("failed to parse {}: {err}", path.display()))
        })?;
        Ok(Self {
            speaker_embedding: raw.speaker_embedding,
            x_vector_only_mode: raw.x_vector_only_mode,
            icl_mode: raw.icl_mode,
            ref_text: raw.ref_text,
            ref_codes: raw.ref_codes,
        })
    }
}
