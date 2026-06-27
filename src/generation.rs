use std::path::{Path, PathBuf};

use crate::audio::codec_hip::HipCodecInitial;
use crate::error::{Error, Result};
use crate::gpu::buffer::DeviceBuffer;
use crate::gpu::runtime::HipRuntime;
use crate::model::code_predictor::HipCodePredictor;
use crate::model::config::Qwen3TtsConfig;
use crate::model::talker::HipTalker;
use crate::model::text::{CustomVoiceInputs, CustomVoiceTextPrep};
pub use crate::model::text::{Language, Speaker};

pub const SAMPLE_RATE: u32 = 24_000;
pub const DEFAULT_MAX_CACHE_STEPS: usize = 512;
pub const DEFAULT_PREFILL_HEADROOM: usize = 256;

#[derive(Clone, Debug)]
pub struct EngineOptions {
    pub device: i32,
    pub max_cache_steps: usize,
}

impl Default for EngineOptions {
    fn default() -> Self {
        Self {
            device: 0,
            max_cache_steps: DEFAULT_MAX_CACHE_STEPS,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GenerateOptions {
    pub speaker: Speaker,
    pub language: Language,
    pub max_frames: usize,
    pub decode_audio: bool,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            speaker: Speaker::Ryan,
            language: Language::English,
            max_frames: 240,
            decode_audio: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct GeneratedSpeech {
    pub text: String,
    pub codes: Vec<i32>,
    pub frames: usize,
    pub ended_by_eos: bool,
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

pub struct HipTtsEngine {
    runtime: HipRuntime,
    model_dir: PathBuf,
    prep: CustomVoiceTextPrep,
    talker: HipTalker,
    predictor: HipCodePredictor,
    codec: HipCodecInitial,
    max_cache_steps: usize,
    code_groups: usize,
    codec_eos_token: i32,
}

impl HipTtsEngine {
    pub fn load(model_dir: impl AsRef<Path>, device: i32) -> Result<Self> {
        Self::load_with_options(
            model_dir,
            EngineOptions {
                device,
                ..EngineOptions::default()
            },
        )
    }

    pub fn load_with_max_frames(
        model_dir: impl AsRef<Path>,
        device: i32,
        max_frames: usize,
    ) -> Result<Self> {
        if max_frames == 0 {
            return Err(Error::InvalidInput(
                "max_frames must be non-zero".to_string(),
            ));
        }
        Self::load_with_options(
            model_dir,
            EngineOptions {
                device,
                max_cache_steps: DEFAULT_PREFILL_HEADROOM + max_frames,
            },
        )
    }

    pub fn load_with_options(model_dir: impl AsRef<Path>, options: EngineOptions) -> Result<Self> {
        if options.max_cache_steps == 0 {
            return Err(Error::InvalidInput(
                "max_cache_steps must be non-zero".to_string(),
            ));
        }
        let model_dir = model_dir.as_ref().to_path_buf();
        let config = Qwen3TtsConfig::load(&model_dir)?;
        let prep = CustomVoiceTextPrep::load(&model_dir)?;
        let runtime = HipRuntime::new(options.device)?;
        let talker = HipTalker::load(&runtime, &model_dir, options.max_cache_steps)?;
        let predictor = HipCodePredictor::load(&runtime, &model_dir)?;
        if predictor.talker_hidden() != talker.hidden_size() {
            return Err(Error::InvalidInput(format!(
                "CodePredictor talker hidden {} does not match talker hidden {}",
                predictor.talker_hidden(),
                talker.hidden_size()
            )));
        }
        let codec = HipCodecInitial::load(&runtime, &model_dir)?;
        Ok(Self {
            runtime,
            model_dir,
            prep,
            talker,
            predictor,
            codec,
            max_cache_steps: options.max_cache_steps,
            code_groups: config.talker.num_code_groups,
            codec_eos_token: config.tokens.codec_eos as i32,
        })
    }

    pub fn generate(&self, text: &str, options: GenerateOptions) -> Result<GeneratedSpeech> {
        let codes = self.generate_codes(text, options.clone())?;
        let samples = if options.decode_audio && !codes.codes.is_empty() {
            self.decode_codes(&codes.codes)?
        } else {
            Vec::new()
        };
        Ok(GeneratedSpeech {
            text: text.to_string(),
            frames: codes.frames,
            ended_by_eos: codes.ended_by_eos,
            codes: codes.codes,
            samples,
            sample_rate: SAMPLE_RATE,
        })
    }

    pub fn generate_codes(&self, text: &str, options: GenerateOptions) -> Result<GeneratedCodes> {
        if options.max_frames == 0 {
            return Err(Error::InvalidInput(
                "max_frames must be non-zero".to_string(),
            ));
        }
        let inputs = self
            .prep
            .prepare_custom_voice(text, options.speaker, options.language)?;
        if inputs.prefill_steps + options.max_frames > self.max_cache_steps {
            return Err(Error::InvalidInput(format!(
                "requested {} cache steps but engine was loaded for {}; call load_with_max_frames with a larger max_frames",
                inputs.prefill_steps + options.max_frames,
                self.max_cache_steps
            )));
        }
        let codes = self.rollout(&inputs, options.max_frames)?;
        Ok(codes)
    }

    pub fn decode_codes(&self, codes: &[i32]) -> Result<Vec<f32>> {
        if codes.len() % self.code_groups != 0 {
            return Err(Error::InvalidInput(format!(
                "code length {} is not divisible by {}",
                codes.len(),
                self.code_groups
            )));
        }
        if codes.is_empty() {
            return Ok(Vec::new());
        }
        let frames = codes.len() / self.code_groups;
        let initial = self.codec.run(&self.runtime, codes, frames)?;
        let pre_transformer =
            self.codec
                .run_pre_transformer(&self.runtime, &initial.pre_conv, initial.frames)?;
        let upsample =
            self.codec
                .run_upsample_stages(&self.runtime, &pre_transformer, initial.frames)?;
        let output = self.codec.run_decoder_stages(
            &self.runtime,
            &upsample.upsample_1_1,
            upsample.frames_1,
        )?;
        self.runtime.synchronize()?;
        output.waveform.copy_to_host()
    }

    pub fn runtime(&self) -> &HipRuntime {
        &self.runtime
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    fn rollout(&self, inputs: &CustomVoiceInputs, max_frames: usize) -> Result<GeneratedCodes> {
        let hidden = self.talker.hidden_size();
        let prefill = self.runtime.buffer_from_slice(&inputs.prefill)?;
        let trailing = self.upload_trailing(
            &inputs.trailing_text,
            &inputs.tts_pad_embed,
            max_frames.saturating_sub(1),
            hidden,
        )?;
        let cp_prefix = self.runtime.empty_buffer::<f32>(2 * hidden)?;
        let acoustic_sum = self.runtime.empty_buffer::<f32>(hidden)?;
        let mut semantic = self.talker.prefill_token(&prefill, inputs.prefill_steps)?;
        let mut codes = Vec::with_capacity(max_frames * self.code_groups);
        let mut ended_by_eos = false;
        for frame in 0..max_frames {
            if semantic == self.codec_eos_token {
                ended_by_eos = true;
                break;
            }
            self.talker.prepare_code_predictor_prefix(&cp_prefix)?;
            let acoustic = self
                .predictor
                .generate_to_buffer(&cp_prefix, &acoustic_sum)?;
            codes.push(semantic);
            codes.extend(acoustic);
            if frame + 1 < max_frames {
                self.talker
                    .build_step_input(&acoustic_sum, &trailing[frame])?;
                semantic = self
                    .talker
                    .decode_prepared_token(inputs.prefill_steps + frame)?;
            }
        }
        self.runtime.synchronize()?;
        let frames = codes.len() / self.code_groups;
        Ok(GeneratedCodes {
            codes,
            frames,
            ended_by_eos,
        })
    }

    fn upload_trailing(
        &self,
        trailing: &[f32],
        tts_pad: &[f32],
        frames: usize,
        hidden: usize,
    ) -> Result<Vec<DeviceBuffer<f32>>> {
        let mut buffers = Vec::with_capacity(frames);
        for frame in 0..frames {
            let offset = frame * hidden;
            if offset + hidden <= trailing.len() {
                buffers.push(
                    self.runtime
                        .buffer_from_slice(&trailing[offset..offset + hidden])?,
                );
            } else {
                buffers.push(self.runtime.buffer_from_slice(tts_pad)?);
            }
        }
        Ok(buffers)
    }
}

#[derive(Clone, Debug)]
pub struct GeneratedCodes {
    pub codes: Vec<i32>,
    pub frames: usize,
    pub ended_by_eos: bool,
}

impl GeneratedSpeech {
    pub fn audio_seconds(&self) -> f64 {
        self.samples.len() as f64 / self.sample_rate as f64
    }

    pub fn write_wav(&self, path: impl AsRef<Path>, gain: f32) -> Result<()> {
        crate::audio::codec::write_wav(path.as_ref(), &self.samples, self.sample_rate, gain)
    }

    pub fn to_wav_bytes(&self, gain: f32) -> Result<Vec<u8>> {
        let data_bytes = self.samples.len() * std::mem::size_of::<i16>();
        if data_bytes > u32::MAX as usize - 36 {
            return Err(Error::InvalidInput("wav is too large".to_string()));
        }
        let mut bytes = Vec::with_capacity(44 + data_bytes);
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36u32 + data_bytes as u32).to_le_bytes());
        bytes.extend_from_slice(b"WAVEfmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&1u16.to_le_bytes());
        bytes.extend_from_slice(&self.sample_rate.to_le_bytes());
        bytes.extend_from_slice(&(self.sample_rate * 2).to_le_bytes());
        bytes.extend_from_slice(&2u16.to_le_bytes());
        bytes.extend_from_slice(&16u16.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&(data_bytes as u32).to_le_bytes());
        for &sample in &self.samples {
            let value = (sample * gain).clamp(-1.0, 1.0);
            bytes.extend_from_slice(&((value * i16::MAX as f32) as i16).to_le_bytes());
        }
        Ok(bytes)
    }
}
