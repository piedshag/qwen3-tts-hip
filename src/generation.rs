use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::audio::codec_hip::HipCodecInitial;
use crate::error::{Error, Result};
use crate::gpu::buffer::DeviceBuffer;
use crate::gpu::profile;
use crate::gpu::runtime::HipRuntime;
use crate::model::code_predictor::HipCodePredictor;
use crate::model::config::Qwen3TtsConfig;
use crate::model::sampling::SamplingConfig;
use crate::model::talker::HipTalker;
pub use crate::model::text::{Language, Speaker};
use crate::model::text::{TtsPreparedInputs, TtsTextPrep};
pub use crate::model::voice_clone::VoiceClonePrompt;

pub const SAMPLE_RATE: u32 = 24_000;
pub const DEFAULT_MAX_CACHE_STEPS: usize = 512;
pub const DEFAULT_PREFILL_HEADROOM: usize = 256;
pub const DEFAULT_STREAM_LEFT_CONTEXT_FRAMES: usize = 25;
pub const DEFAULT_TEXT_LOOKAHEAD_TOKENS: usize = 8;

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
    pub do_sample: bool,
    pub top_k: usize,
    pub top_p: f32,
    pub temperature: f32,
    pub repetition_penalty: f32,
    pub subtalker_dosample: bool,
    pub subtalker_top_k: usize,
    pub subtalker_top_p: f32,
    pub subtalker_temperature: f32,
    pub seed: u64,
    pub text_lookahead_tokens: usize,
}

impl Default for GenerateOptions {
    fn default() -> Self {
        Self {
            speaker: Speaker::Ryan,
            language: Language::English,
            max_frames: 240,
            decode_audio: true,
            do_sample: true,
            top_k: 50,
            top_p: 1.0,
            temperature: 0.9,
            repetition_penalty: 1.05,
            subtalker_dosample: true,
            subtalker_top_k: 50,
            subtalker_top_p: 1.0,
            subtalker_temperature: 0.9,
            seed: 0,
            text_lookahead_tokens: DEFAULT_TEXT_LOOKAHEAD_TOKENS,
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

#[derive(Clone, Debug, Default)]
pub struct GenerationProfile {
    pub prepare_text_seconds: f64,
    pub upload_seconds: f64,
    pub prefill_seconds: f64,
    pub prepare_prefix_seconds: f64,
    pub code_predictor_seconds: f64,
    pub build_step_input_seconds: f64,
    pub talker_decode_seconds: f64,
}

impl GenerationProfile {
    pub fn total_seconds(&self) -> f64 {
        self.prepare_text_seconds
            + self.upload_seconds
            + self.prefill_seconds
            + self.prepare_prefix_seconds
            + self.code_predictor_seconds
            + self.build_step_input_seconds
            + self.talker_decode_seconds
    }
}

#[derive(Clone, Debug)]
pub struct ProfiledGeneratedCodes {
    pub codes: GeneratedCodes,
    pub profile: GenerationProfile,
}

pub struct HipTtsEngine {
    runtime: HipRuntime,
    model_dir: PathBuf,
    prep: TtsTextPrep,
    talker: HipTalker,
    predictor: HipCodePredictor,
    codec: HipCodecInitial,
    max_cache_steps: usize,
    code_groups: usize,
    codec_eos_token: i32,
}

pub struct HipTtsStream<'a> {
    engine: &'a HipTtsEngine,
    text: String,
    prefill_steps: usize,
    max_frames: usize,
    talker_sampling: SamplingConfig,
    subtalker_sampling: SamplingConfig,
    repetition_penalty: f32,
    rng_state: u64,
    trailing: Vec<DeviceBuffer<f32>>,
    cp_prefix: DeviceBuffer<f32>,
    acoustic_sum: DeviceBuffer<f32>,
    semantic: i32,
    semantic_history: Vec<i32>,
    codes: Vec<i32>,
    frame: usize,
    ended_by_eos: bool,
}

pub struct HipTextStream<'a> {
    engine: &'a HipTtsEngine,
    options: GenerateOptions,
    stream_options: TextStreamOptions,
    voice_clone_prompt: Option<VoiceClonePrompt>,
    pending_text: String,
    queued_tokens: VecDeque<u32>,
    text_finished: bool,
    eos_pending: bool,
    eos_consumed: bool,
    state: Option<HipTextStreamState>,
    finished_codes: Option<GeneratedCodes>,
}

struct HipTextStreamState {
    prefill_steps: usize,
    talker_sampling: SamplingConfig,
    subtalker_sampling: SamplingConfig,
    repetition_penalty: f32,
    rng_state: u64,
    tts_pad_embedding: Vec<f32>,
    cp_prefix: DeviceBuffer<f32>,
    acoustic_sum: DeviceBuffer<f32>,
    semantic: i32,
    semantic_history: Vec<i32>,
    codes: Vec<i32>,
    frame: usize,
    ended_by_eos: bool,
}

#[derive(Clone, Debug)]
pub struct TextStreamOptions {
    pub min_start_tokens: usize,
    pub min_resume_tokens: usize,
    pub flush_on_punctuation: bool,
}

impl Default for TextStreamOptions {
    fn default() -> Self {
        Self {
            min_start_tokens: DEFAULT_TEXT_LOOKAHEAD_TOKENS,
            min_resume_tokens: 1,
            flush_on_punctuation: true,
        }
    }
}

#[derive(Clone, Debug)]
pub enum IncrementalAudio {
    Chunk(GeneratedAudioChunk),
    NeedMoreText,
    Finished(GeneratedCodes),
}

#[derive(Clone, Debug)]
pub struct GeneratedCodesChunk {
    pub codes: Vec<i32>,
    pub frames: usize,
    pub total_frames: usize,
    pub ended_by_eos: bool,
}

#[derive(Clone, Debug)]
pub struct GeneratedAudioChunk {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub frames: usize,
    pub total_frames: usize,
    pub ended_by_eos: bool,
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
        let prep = TtsTextPrep::load(&model_dir)?;
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

    pub fn generate_voice_clone(
        &self,
        text: &str,
        prompt: &VoiceClonePrompt,
        options: GenerateOptions,
    ) -> Result<GeneratedSpeech> {
        let codes = self.generate_voice_clone_codes(text, prompt, options.clone())?;
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
        let _range = profile::range("engine.generate_codes");
        Self::validate_generate_options(&options)?;
        let inputs = {
            let _range = profile::range("engine.prepare_text");
            self.prep.prepare_custom_voice_with_lookahead(
                text,
                options.speaker,
                options.language,
                options.text_lookahead_tokens,
            )?
        };
        self.check_cache_capacity(&inputs, options.max_frames)?;
        let codes = self.rollout(&inputs, &options)?;
        Ok(codes)
    }

    pub fn generate_voice_clone_codes(
        &self,
        text: &str,
        prompt: &VoiceClonePrompt,
        options: GenerateOptions,
    ) -> Result<GeneratedCodes> {
        let _range = profile::range("engine.generate_voice_clone_codes");
        Self::validate_generate_options(&options)?;
        let inputs = {
            let _range = profile::range("engine.prepare_voice_clone_text");
            self.prep.prepare_voice_clone_xvector_with_lookahead(
                text,
                prompt,
                options.language,
                options.text_lookahead_tokens,
            )?
        };
        self.check_cache_capacity(&inputs, options.max_frames)?;
        let codes = self.rollout(&inputs, &options)?;
        Ok(codes)
    }

    pub fn generate_codes_profiled(
        &self,
        text: &str,
        options: GenerateOptions,
    ) -> Result<ProfiledGeneratedCodes> {
        let _range = profile::range("engine.generate_codes_profiled");
        Self::validate_generate_options(&options)?;

        let mut timings = GenerationProfile::default();
        let start = Instant::now();
        let inputs = self.prep.prepare_custom_voice_with_lookahead(
            text,
            options.speaker,
            options.language,
            options.text_lookahead_tokens,
        )?;
        timings.prepare_text_seconds = start.elapsed().as_secs_f64();
        self.check_cache_capacity(&inputs, options.max_frames)?;

        let codes = self.rollout_profiled(&inputs, &options, &mut timings)?;
        Ok(ProfiledGeneratedCodes {
            codes,
            profile: timings,
        })
    }

    pub fn start_stream(&self, text: &str, options: GenerateOptions) -> Result<HipTtsStream<'_>> {
        let _range = profile::range("engine.start_stream");
        Self::validate_generate_options(&options)?;
        let inputs = {
            let _range = profile::range("engine.prepare_text");
            self.prep.prepare_custom_voice_with_lookahead(
                text,
                options.speaker,
                options.language,
                options.text_lookahead_tokens,
            )?
        };
        self.start_stream_with_inputs(text, inputs, options)
    }

    pub fn start_voice_clone_stream(
        &self,
        text: &str,
        prompt: &VoiceClonePrompt,
        options: GenerateOptions,
    ) -> Result<HipTtsStream<'_>> {
        let _range = profile::range("engine.start_voice_clone_stream");
        Self::validate_generate_options(&options)?;
        let inputs = {
            let _range = profile::range("engine.prepare_voice_clone_text");
            self.prep.prepare_voice_clone_xvector_with_lookahead(
                text,
                prompt,
                options.language,
                options.text_lookahead_tokens,
            )?
        };
        self.start_stream_with_inputs(text, inputs, options)
    }

    pub fn start_text_stream(
        &self,
        options: GenerateOptions,
        stream_options: TextStreamOptions,
    ) -> Result<HipTextStream<'_>> {
        let _range = profile::range("engine.start_text_stream");
        Self::validate_generate_options(&options)?;
        Self::validate_text_stream_options(&stream_options)?;
        Ok(HipTextStream::new(self, options, stream_options, None))
    }

    pub fn start_voice_clone_text_stream(
        &self,
        prompt: &VoiceClonePrompt,
        options: GenerateOptions,
        stream_options: TextStreamOptions,
    ) -> Result<HipTextStream<'_>> {
        let _range = profile::range("engine.start_voice_clone_text_stream");
        Self::validate_generate_options(&options)?;
        Self::validate_text_stream_options(&stream_options)?;
        Ok(HipTextStream::new(
            self,
            options,
            stream_options,
            Some(prompt.clone()),
        ))
    }

    fn start_stream_with_inputs(
        &self,
        text: &str,
        inputs: TtsPreparedInputs,
        options: GenerateOptions,
    ) -> Result<HipTtsStream<'_>> {
        self.check_cache_capacity(&inputs, options.max_frames)?;
        let talker_sampling = options.talker_sampling();
        let subtalker_sampling = options.subtalker_sampling();
        let mut rng_state = options.seed ^ 0x9e37_79b9_7f4a_7c15;
        let hidden = self.talker.hidden_size();
        let prefill = self.runtime.buffer_from_slice(&inputs.prefill)?;
        let trailing = {
            let _range = profile::range("engine.rollout.upload_trailing");
            self.upload_trailing(
                &inputs.trailing_text,
                &inputs.tts_pad_embed,
                options.max_frames.saturating_sub(1),
                hidden,
            )?
        };
        let semantic = {
            let _range = profile::range("engine.rollout.talker_prefill");
            if talker_sampling.do_sample {
                self.talker.prefill_token_with_sampling(
                    &prefill,
                    inputs.prefill_steps,
                    talker_sampling,
                    &mut rng_state,
                )?
            } else {
                self.talker.prefill_token(&prefill, inputs.prefill_steps)?
            }
        };
        Ok(HipTtsStream {
            engine: self,
            text: text.to_string(),
            prefill_steps: inputs.prefill_steps,
            max_frames: options.max_frames,
            talker_sampling,
            subtalker_sampling,
            repetition_penalty: options.repetition_penalty,
            rng_state,
            trailing,
            cp_prefix: self.runtime.empty_buffer::<f32>(2 * hidden)?,
            acoustic_sum: self.runtime.empty_buffer::<f32>(hidden)?,
            semantic,
            semantic_history: Vec::with_capacity(options.max_frames),
            codes: Vec::with_capacity(options.max_frames * self.code_groups),
            frame: 0,
            ended_by_eos: false,
        })
    }

    fn validate_generate_options(options: &GenerateOptions) -> Result<()> {
        if options.max_frames == 0 {
            return Err(Error::InvalidInput(
                "max_frames must be non-zero".to_string(),
            ));
        }
        if options.repetition_penalty <= 0.0 {
            return Err(Error::InvalidInput(
                "repetition_penalty must be positive".to_string(),
            ));
        }
        if options.text_lookahead_tokens == 0 {
            return Err(Error::InvalidInput(
                "text_lookahead_tokens must be non-zero".to_string(),
            ));
        }
        options.talker_sampling().validate("talker")?;
        options.subtalker_sampling().validate("subtalker")?;
        Ok(())
    }

    fn validate_text_stream_options(options: &TextStreamOptions) -> Result<()> {
        if options.min_start_tokens == 0 {
            return Err(Error::InvalidInput(
                "min_start_tokens must be non-zero".to_string(),
            ));
        }
        if options.min_resume_tokens == 0 {
            return Err(Error::InvalidInput(
                "min_resume_tokens must be non-zero".to_string(),
            ));
        }
        Ok(())
    }

    fn check_cache_capacity(&self, inputs: &TtsPreparedInputs, max_frames: usize) -> Result<()> {
        if inputs.prefill_steps + max_frames > self.max_cache_steps {
            return Err(Error::InvalidInput(format!(
                "requested {} cache steps but engine was loaded for {}; call load_with_max_frames with a larger max_frames",
                inputs.prefill_steps + max_frames,
                self.max_cache_steps
            )));
        }
        Ok(())
    }

    pub fn decode_codes(&self, codes: &[i32]) -> Result<Vec<f32>> {
        let _range = profile::range("engine.decode_codes");
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
        let _range = profile::range("engine.decode.copy_to_host");
        output.waveform.copy_to_host()
    }

    pub fn runtime(&self) -> &HipRuntime {
        &self.runtime
    }

    pub fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    fn rollout(
        &self,
        inputs: &TtsPreparedInputs,
        options: &GenerateOptions,
    ) -> Result<GeneratedCodes> {
        let _range = profile::range("engine.rollout");
        let max_frames = options.max_frames;
        let talker_sampling = options.talker_sampling();
        let subtalker_sampling = options.subtalker_sampling();
        let repetition_penalty = options.repetition_penalty;
        let mut rng_state = options.seed ^ 0x9e37_79b9_7f4a_7c15;
        let hidden = self.talker.hidden_size();
        let prefill = self.runtime.buffer_from_slice(&inputs.prefill)?;
        let trailing = {
            let _range = profile::range("engine.rollout.upload_trailing");
            self.upload_trailing(
                &inputs.trailing_text,
                &inputs.tts_pad_embed,
                max_frames.saturating_sub(1),
                hidden,
            )?
        };
        let cp_prefix = self.runtime.empty_buffer::<f32>(2 * hidden)?;
        let acoustic_sum = self.runtime.empty_buffer::<f32>(hidden)?;
        let mut semantic = {
            let _range = profile::range("engine.rollout.talker_prefill");
            if talker_sampling.do_sample {
                self.talker.prefill_token_with_sampling(
                    &prefill,
                    inputs.prefill_steps,
                    talker_sampling,
                    &mut rng_state,
                )?
            } else {
                self.talker.prefill_token(&prefill, inputs.prefill_steps)?
            }
        };
        let mut codes = Vec::with_capacity(max_frames * self.code_groups);
        let mut semantic_history = Vec::with_capacity(max_frames);
        let mut ended_by_eos = false;
        for frame in 0..max_frames {
            if semantic == self.codec_eos_token {
                ended_by_eos = true;
                break;
            }
            {
                let _range = profile::range("engine.rollout.prepare_code_predictor_prefix");
                self.talker.prepare_code_predictor_prefix(&cp_prefix)?;
            }
            let acoustic = {
                let _range = profile::range("engine.rollout.code_predictor");
                self.predictor.generate_to_buffer_with_options(
                    &cp_prefix,
                    &acoustic_sum,
                    subtalker_sampling,
                    &mut rng_state,
                )?
            };
            codes.push(semantic);
            codes.extend(acoustic);
            semantic_history.push(semantic);
            if frame + 1 < max_frames {
                {
                    let _range = profile::range("engine.rollout.build_step_input");
                    self.talker
                        .build_step_input(&acoustic_sum, &trailing[frame])?;
                }
                semantic = {
                    let _range = profile::range("engine.rollout.talker_decode");
                    if repetition_penalty != 1.0 || talker_sampling.do_sample {
                        let previous = self.runtime.buffer_from_slice(&semantic_history)?;
                        self.talker.decode_prepared_token_with_options(
                            inputs.prefill_steps + frame,
                            &previous,
                            repetition_penalty,
                            talker_sampling,
                            &mut rng_state,
                        )?
                    } else {
                        self.talker
                            .decode_prepared_token(inputs.prefill_steps + frame)?
                    }
                };
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

    fn rollout_profiled(
        &self,
        inputs: &TtsPreparedInputs,
        options: &GenerateOptions,
        timings: &mut GenerationProfile,
    ) -> Result<GeneratedCodes> {
        let max_frames = options.max_frames;
        let talker_sampling = options.talker_sampling();
        let subtalker_sampling = options.subtalker_sampling();
        let repetition_penalty = options.repetition_penalty;
        let mut rng_state = options.seed ^ 0x9e37_79b9_7f4a_7c15;
        let hidden = self.talker.hidden_size();

        let start = Instant::now();
        let prefill = self.runtime.buffer_from_slice(&inputs.prefill)?;
        let trailing = self.upload_trailing(
            &inputs.trailing_text,
            &inputs.tts_pad_embed,
            max_frames.saturating_sub(1),
            hidden,
        )?;
        let cp_prefix = self.runtime.empty_buffer::<f32>(2 * hidden)?;
        let acoustic_sum = self.runtime.empty_buffer::<f32>(hidden)?;
        self.runtime.synchronize()?;
        timings.upload_seconds += start.elapsed().as_secs_f64();

        let start = Instant::now();
        let mut semantic = if talker_sampling.do_sample {
            self.talker.prefill_token_with_sampling(
                &prefill,
                inputs.prefill_steps,
                talker_sampling,
                &mut rng_state,
            )?
        } else {
            self.talker.prefill_token(&prefill, inputs.prefill_steps)?
        };
        self.runtime.synchronize()?;
        timings.prefill_seconds += start.elapsed().as_secs_f64();

        let mut codes = Vec::with_capacity(max_frames * self.code_groups);
        let mut semantic_history = Vec::with_capacity(max_frames);
        let mut ended_by_eos = false;
        for frame in 0..max_frames {
            if semantic == self.codec_eos_token {
                ended_by_eos = true;
                break;
            }

            let start = Instant::now();
            self.talker.prepare_code_predictor_prefix(&cp_prefix)?;
            self.runtime.synchronize()?;
            timings.prepare_prefix_seconds += start.elapsed().as_secs_f64();

            let start = Instant::now();
            let acoustic = self.predictor.generate_to_buffer_with_options(
                &cp_prefix,
                &acoustic_sum,
                subtalker_sampling,
                &mut rng_state,
            )?;
            self.runtime.synchronize()?;
            timings.code_predictor_seconds += start.elapsed().as_secs_f64();

            codes.push(semantic);
            codes.extend(acoustic);
            semantic_history.push(semantic);
            if frame + 1 < max_frames {
                let start = Instant::now();
                self.talker
                    .build_step_input(&acoustic_sum, &trailing[frame])?;
                self.runtime.synchronize()?;
                timings.build_step_input_seconds += start.elapsed().as_secs_f64();

                let start = Instant::now();
                semantic = if repetition_penalty != 1.0 || talker_sampling.do_sample {
                    let previous = self.runtime.buffer_from_slice(&semantic_history)?;
                    self.talker.decode_prepared_token_with_options(
                        inputs.prefill_steps + frame,
                        &previous,
                        repetition_penalty,
                        talker_sampling,
                        &mut rng_state,
                    )?
                } else {
                    self.talker
                        .decode_prepared_token(inputs.prefill_steps + frame)?
                };
                self.runtime.synchronize()?;
                timings.talker_decode_seconds += start.elapsed().as_secs_f64();
            }
        }

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

impl HipTtsStream<'_> {
    pub fn next_codes_chunk(
        &mut self,
        max_chunk_frames: usize,
    ) -> Result<Option<GeneratedCodesChunk>> {
        let _range = profile::range("engine.stream.next_codes_chunk");
        if max_chunk_frames == 0 {
            return Err(Error::InvalidInput(
                "max_chunk_frames must be non-zero".to_string(),
            ));
        }
        if self.ended_by_eos || self.frame >= self.max_frames {
            return Ok(None);
        }

        let start = self.codes.len();
        let mut produced = 0usize;
        while produced < max_chunk_frames && self.frame < self.max_frames {
            if self.semantic == self.engine.codec_eos_token {
                self.ended_by_eos = true;
                break;
            }
            {
                let _range = profile::range("engine.rollout.prepare_code_predictor_prefix");
                self.engine
                    .talker
                    .prepare_code_predictor_prefix(&self.cp_prefix)?;
            }
            let acoustic = {
                let _range = profile::range("engine.rollout.code_predictor");
                self.engine.predictor.generate_to_buffer_with_options(
                    &self.cp_prefix,
                    &self.acoustic_sum,
                    self.subtalker_sampling,
                    &mut self.rng_state,
                )?
            };
            self.codes.push(self.semantic);
            self.codes.extend(acoustic);
            self.semantic_history.push(self.semantic);
            produced += 1;

            let current_frame = self.frame;
            if current_frame + 1 < self.max_frames {
                {
                    let _range = profile::range("engine.rollout.build_step_input");
                    self.engine
                        .talker
                        .build_step_input(&self.acoustic_sum, &self.trailing[current_frame])?;
                }
                self.semantic = {
                    let _range = profile::range("engine.rollout.talker_decode");
                    if self.repetition_penalty != 1.0 || self.talker_sampling.do_sample {
                        let previous = self
                            .engine
                            .runtime
                            .buffer_from_slice(&self.semantic_history)?;
                        self.engine.talker.decode_prepared_token_with_options(
                            self.prefill_steps + current_frame,
                            &previous,
                            self.repetition_penalty,
                            self.talker_sampling,
                            &mut self.rng_state,
                        )?
                    } else {
                        self.engine
                            .talker
                            .decode_prepared_token(self.prefill_steps + current_frame)?
                    }
                };
            }
            self.frame += 1;
        }
        if self.frame < self.max_frames && self.semantic == self.engine.codec_eos_token {
            self.ended_by_eos = true;
        }
        self.engine.runtime.synchronize()?;

        if self.codes.len() == start {
            return Ok(None);
        }
        let codes = self.codes[start..].to_vec();
        Ok(Some(GeneratedCodesChunk {
            frames: codes.len() / self.engine.code_groups,
            codes,
            total_frames: self.frame,
            ended_by_eos: self.ended_by_eos,
        }))
    }

    pub fn next_audio_chunk(
        &mut self,
        max_chunk_frames: usize,
    ) -> Result<Option<GeneratedAudioChunk>> {
        let Some(codes) = self.next_codes_chunk(max_chunk_frames)? else {
            return Ok(None);
        };
        let start_frame = codes.total_frames - codes.frames;
        let context_frames = DEFAULT_STREAM_LEFT_CONTEXT_FRAMES.min(start_frame);
        let decode_start_frame = start_frame - context_frames;
        let decode_start = decode_start_frame * self.engine.code_groups;
        let decode_end = codes.total_frames * self.engine.code_groups;
        let decoded = self
            .engine
            .decode_codes(&self.codes[decode_start..decode_end])?;
        let context_samples = context_frames * self.engine.codec.samples_per_code_frame();
        let samples = decoded[context_samples.min(decoded.len())..].to_vec();
        Ok(Some(GeneratedAudioChunk {
            samples,
            sample_rate: SAMPLE_RATE,
            frames: codes.frames,
            total_frames: codes.total_frames,
            ended_by_eos: codes.ended_by_eos,
        }))
    }

    pub fn finish_codes(self) -> GeneratedCodes {
        GeneratedCodes {
            frames: self.codes.len() / self.engine.code_groups,
            codes: self.codes,
            ended_by_eos: self.ended_by_eos,
        }
    }

    pub fn text(&self) -> &str {
        &self.text
    }
}

impl<'a> HipTextStream<'a> {
    fn new(
        engine: &'a HipTtsEngine,
        options: GenerateOptions,
        stream_options: TextStreamOptions,
        voice_clone_prompt: Option<VoiceClonePrompt>,
    ) -> HipTextStream<'a> {
        Self {
            engine,
            options,
            stream_options,
            voice_clone_prompt,
            pending_text: String::new(),
            queued_tokens: VecDeque::new(),
            text_finished: false,
            eos_pending: false,
            eos_consumed: false,
            state: None,
            finished_codes: None,
        }
    }

    pub fn push_text(&mut self, text: &str) -> Result<()> {
        if self.text_finished {
            return Err(Error::InvalidInput(
                "cannot push text after finish_text".to_string(),
            ));
        }
        self.pending_text.push_str(text);
        self.flush_stable_text()
    }

    pub fn finish_text(&mut self) -> Result<()> {
        if self.text_finished {
            return Ok(());
        }
        if !self.pending_text.is_empty() {
            let text = std::mem::take(&mut self.pending_text);
            self.enqueue_text_tokens(&text)?;
        }
        self.text_finished = true;
        self.eos_pending = true;
        Ok(())
    }

    pub fn next_audio_chunk(&mut self, max_chunk_frames: usize) -> Result<IncrementalAudio> {
        if max_chunk_frames == 0 {
            return Err(Error::InvalidInput(
                "max_chunk_frames must be non-zero".to_string(),
            ));
        }
        if let Some(codes) = self.finished_codes.as_ref() {
            return Ok(IncrementalAudio::Finished(codes.clone()));
        }
        if self.state.is_none() && !self.try_start_generation()? {
            if self.text_finished && self.queued_tokens.is_empty() && self.eos_pending {
                let codes = GeneratedCodes {
                    codes: Vec::new(),
                    frames: 0,
                    ended_by_eos: false,
                };
                self.finished_codes = Some(codes.clone());
                return Ok(IncrementalAudio::Finished(codes));
            }
            return Ok(IncrementalAudio::NeedMoreText);
        }

        let Some(codes) = self.next_codes_chunk(max_chunk_frames)? else {
            if self.is_generation_finished() {
                let codes = self.finish_codes();
                return Ok(IncrementalAudio::Finished(codes));
            }
            return Ok(IncrementalAudio::NeedMoreText);
        };
        let Some(state) = self.state.as_ref() else {
            let codes = self.finish_codes();
            return Ok(IncrementalAudio::Finished(codes));
        };
        let start_frame = codes.total_frames - codes.frames;
        let context_frames = DEFAULT_STREAM_LEFT_CONTEXT_FRAMES.min(start_frame);
        let decode_start_frame = start_frame - context_frames;
        let decode_start = decode_start_frame * self.engine.code_groups;
        let decode_end = codes.total_frames * self.engine.code_groups;
        let decoded = self
            .engine
            .decode_codes(&state.codes[decode_start..decode_end])?;
        let context_samples = context_frames * self.engine.codec.samples_per_code_frame();
        let samples = decoded[context_samples.min(decoded.len())..].to_vec();
        Ok(IncrementalAudio::Chunk(GeneratedAudioChunk {
            samples,
            sample_rate: SAMPLE_RATE,
            frames: codes.frames,
            total_frames: codes.total_frames,
            ended_by_eos: codes.ended_by_eos,
        }))
    }

    fn next_codes_chunk(&mut self, max_chunk_frames: usize) -> Result<Option<GeneratedCodesChunk>> {
        let Some(mut state) = self.state.take() else {
            return Ok(None);
        };
        if state.ended_by_eos || state.frame >= self.options.max_frames {
            self.state = Some(state);
            return Ok(None);
        }

        let start = state.codes.len();
        let mut produced = 0usize;
        while produced < max_chunk_frames && state.frame < self.options.max_frames {
            if state.semantic == self.engine.codec_eos_token {
                state.ended_by_eos = true;
                break;
            }

            let current_frame = state.frame;
            let trailing = if current_frame + 1 < self.options.max_frames {
                match self.next_trailing_buffer(&state.tts_pad_embedding)? {
                    Some(trailing) => Some(trailing),
                    None => break,
                }
            } else {
                None
            };

            {
                let _range = profile::range("engine.text_stream.prepare_code_predictor_prefix");
                self.engine
                    .talker
                    .prepare_code_predictor_prefix(&state.cp_prefix)?;
            }
            let acoustic = {
                let _range = profile::range("engine.text_stream.code_predictor");
                self.engine.predictor.generate_to_buffer_with_options(
                    &state.cp_prefix,
                    &state.acoustic_sum,
                    state.subtalker_sampling,
                    &mut state.rng_state,
                )?
            };
            state.codes.push(state.semantic);
            state.codes.extend(acoustic);
            state.semantic_history.push(state.semantic);
            produced += 1;

            if let Some(trailing) = trailing.as_ref() {
                {
                    let _range = profile::range("engine.text_stream.build_step_input");
                    self.engine
                        .talker
                        .build_step_input(&state.acoustic_sum, trailing)?;
                }
                state.semantic = {
                    let _range = profile::range("engine.text_stream.talker_decode");
                    if state.repetition_penalty != 1.0 || state.talker_sampling.do_sample {
                        let previous = self
                            .engine
                            .runtime
                            .buffer_from_slice(&state.semantic_history)?;
                        self.engine.talker.decode_prepared_token_with_options(
                            state.prefill_steps + current_frame,
                            &previous,
                            state.repetition_penalty,
                            state.talker_sampling,
                            &mut state.rng_state,
                        )?
                    } else {
                        self.engine
                            .talker
                            .decode_prepared_token(state.prefill_steps + current_frame)?
                    }
                };
            }
            state.frame += 1;
        }
        if state.frame < self.options.max_frames && state.semantic == self.engine.codec_eos_token {
            state.ended_by_eos = true;
        }
        self.engine.runtime.synchronize()?;

        if state.codes.len() == start {
            self.state = Some(state);
            return Ok(None);
        }
        let codes = state.codes[start..].to_vec();
        let chunk = GeneratedCodesChunk {
            frames: codes.len() / self.engine.code_groups,
            codes,
            total_frames: state.frame,
            ended_by_eos: state.ended_by_eos,
        };
        self.state = Some(state);
        Ok(Some(chunk))
    }

    fn try_start_generation(&mut self) -> Result<bool> {
        if self.state.is_some() {
            return Ok(true);
        }
        if self.queued_tokens.len() < self.stream_options.min_start_tokens && !self.text_finished {
            return Ok(false);
        }
        if self.queued_tokens.is_empty() {
            return Ok(false);
        }

        let lookahead = self
            .options
            .text_lookahead_tokens
            .min(self.queued_tokens.len())
            .max(1);
        let mut prefill_tokens = Vec::with_capacity(lookahead);
        for _ in 0..lookahead {
            if let Some(token) = self.queued_tokens.pop_front() {
                prefill_tokens.push(token);
            }
        }
        let prefill = if let Some(prompt) = self.voice_clone_prompt.as_ref() {
            self.engine
                .prep
                .voice_clone_xvector_prefill_from_content_tokens(
                    &prefill_tokens,
                    prompt,
                    self.options.language,
                )?
        } else {
            self.engine.prep.custom_voice_prefill_from_content_tokens(
                &prefill_tokens,
                self.options.speaker,
                self.options.language,
            )?
        };
        let prefill_steps = prefill.len() / self.engine.talker.hidden_size();
        if prefill_steps + self.options.max_frames > self.engine.max_cache_steps {
            return Err(Error::InvalidInput(format!(
                "requested {} cache steps but engine was loaded for {}; call load_with_max_frames with a larger max_frames",
                prefill_steps + self.options.max_frames,
                self.engine.max_cache_steps
            )));
        }

        let talker_sampling = self.options.talker_sampling();
        let subtalker_sampling = self.options.subtalker_sampling();
        let mut rng_state = self.options.seed ^ 0x9e37_79b9_7f4a_7c15;
        let hidden = self.engine.talker.hidden_size();
        let prefill = self.engine.runtime.buffer_from_slice(&prefill)?;
        let tts_pad_embedding = self.engine.prep.tts_pad_embedding()?;
        let semantic = if talker_sampling.do_sample {
            self.engine.talker.prefill_token_with_sampling(
                &prefill,
                prefill_steps,
                talker_sampling,
                &mut rng_state,
            )?
        } else {
            self.engine.talker.prefill_token(&prefill, prefill_steps)?
        };
        self.state = Some(HipTextStreamState {
            prefill_steps,
            talker_sampling,
            subtalker_sampling,
            repetition_penalty: self.options.repetition_penalty,
            rng_state,
            tts_pad_embedding,
            cp_prefix: self.engine.runtime.empty_buffer::<f32>(2 * hidden)?,
            acoustic_sum: self.engine.runtime.empty_buffer::<f32>(hidden)?,
            semantic,
            semantic_history: Vec::with_capacity(self.options.max_frames),
            codes: Vec::with_capacity(self.options.max_frames * self.engine.code_groups),
            frame: 0,
            ended_by_eos: false,
        });
        Ok(true)
    }

    fn next_trailing_buffer(
        &mut self,
        tts_pad_embedding: &[f32],
    ) -> Result<Option<DeviceBuffer<f32>>> {
        if !self.text_finished && self.queued_tokens.len() < self.stream_options.min_resume_tokens {
            return Ok(None);
        }
        if let Some(token) = self.queued_tokens.pop_front() {
            let embedding = self.engine.prep.projected_text_embedding_for_token(token)?;
            return self.engine.runtime.buffer_from_slice(&embedding).map(Some);
        }
        if self.eos_pending {
            self.eos_pending = false;
            self.eos_consumed = true;
            let embedding = self
                .engine
                .prep
                .projected_text_embedding_for_token(self.engine.prep.tts_eos_token())?;
            return self.engine.runtime.buffer_from_slice(&embedding).map(Some);
        }
        if self.text_finished || self.eos_consumed {
            return self
                .engine
                .runtime
                .buffer_from_slice(tts_pad_embedding)
                .map(Some);
        }
        Ok(None)
    }

    fn flush_stable_text(&mut self) -> Result<()> {
        let Some(split) =
            stable_prefix_len(&self.pending_text, self.stream_options.flush_on_punctuation)
        else {
            return Ok(());
        };
        if split == 0 {
            return Ok(());
        }
        let suffix = self.pending_text.split_off(split);
        let prefix = std::mem::replace(&mut self.pending_text, suffix);
        self.enqueue_text_tokens(&prefix)
    }

    fn enqueue_text_tokens(&mut self, text: &str) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }
        let tokens = self.engine.prep.content_ids_for_text(text)?;
        self.queued_tokens.extend(tokens);
        Ok(())
    }

    fn finish_codes(&mut self) -> GeneratedCodes {
        let codes = if let Some(state) = self.state.as_ref() {
            GeneratedCodes {
                frames: state.codes.len() / self.engine.code_groups,
                codes: state.codes.clone(),
                ended_by_eos: state.ended_by_eos,
            }
        } else {
            GeneratedCodes {
                frames: 0,
                codes: Vec::new(),
                ended_by_eos: false,
            }
        };
        self.finished_codes = Some(codes.clone());
        codes
    }

    fn is_generation_finished(&self) -> bool {
        self.state
            .as_ref()
            .map(|state| state.ended_by_eos || state.frame >= self.options.max_frames)
            .unwrap_or(false)
    }
}

fn stable_prefix_len(text: &str, flush_on_punctuation: bool) -> Option<usize> {
    let mut split = None;
    for (index, ch) in text.char_indices() {
        if ch.is_whitespace() || (flush_on_punctuation && is_sentence_boundary(ch)) {
            split = Some(index + ch.len_utf8());
        }
    }
    split
}

fn is_sentence_boundary(ch: char) -> bool {
    matches!(ch, '.' | ',' | ';' | ':' | '!' | '?' | '\n')
}

impl GenerateOptions {
    fn talker_sampling(&self) -> SamplingConfig {
        SamplingConfig {
            do_sample: self.do_sample,
            top_k: self.top_k,
            top_p: self.top_p,
            temperature: self.temperature,
        }
    }

    fn subtalker_sampling(&self) -> SamplingConfig {
        SamplingConfig {
            do_sample: self.subtalker_dosample,
            top_k: self.subtalker_top_k,
            top_p: self.subtalker_top_p,
            temperature: self.subtalker_temperature,
        }
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
