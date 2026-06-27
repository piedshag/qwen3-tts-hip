use std::path::PathBuf;
use std::time::Instant;

use qwen3_hip_runtime::code_predictor::HipCodePredictor;
use qwen3_hip_runtime::codec::write_wav;
use qwen3_hip_runtime::codec_hip::HipCodecInitial;
use qwen3_hip_runtime::talker::HipTalker;
use qwen3_hip_runtime::text::{CustomVoiceTextPrep, Language, Speaker};
use qwen3_hip_runtime::{DeviceBuffer, Error, HipRuntime};

const HIDDEN: usize = 1024;
const CODE_GROUPS: usize = 16;
const CODEC_EOS_TOKEN: i32 = 2150;

fn main() -> qwen3_hip_runtime::Result<()> {
    let mut args = std::env::args_os().skip(1);
    let model_dir = args.next().map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("/home/flynn/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5")
    });
    let text = args
        .next()
        .map(|value| value.to_string_lossy().into_owned())
        .unwrap_or_else(|| "She said she would be here by noon.".to_string());
    let max_frames = parse_arg(args.next(), "max_frames")?.unwrap_or(39);
    let iterations = parse_arg(args.next(), "iterations")?.unwrap_or(3);
    let warmup = parse_arg(args.next(), "warmup")?.unwrap_or(1);
    let output_wav = args.next().map(PathBuf::from);
    let speaker = Speaker::Ryan;
    let language = Language::English;

    let load_start = Instant::now();
    let prep = CustomVoiceTextPrep::load(&model_dir)?;
    let inputs = prep.prepare_custom_voice(&text, speaker, language)?;
    let runtime = HipRuntime::new(0)?;
    let talker = HipTalker::load(&runtime, &model_dir, inputs.prefill_steps + max_frames)?;
    let predictor = HipCodePredictor::load(&runtime, &model_dir)?;
    let decoder = HipCodecInitial::load(&runtime, &model_dir)?;
    let prefill = runtime.buffer_from_slice(&inputs.prefill)?;
    let trailing = upload_trailing(
        &runtime,
        &inputs.trailing_text,
        &inputs.tts_pad_embed,
        max_frames.saturating_sub(1),
    )?;
    let cp_prefix = runtime.empty_buffer::<f32>(2 * HIDDEN)?;
    let acoustic_sum = runtime.empty_buffer::<f32>(HIDDEN)?;
    runtime.synchronize()?;
    let load_seconds = load_start.elapsed().as_secs_f64();

    let mut last_waveform = Vec::new();
    for _ in 0..warmup {
        let frames = rollout(
            &talker,
            &predictor,
            &prefill,
            &cp_prefix,
            &acoustic_sum,
            &trailing,
            inputs.prefill_steps,
            max_frames,
        )?;
        runtime.synchronize()?;
        last_waveform = decode_waveform(&runtime, &decoder, &frames)?;
    }

    let mut generation_seconds = Vec::with_capacity(iterations);
    let mut decode_seconds = Vec::with_capacity(iterations);
    let start = Instant::now();
    for _ in 0..iterations {
        let gen_start = Instant::now();
        let frames = rollout(
            &talker,
            &predictor,
            &prefill,
            &cp_prefix,
            &acoustic_sum,
            &trailing,
            inputs.prefill_steps,
            max_frames,
        )?;
        runtime.synchronize()?;
        generation_seconds.push(gen_start.elapsed().as_secs_f64());

        let decode_start = Instant::now();
        last_waveform = decode_waveform(&runtime, &decoder, &frames)?;
        decode_seconds.push(decode_start.elapsed().as_secs_f64());
    }
    let total_seconds = start.elapsed().as_secs_f64();

    if let Some(path) = output_wav.as_deref() {
        write_wav(path, &last_waveform, 24_000, 1.0)?;
    }

    let audio_seconds = last_waveform.len() as f64 / 24_000.0;
    let gen_mean = mean(&generation_seconds);
    let decode_mean = mean(&decode_seconds);
    let e2e_mean = total_seconds / iterations as f64;
    println!(
        "HIP e2e bench: frames={}, samples={}, audio_seconds={audio_seconds:.6}, iterations={iterations}, warmup={warmup}, load_seconds={load_seconds:.6}, generation_mean={gen_mean:.6}, decode_mean={decode_mean:.6}, e2e_mean={e2e_mean:.6}, generation_rtf={:.6}, decode_rtf={:.6}, e2e_rtf={:.6}, output_wav={:?}",
        max_frames,
        last_waveform.len(),
        gen_mean / audio_seconds,
        decode_mean / audio_seconds,
        e2e_mean / audio_seconds,
        output_wav
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn rollout(
    talker: &HipTalker,
    predictor: &HipCodePredictor,
    prefill: &DeviceBuffer<f32>,
    cp_prefix: &DeviceBuffer<f32>,
    acoustic_sum: &DeviceBuffer<f32>,
    trailing: &[DeviceBuffer<f32>],
    prefill_steps: usize,
    max_frames: usize,
) -> qwen3_hip_runtime::Result<Vec<i32>> {
    let mut semantic = talker.prefill_token(prefill, prefill_steps)?;
    let mut generated = Vec::with_capacity(max_frames * CODE_GROUPS);
    for frame in 0..max_frames {
        if semantic == CODEC_EOS_TOKEN {
            break;
        }
        talker.prepare_code_predictor_prefix(cp_prefix)?;
        let acoustic = predictor.generate_to_buffer(cp_prefix, acoustic_sum)?;
        generated.push(semantic);
        generated.extend(acoustic);
        if frame + 1 < max_frames {
            talker.build_step_input(acoustic_sum, &trailing[frame])?;
            semantic = talker.decode_prepared_token(prefill_steps + frame)?;
        }
    }
    Ok(generated)
}

fn decode_waveform(
    runtime: &HipRuntime,
    decoder: &HipCodecInitial,
    codes: &[i32],
) -> qwen3_hip_runtime::Result<Vec<f32>> {
    let frames = codes.len() / CODE_GROUPS;
    let initial = decoder.run(runtime, codes, frames)?;
    let pre_transformer =
        decoder.run_pre_transformer(runtime, &initial.pre_conv, initial.frames)?;
    let upsample = decoder.run_upsample_stages(runtime, &pre_transformer, initial.frames)?;
    let output = decoder.run_decoder_stages(runtime, &upsample.upsample_1_1, upsample.frames_1)?;
    runtime.synchronize()?;
    output.waveform.copy_to_host()
}

fn upload_trailing(
    runtime: &HipRuntime,
    trailing: &[f32],
    tts_pad: &[f32],
    frames: usize,
) -> qwen3_hip_runtime::Result<Vec<DeviceBuffer<f32>>> {
    let mut buffers = Vec::with_capacity(frames);
    for frame in 0..frames {
        let offset = frame * HIDDEN;
        if offset + HIDDEN <= trailing.len() {
            buffers.push(runtime.buffer_from_slice(&trailing[offset..offset + HIDDEN])?);
        } else {
            buffers.push(runtime.buffer_from_slice(tts_pad)?);
        }
    }
    Ok(buffers)
}

fn parse_arg(
    value: Option<std::ffi::OsString>,
    name: &str,
) -> qwen3_hip_runtime::Result<Option<usize>> {
    value
        .map(|value| {
            value
                .to_string_lossy()
                .parse::<usize>()
                .map_err(|err| Error::InvalidInput(format!("invalid {name}: {err}")))
        })
        .transpose()
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}
