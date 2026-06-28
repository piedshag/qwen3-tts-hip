use std::path::PathBuf;
use std::time::Instant;

use qwen3_hip_runtime::{Error, GenerateOptions, HipTtsEngine, Language, Result, Speaker};

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let model_dir = args.next().map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("/home/flynn/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5")
    });
    let text = args
        .next()
        .map(|value| value.to_string_lossy().into_owned())
        .unwrap_or_else(|| "She said she would be here by noon.".to_string());
    let max_frames = parse_usize_arg(args.next(), "max_frames")?.unwrap_or(240);
    let iterations = parse_usize_arg(args.next(), "iterations")?.unwrap_or(5);
    let warmup = parse_usize_arg(args.next(), "warmup")?.unwrap_or(1);
    let speaker = args
        .next()
        .map(|value| value.to_string_lossy().parse::<Speaker>())
        .transpose()?
        .unwrap_or(Speaker::Ryan);
    let language = args
        .next()
        .map(|value| value.to_string_lossy().parse::<Language>())
        .transpose()?
        .unwrap_or(Language::English);

    if max_frames == 0 || iterations == 0 {
        return Err(Error::InvalidInput(
            "max_frames and iterations must be non-zero".to_string(),
        ));
    }

    let load_start = Instant::now();
    let engine = HipTtsEngine::load_with_max_frames(&model_dir, 0, max_frames)?;
    engine.runtime().synchronize()?;
    let load_seconds = load_start.elapsed().as_secs_f64();

    let generation_start = Instant::now();
    let codes = engine.generate_codes(
        &text,
        GenerateOptions {
            speaker,
            language,
            max_frames,
            decode_audio: false,
            ..GenerateOptions::default()
        },
    )?;
    let generation_seconds = generation_start.elapsed().as_secs_f64();

    let mut last_samples = Vec::new();
    for _ in 0..warmup {
        last_samples = engine.decode_codes(&codes.codes)?;
    }

    let mut decode_seconds = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        let decode_start = Instant::now();
        last_samples = engine.decode_codes(&codes.codes)?;
        decode_seconds.push(decode_start.elapsed().as_secs_f64());
    }

    let audio_seconds =
        last_samples.len() as f64 / qwen3_hip_runtime::generation::SAMPLE_RATE as f64;
    let decode_mean = mean(&decode_seconds);
    println!(
        "HIP codec bench: frames={}, samples={}, ended_by_eos={}, audio_seconds={audio_seconds:.6}, iterations={iterations}, warmup={warmup}, load_seconds={load_seconds:.6}, generation_seconds={generation_seconds:.6}, decode_mean={decode_mean:.6}, decode_rtf={:.6}",
        codes.frames,
        last_samples.len(),
        codes.ended_by_eos,
        decode_mean / audio_seconds,
    );
    Ok(())
}

fn parse_usize_arg(value: Option<std::ffi::OsString>, name: &str) -> Result<Option<usize>> {
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
