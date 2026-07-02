use std::path::PathBuf;
use std::time::Instant;

use qwen3_hip_runtime::{Error, GenerateOptions, HipTtsEngine, Language, Result, Speaker};

fn main() -> Result<()> {
    let mut args = std::env::args_os().skip(1);
    let model_dir = args.next().map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("/home/flynn/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-CustomVoice/snapshots/0c0e3051f131929182e2c023b9537f8b1c68adfe")
    });
    let text = args
        .next()
        .map(|value| value.to_string_lossy().into_owned())
        .unwrap_or_else(|| {
            "The speaker describes a calm morning in the city, where people walk to work, shops open their doors, and the first trains leave the station on time.".to_string()
        });
    let max_frames = parse_usize_arg(args.next(), "max_frames")?.unwrap_or(240);
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
    if max_frames == 0 {
        return Err(Error::InvalidInput(
            "max_frames must be non-zero".to_string(),
        ));
    }

    let load_start = Instant::now();
    let engine = HipTtsEngine::load_with_max_frames(&model_dir, 0, max_frames)?;
    engine.runtime().synchronize()?;
    let load_seconds = load_start.elapsed().as_secs_f64();
    let options = GenerateOptions {
        speaker,
        language,
        max_frames,
        decode_audio: false,
        ..GenerateOptions::default()
    };

    for _ in 0..warmup {
        let codes = engine.generate_codes(&text, options.clone())?;
        let _samples = engine.decode_codes(&codes.codes)?;
    }

    qwen3_hip_runtime::blas::reset_sgemm_profile();
    let profiled = engine.generate_codes_profiled(&text, options)?;
    let decode_start = Instant::now();
    let samples = engine.decode_codes(&profiled.codes.codes)?;
    let decode_seconds = decode_start.elapsed().as_secs_f64();
    let audio_seconds = samples.len() as f64 / qwen3_hip_runtime::generation::SAMPLE_RATE as f64;
    let generation_seconds = profiled.profile.total_seconds();
    let e2e_seconds = generation_seconds + decode_seconds;
    let total = e2e_seconds.max(f64::MIN_POSITIVE);

    println!(
        "HIP engine profile: frames={}, samples={}, ended_by_eos={}, audio_seconds={audio_seconds:.6}, warmup={warmup}, load_seconds={load_seconds:.6}, generation_seconds={generation_seconds:.6}, decode_seconds={decode_seconds:.6}, e2e_seconds={e2e_seconds:.6}, generation_rtf={:.6}, decode_rtf={:.6}, e2e_rtf={:.6}",
        profiled.codes.frames,
        samples.len(),
        profiled.codes.ended_by_eos,
        generation_seconds / audio_seconds,
        decode_seconds / audio_seconds,
        e2e_seconds / audio_seconds,
    );
    println!(
        "HIP engine profile split: prepare_text={:.6}s ({:.2}%), upload={:.6}s ({:.2}%), talker_prefill={:.6}s ({:.2}%), prepare_prefix={:.6}s ({:.2}%), code_predictor={:.6}s ({:.2}%), build_step_input={:.6}s ({:.2}%), talker_decode={:.6}s ({:.2}%), codec_decode={decode_seconds:.6}s ({:.2}%)",
        profiled.profile.prepare_text_seconds,
        pct(profiled.profile.prepare_text_seconds, total),
        profiled.profile.upload_seconds,
        pct(profiled.profile.upload_seconds, total),
        profiled.profile.prefill_seconds,
        pct(profiled.profile.prefill_seconds, total),
        profiled.profile.prepare_prefix_seconds,
        pct(profiled.profile.prepare_prefix_seconds, total),
        profiled.profile.code_predictor_seconds,
        pct(profiled.profile.code_predictor_seconds, total),
        profiled.profile.build_step_input_seconds,
        pct(profiled.profile.build_step_input_seconds, total),
        profiled.profile.talker_decode_seconds,
        pct(profiled.profile.talker_decode_seconds, total),
        pct(decode_seconds, total),
    );
    println!(
        "HIP engine profile core: talker_total={:.6}s ({:.2}%), code_predictor={:.6}s ({:.2}%), codec_decode={decode_seconds:.6}s ({:.2}%)",
        profiled.profile.prefill_seconds + profiled.profile.talker_decode_seconds,
        pct(
            profiled.profile.prefill_seconds + profiled.profile.talker_decode_seconds,
            total
        ),
        profiled.profile.code_predictor_seconds,
        pct(profiled.profile.code_predictor_seconds, total),
        pct(decode_seconds, total),
    );
    let gemms = qwen3_hip_runtime::blas::sgemm_profile_entries();
    if !gemms.is_empty() {
        println!(
            "HIP engine profile GEMM shapes: top={}",
            gemms.len().min(16)
        );
        for entry in gemms.iter().take(16) {
            println!(
                "  m={}, n={}, k={}, calls={}, gflop={:.3}",
                entry.m,
                entry.n,
                entry.k,
                entry.calls,
                entry.flops as f64 / 1.0e9,
            );
        }
    }
    Ok(())
}

fn pct(value: f64, total: f64) -> f64 {
    100.0 * value / total.max(f64::MIN_POSITIVE)
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
