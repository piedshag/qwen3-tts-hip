use std::path::PathBuf;
use std::time::Instant;

use qwen3_hip_runtime::code_predictor::HipCodePredictor;
use qwen3_hip_runtime::{Error, HipRuntime};

fn main() -> qwen3_hip_runtime::Result<()> {
    let mut args = std::env::args_os().skip(1);
    let model_dir = args.next().map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("/home/flynn/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5")
    });
    let iterations = parse_arg(args.next(), "iterations")?.unwrap_or(100);
    let warmup = parse_arg(args.next(), "warmup")?.unwrap_or(10);
    if iterations == 0 {
        return Err(Error::InvalidInput(
            "iterations must be non-zero".to_string(),
        ));
    }

    let runtime = HipRuntime::new(0)?;
    let load_start = Instant::now();
    let predictor = HipCodePredictor::load(&runtime, &model_dir)?;
    runtime.synchronize()?;
    let load_seconds = load_start.elapsed().as_secs_f64();

    let hidden = predictor.hidden();
    let talker_hidden = deterministic_hidden(hidden, 17.0, 15.0);
    let semantic_embed = deterministic_hidden(hidden, 23.0, 11.0);
    let mut prefix = Vec::with_capacity(2 * hidden);
    prefix.extend_from_slice(&talker_hidden);
    prefix.extend_from_slice(&semantic_embed);
    let prefix_dev = runtime.buffer_from_slice(&prefix)?;

    let first = predictor.generate(&prefix_dev)?;
    runtime.synchronize()?;
    for _ in 0..warmup {
        let _ = predictor.generate(&prefix_dev)?;
    }
    runtime.synchronize()?;

    let start = Instant::now();
    let mut last = first.clone();
    for _ in 0..iterations {
        last = predictor.generate(&prefix_dev)?;
    }
    runtime.synchronize()?;
    let total_seconds = start.elapsed().as_secs_f64();
    let mean_seconds = total_seconds / iterations as f64;
    if first.acoustic_tokens != last.acoustic_tokens {
        return Err(Error::InvalidInput(format!(
            "CodePredictor output changed across identical runs: first={:?}, last={:?}",
            first.acoustic_tokens, last.acoustic_tokens
        )));
    }

    println!(
        "CodePredictor bench: acoustic_groups={}, hidden={hidden}, iterations={iterations}, warmup={warmup}, load_seconds={load_seconds:.6}, total_seconds={total_seconds:.6}, mean_seconds={mean_seconds:.6}, first_tokens={:?}, last_tokens={:?}, embedding_sum_first8={:?}",
        predictor.num_acoustic_groups(),
        first.acoustic_tokens,
        last.acoustic_tokens,
        &last.embedding_sum[..8]
    );
    Ok(())
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

fn deterministic_hidden(len: usize, modulus: f32, center: f32) -> Vec<f32> {
    (0..len)
        .map(|idx| ((idx % modulus as usize) as f32 - center) / modulus)
        .collect()
}
