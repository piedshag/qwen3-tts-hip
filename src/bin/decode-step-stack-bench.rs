use std::path::PathBuf;
use std::time::Instant;

use qwen3_hip_runtime::decode::DecodeStepStack;
use qwen3_hip_runtime::{Error, HipRuntime};

fn main() -> qwen3_hip_runtime::Result<()> {
    let mut args = std::env::args_os().skip(1);
    let model_dir = args.next().map(PathBuf::from).unwrap_or_else(|| {
        PathBuf::from("/home/flynn/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5")
    });
    let layer_count = parse_arg(args.next(), "layer count")?.unwrap_or(28);
    let prefix_steps = parse_arg(args.next(), "prefix steps")?.unwrap_or(2);
    let iterations = parse_arg(args.next(), "iterations")?.unwrap_or(50);
    let warmup = parse_arg(args.next(), "warmup")?.unwrap_or(5);
    if layer_count == 0 || prefix_steps == 0 || iterations == 0 {
        return Err(Error::InvalidInput(
            "layer count, prefix steps, and iterations must be non-zero".to_string(),
        ));
    }
    let max_cache_steps = prefix_steps + 1;

    let runtime = HipRuntime::new(0)?;
    let load_start = Instant::now();
    let stack = DecodeStepStack::load(&runtime, &model_dir, layer_count, max_cache_steps)?;
    runtime.synchronize()?;
    let load_seconds = load_start.elapsed().as_secs_f64();
    let dims = stack.dims();

    let hidden = deterministic_hidden(max_cache_steps * dims.hidden);
    let prefix = &hidden[..prefix_steps * dims.hidden];
    let current = &hidden[prefix_steps * dims.hidden..];
    let prefix_dev = runtime.buffer_from_slice(prefix)?;
    let current_dev = runtime.buffer_from_slice(current)?;
    let output_dev = runtime.empty_buffer::<f32>(stack.input_len())?;

    stack.prefill(&prefix_dev, prefix_steps)?;
    for _ in 0..warmup {
        stack.decode_step(&current_dev, &output_dev, prefix_steps)?;
    }
    runtime.synchronize()?;

    let start = Instant::now();
    for _ in 0..iterations {
        stack.decode_step(&current_dev, &output_dev, prefix_steps)?;
    }
    runtime.synchronize()?;
    let total_seconds = start.elapsed().as_secs_f64();
    let mean_seconds = total_seconds / iterations as f64;
    let layers_per_second = layer_count as f64 / mean_seconds;

    let output = output_dev.copy_to_host()?;
    println!(
        "Decode-step stack bench: layers={layer_count}, prefix_steps={prefix_steps}, iterations={iterations}, warmup={warmup}, hidden={}, q_heads={}, kv_heads={}, head_dim={}, load_seconds={load_seconds:.6}, total_seconds={total_seconds:.6}, mean_seconds={mean_seconds:.6}, layers_per_second={layers_per_second:.2}, output_first8={:?}",
        dims.hidden,
        dims.q_heads,
        dims.kv_heads,
        dims.head_dim,
        &output[..8]
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

fn deterministic_hidden(len: usize) -> Vec<f32> {
    (0..len)
        .map(|idx| ((idx % 31) as f32 - 15.0) / 17.0)
        .collect()
}
