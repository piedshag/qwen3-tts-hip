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
    let iterations = parse_arg(args.next(), "iterations")?.unwrap_or(100);
    let warmup = parse_arg(args.next(), "warmup")?.unwrap_or(10);
    if layer_count == 0 || prefix_steps == 0 || iterations == 0 {
        return Err(Error::InvalidInput(
            "layer count, prefix steps, and iterations must be non-zero".to_string(),
        ));
    }
    let max_cache_steps = prefix_steps + 1;

    let runtime = HipRuntime::new(0)?;
    let stream = runtime.create_stream()?;
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
    let eager_dev = runtime.empty_buffer::<f32>(stack.input_len())?;
    let output_dev = runtime.empty_buffer::<f32>(stack.input_len())?;

    stack.prefill(&prefix_dev, prefix_steps)?;
    runtime.synchronize()?;

    for _ in 0..warmup {
        stack.decode_step_on_stream(&current_dev, &eager_dev, prefix_steps, &stream)?;
    }
    stream.synchronize()?;
    let eager_start = Instant::now();
    for _ in 0..iterations {
        stack.decode_step_on_stream(&current_dev, &eager_dev, prefix_steps, &stream)?;
    }
    stream.synchronize()?;
    let eager_total_seconds = eager_start.elapsed().as_secs_f64();
    let eager_mean_seconds = eager_total_seconds / iterations as f64;
    let eager = eager_dev.copy_to_host()?;

    stream.begin_capture()?;
    stack.decode_step_on_stream(&current_dev, &output_dev, prefix_steps, &stream)?;
    let graph = stream.end_capture()?;
    let exec = graph.instantiate()?;

    for _ in 0..warmup {
        exec.launch(&stream)?;
    }
    stream.synchronize()?;

    let start = Instant::now();
    for _ in 0..iterations {
        exec.launch(&stream)?;
    }
    stream.synchronize()?;
    let total_seconds = start.elapsed().as_secs_f64();
    let mean_seconds = total_seconds / iterations as f64;
    let layers_per_second = layer_count as f64 / mean_seconds;
    let graph_speedup = eager_mean_seconds / mean_seconds;

    let output = output_dev.copy_to_host()?;
    let max_abs = max_abs_diff(&output, &eager);
    if max_abs > 1e-5 {
        return Err(Error::InvalidInput(format!(
            "graph output mismatch: max_abs={max_abs}"
        )));
    }
    println!(
        "Decode-step graph bench: layers={layer_count}, prefix_steps={prefix_steps}, iterations={iterations}, warmup={warmup}, hidden={}, q_heads={}, kv_heads={}, head_dim={}, load_seconds={load_seconds:.6}, eager_total_seconds={eager_total_seconds:.6}, eager_mean_seconds={eager_mean_seconds:.6}, graph_total_seconds={total_seconds:.6}, graph_mean_seconds={mean_seconds:.6}, graph_speedup={graph_speedup:.3}, layers_per_second={layers_per_second:.2}, graph_vs_eager_max_abs={max_abs:.9}, output_first8={:?}",
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

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}
