# qwen3-tts-hip

Standalone Rust + ROCm/HIP runtime for Qwen3-TTS CustomVoice inference on AMD GPUs.

This crate provides a high-level text-to-speech API backed by custom HIP kernels,
rocBLAS GEMMs, HIPRTC compilation, and a native HIP codec decoder. The lower-level
GPU/model/audio modules are still public for diagnostics and experimentation, but
normal callers should use `HipTtsEngine`.

## Goals

- Optimize Qwen3-TTS inference performance for AMD ROCm/HIP systems.
- Make real-time streaming a top priority.

## Status

- Supports Qwen3-TTS 12 Hz CustomVoice 0.6B and basic 1.7B loading/generation.
- 0.6B deterministic generation matches exported Python fixtures exactly.
- Native HIP codec waveform parity passes against Python codec-stage fixtures.
- Hot 0.6B streaming generation with Qwen-default sampling is about `0.54` RTF on the local R9700 test system.
- The matching Python streaming/Qwen-default path measured about `1.91` RTF on the same system.
- The optimized deterministic greedy 0.6B path is about `0.41` RTF.
- The optimized deterministic greedy 1.7B 39-frame benchmark is about `0.50` RTF, but EOS/stopping behavior still needs more parity work.

## Requirements

- Linux x86_64
- Rust 2024 edition toolchain
- ROCm/HIP installed under `/opt/rocm`
- Qwen3-TTS model snapshot available locally
- `uv` only if generating Python parity fixtures

Typical ROCm environment:

```bash
export ROCM_PATH=/opt/rocm
export HIP_PATH=/opt/rocm
export LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm-7.2.4/lib:${LD_LIBRARY_PATH}"
```

## Library Usage

```rust,no_run
use qwen3_hip_runtime::{GenerateOptions, HipTtsEngine, Language, Speaker};

fn main() -> qwen3_hip_runtime::Result<()> {
    let model_dir = "/path/to/Qwen3-TTS-12Hz-0.6B-CustomVoice";
    let engine = HipTtsEngine::load_with_max_frames(model_dir, 0, 240)?;

    let speech = engine.generate(
        "She said she would be here by noon.",
        GenerateOptions {
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
        },
    )?;

    speech.write_wav("out.wav", 1.0)?;
    Ok(())
}
```

If you need a fixed KV-cache capacity independent of a single request's frame cap,
use `HipTtsEngine::load_with_options(...)` and set `EngineOptions::max_cache_steps`.

For incremental generation, create a persistent stream and pull code or audio chunks:

```rust,no_run
use qwen3_hip_runtime::{GenerateOptions, HipTtsEngine};

fn main() -> qwen3_hip_runtime::Result<()> {
    let engine = HipTtsEngine::load_with_max_frames("/path/to/model", 0, 240)?;
    let mut stream = engine.start_stream("She said she would be here by noon.", GenerateOptions::default())?;

    while let Some(chunk) = stream.next_audio_chunk(6)? {
        println!("streamed {} new samples", chunk.samples.len());
    }

    Ok(())
}
```

## Python Parity Fixtures

The parity workflow is self-contained in this repository. Generated fixture data
is written to `python-reference/out/` and ignored by git.

Generate fixtures:

```bash
./scripts/qwen3-hip-generate-fixtures.sh
```

Run the quick parity loop:

```bash
./scripts/qwen3-hip-parity.sh quick
```

Run the full parity loop, including 39-frame generation and codec-stage checks:

```bash
./scripts/qwen3-hip-parity.sh full
```

Useful environment overrides:

```bash
QWEN3_MODEL_DIR=/path/to/local/snapshot
QWEN3_MODEL=Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice
QWEN3_FIXTURE_ROOT=/path/to/fixtures
QWEN3_TTS_PYTHON_SRC=/path/to/Qwen3-TTS
QWEN3_TEXT="She said she would be here by noon."
QWEN3_PY_DEVICE=cuda:0
QWEN3_PY_DTYPE=float32
```

Quick parity currently checks:

- tokenizer/text-prep parity
- CodePredictor token and embedding-sum parity
- 12-frame exact text-to-code parity

Full parity additionally checks:

- 39-frame exact text-to-code parity
- native HIP codec stages through final waveform

## Development Checks

```bash
cargo fmt --check
cargo check
./scripts/qwen3-hip-parity.sh quick
```

The repository also contains lower-level smoke, parity, and benchmark binaries for
HIP runtime primitives, decoder stacks, graph capture, CodePredictor, talker, and
codec debugging. They are intentionally not part of the public API.

## Module Layout

- `generation`: public text-to-speech API
- `gpu`: HIP runtime, buffers, rocBLAS, HIPRTC kernels, graph support
- `model`: Qwen talker, CodePredictor, text prep, config, weights, decoder stack
- `audio`: codec decoder and WAV helpers
- `python-reference`: Python fixture exporter used by parity scripts

## Notes

- The runtime dynamically loads versioned ROCm libraries such as `libamdhip64.so*`,
  `libhiprtc.so*`, and `librocblas.so*`.
- Model weights are expected in Hugging Face snapshot layout, including
  `model.safetensors`, tokenizer files, and `speech_tokenizer/model.safetensors`.
- `HipTtsEngine::start_stream(...)` exposes persistent frame-incremental generation.
  Audio chunks use Qwen-style left-context window decode, currently with 25 context
  frames. Lower-latency persistent codec-cache streaming is still future work.
