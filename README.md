# qwen3-tts-hip

Standalone Rust + ROCm/HIP runtime for Qwen3-TTS CustomVoice and Base x-vector
voice-clone inference on AMD GPUs.

This crate provides a high-level text-to-speech API backed by custom HIP kernels,
rocBLAS GEMMs, HIPRTC compilation, and a native HIP codec decoder. The lower-level
GPU/model/audio modules are still public for diagnostics and experimentation, but
normal callers should use `HipTtsEngine`.

## Goals

- Optimize Qwen3-TTS inference performance for AMD ROCm/HIP systems.
- Make real-time streaming a top priority.

## Status

- Supports Qwen3-TTS 12 Hz CustomVoice 0.6B, basic CustomVoice 1.7B, and
  precomputed x-vector voice-clone prompts on Base models.
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

`EngineOptions::max_cache_steps` sets the initial Talker KV-cache capacity. The cache
grows geometrically as prefill or generation exceeds that size, preserving the full
attention history. Set a larger initial value to avoid occasional reallocation pauses
on long requests.

Useful generation options:

- `max_frames`: maximum generated acoustic frames; each frame is 1,920 samples, or 80 ms at 24 kHz.
- `do_sample`, `top_k`, `top_p`, `temperature`, `repetition_penalty`: semantic token sampling controls.
- `subtalker_dosample`, `subtalker_top_k`, `subtalker_top_p`, `subtalker_temperature`: acoustic CodePredictor sampling controls.

Streaming-specific options live in `StreamOptions`, including
`text_lookahead_tokens` for the fixed-text streaming prefill and
`left_context_frames` for audio chunk decoding context.

Base-model voice cloning currently supports precomputed x-vector-only prompt JSON
artifacts. Export one with the Python reference helper, then load it with
`VoiceClonePrompt::from_json(...)` and call `HipTtsEngine::generate_voice_clone(...)`.
Full reference-audio encoding and ICL/ref-code prompting are not ported to Rust yet.

```rust,no_run
use qwen3_hip_runtime::{GenerateOptions, HipTtsEngine, VoiceClonePrompt};

fn main() -> qwen3_hip_runtime::Result<()> {
    let engine = HipTtsEngine::load_with_max_frames("/path/to/Qwen3-TTS-12Hz-1.7B-Base", 0, 240)?;
    let prompt = VoiceClonePrompt::from_json("prompt.json")?;
    let speech = engine.generate_voice_clone(
        "We are going to build something tremendous.",
        &prompt,
        GenerateOptions::default(),
    )?;
    speech.write_wav("clone.wav", 1.0)?;
    Ok(())
}
```

Run deterministic x-vector voice-clone parity from a prompt artifact:

```bash
QWEN3_VOICE_CLONE_PROMPT_JSON=/path/to/prompt.json \
  ./scripts/qwen3-hip-voice-clone-parity.sh
```

The parity script exports Python Base-model codes with sampling disabled and compares
Rust generation against the exported `.npy` using `text_lookahead_tokens=1`.

The std-only HTTP demo server also supports the same precomputed x-vector prompt.
Start `tts-server` with the Base model and pass the prompt JSON as the optional fourth
argument. The browser UI then exposes a voice-mode selector for both WAV generation
and PCM streaming. `Output frames` is a per-request safety ceiling, independent of
the growable KV cache; set it high for a long request.

For incremental generation, create a persistent stream and pull code or audio chunks:

```rust,no_run
use qwen3_hip_runtime::{GenerateOptions, HipTtsEngine, StreamOptions};

fn main() -> qwen3_hip_runtime::Result<()> {
    let engine = HipTtsEngine::load_with_max_frames("/path/to/model", 0, 240)?;
    let mut stream = engine.start_stream(
        "She said she would be here by noon.",
        GenerateOptions::default(),
        StreamOptions::default(),
    )?;

    while let Some(chunk) = stream.next_audio_chunk(6)? {
        println!("streamed {} new samples", chunk.samples.len());
    }

    Ok(())
}
```

For incoming text from an LLM or another producer, use the channel-backed text stream.
`next_audio_chunk` blocks until audio is available or the text input finishes:

```rust,no_run
use qwen3_hip_runtime::{GenerateOptions, HipTtsEngine, TextStreamOptions};

fn main() -> qwen3_hip_runtime::Result<()> {
    let engine = HipTtsEngine::load_with_max_frames("/path/to/model", 0, 240)?;
    let (mut input, mut stream) = engine.start_text_stream(
        GenerateOptions::default(),
        TextStreamOptions::default(),
    )?;

    input.push_text("The model begins speaking ")?;
    input.push_text("as text chunks arrive.")?;
    input.finish()?;

    while let Some(chunk) = stream.next_audio_chunk(12)? {
        println!("{} samples", chunk.samples.len());
    }

    Ok(())
}
```

Send text from another thread or task while the consumer pulls audio. Dropping the
`TextStreamInput` also finishes the input. For event loops that need polling rather
than blocking behavior, use `start_text_stream_polling(...)`; its
`PollingTextStream::next_audio_chunk(...)` returns `IncrementalAudio`.

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

Benchmark the Python Qwen3-TTS reference path with a generic prompt:

```bash
python-reference/.venv/bin/python scripts/qwen3-python-rtf-bench.py
```

The script runs one warmup by default so measured RTF excludes first-use MIOpen and
kernel setup costs.

The repository also contains lower-level parity and benchmark binaries for active
diagnostics. Older one-off experiments are kept out of Cargo's automatic bin build;
see `docs/tools.md` for the supported and archived tool layout.

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
