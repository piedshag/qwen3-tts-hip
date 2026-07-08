# Tools

This repository keeps production and currently supported diagnostics in `src/bin`.
Older one-off smoke tests and experiments live under `tools/experiments/bin` so
Cargo does not build every historical experiment during normal checks.

## Supported Binaries

- `tts-server`: std-only HTTP demo server for WAV generation and PCM streaming.
- `hip-tts-generate`: command-line generation, WAV writing, and code parity helper.
- `hip-engine-bench`: public `HipTtsEngine` benchmark with generation and decode RTF.
- `hip-engine-profile`: stage/GEMM-shape profiler for the public engine path.
- `hip-stream-bench`: chunk-by-chunk streaming latency and RTF diagnostic.
- `hip-codec-bench`: codec decode benchmark using generated codes.
- `hip-codec-decode`: decode `.npy` code fixtures to audio and optionally compare/write WAV.
- `hip-text-prep-parity`: CustomVoice text-prep parity against Python fixtures.
- `code-predictor-parity`: CodePredictor token and embedding parity.
- `hip-codec-initial-parity`: codec-stage parity through waveform.
- `talker-prefill-parity`: focused Talker prefill/decode parity diagnostic.

## Archived Experiments

The files in `tools/experiments/bin` are preserved for reference but are not built
automatically by Cargo. They include early HIP kernel smokes, attention/decoder-stack
smokes, graph experiments, low-level GEMV/GEMM experiments, and older rollout/e2e
benchmark harnesses.

If an archived experiment becomes useful again, move it back to `src/bin` or add an
explicit `[[bin]]` entry in `Cargo.toml` for the duration of the investigation.
