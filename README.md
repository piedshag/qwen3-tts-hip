# qwen3-hip-runtime

Experimental ROCm/HIP runtime for a future custom Qwen3-TTS inference backend.

This crate is intentionally separate from the Burn implementation. Burn remains
the correctness/reference path; this crate is the foundation for a specialized
AMD backend with static buffers, rocBLAS GEMMs, custom HIP kernels, and HIP Graph
capture/replay.

Current scope:

- Dynamically loads versioned ROCm libraries, avoiding missing unversioned
  `libamdhip64.so` / `librocblas.so` linker issues.
- Initializes HIP on a selected device.
- Allocates device buffers and performs host/device copies.
- Creates and destroys a rocBLAS handle.
- Compiles HIPRTC kernels and launches HIP modules.
- Provides smoke tests for HIP runtime, rocBLAS SGEMM, RMSNorm, RoPE, HIP Graphs,
  and persistent Qwen decoder decode-step execution.

## Generation API

Use `HipTtsEngine` for the public text-to-speech interface. The lower-level
`gpu`, `model`, and `audio` modules remain available for diagnostics and parity
tests, but ordinary callers should not need to assemble the talker,
CodePredictor, and codec manually.

```rust,no_run
use qwen3_hip_runtime::{GenerateOptions, HipTtsEngine, Language, Speaker};

# fn main() -> qwen3_hip_runtime::Result<()> {
let engine = HipTtsEngine::load_with_max_frames("/path/to/Qwen3-TTS-12Hz-0.6B-CustomVoice", 0, 240)?;
let speech = engine.generate(
    "She said she would be here by noon.",
    GenerateOptions {
        speaker: Speaker::Ryan,
        language: Language::English,
        max_frames: 240,
        decode_audio: true,
    },
)?;

speech.write_wav("out.wav", 1.0)?;
# Ok(())
# }
```

If you need to reserve a fixed KV-cache size independent of `max_frames`, use
`HipTtsEngine::load_with_options(...)` and set `EngineOptions::max_cache_steps`.

## Python Parity Fixtures

The parity loop is self-contained in this repository. Generate local Python
reference fixtures with:

```bash
./scripts/qwen3-hip-generate-fixtures.sh
```

Then run the fast parity loop:

```bash
./scripts/qwen3-hip-parity.sh quick
```

The generated fixture data is written to `python-reference/out/` and is ignored
by git. The scripts can be pointed at alternate locations with
`QWEN3_FIXTURE_ROOT`, `QWEN3_MODEL_DIR`, `QWEN3_MODEL`, and
`QWEN3_TTS_PYTHON_SRC`.

Smoke test:

```bash
ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm \
LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm-7.2.4/lib:${LD_LIBRARY_PATH}" \
cargo run -p qwen3-hip-runtime --bin hip-smoke
```

Expected output includes:

```text
HIP smoke OK: devices=1, device=0, roundtrip=[1.0, 2.0, 3.0, 4.0]
```

rocBLAS row-major SGEMM smoke test:

```bash
ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm \
LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm-7.2.4/lib:${LD_LIBRARY_PATH}" \
cargo run -p qwen3-hip-runtime --bin sgemm-smoke
```

Expected output includes:

```text
SGEMM smoke OK: max_abs=0, output=[18.25, -7.0, -5.0, 5.0, -4.5, 10.75, -3.0, 0.625]
```

Real Qwen weight linear smoke test:

```bash
ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm \
LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm-7.2.4/lib:${LD_LIBRARY_PATH}" \
cargo run -p qwen3-hip-runtime --bin linear-weight-smoke -- \
  /home/flynn/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5
```

This loads `talker.model.layers.0.self_attn.q_proj.weight`, converts the first
8 BF16 output rows to f32, runs a `1x1024 @ 1024x8` rocBLAS projection, and
checks the result against a CPU reference.

RMSNorm HIPRTC smoke tests:

```bash
ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm \
LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm-7.2.4/lib:${LD_LIBRARY_PATH}" \
cargo run -p qwen3-hip-runtime --bin rmsnorm-smoke

ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm \
LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm-7.2.4/lib:${LD_LIBRARY_PATH}" \
cargo run -p qwen3-hip-runtime --bin rmsnorm-weight-smoke -- \
  /home/flynn/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5
```

The real-weight smoke loads `talker.model.layers.0.input_layernorm.weight` and
checks the HIP kernel against a CPU reference.

RoPE HIPRTC smoke test:

```bash
ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm \
LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm-7.2.4/lib:${LD_LIBRARY_PATH}" \
cargo run -p qwen3-hip-runtime --bin rope-smoke
```

Elementwise HIPRTC smoke test:

```bash
ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm \
LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm-7.2.4/lib:${LD_LIBRARY_PATH}" \
cargo run -p qwen3-hip-runtime --bin elementwise-smoke
```

Masked softmax HIPRTC smoke test:

```bash
ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm \
LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm-7.2.4/lib:${LD_LIBRARY_PATH}" \
cargo run -p qwen3-hip-runtime --bin softmax-smoke
```

Argmax HIPRTC smoke test:

```bash
ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm \
LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm-7.2.4/lib:${LD_LIBRARY_PATH}" \
cargo run -p qwen3-hip-runtime --bin argmax-smoke
```

Real Qwen MLP block smoke test:

```bash
ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm \
LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm-7.2.4/lib:${LD_LIBRARY_PATH}" \
cargo run -p qwen3-hip-runtime --bin mlp-block-smoke -- \
  /home/flynn/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5
```

This validates `talker.model.layers.0` post-attention RMSNorm plus MLP using
real BF16 weights, rocBLAS projections, HIP SwiGLU, and HIP residual add against
a CPU f32 reference.

Real Qwen attention projection smoke test:

```bash
ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm \
LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm-7.2.4/lib:${LD_LIBRARY_PATH}" \
cargo run -p qwen3-hip-runtime --bin attention-proj-smoke -- \
  /home/flynn/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5
```

This validates `talker.model.layers.0` q/k/v projections, q/k RMSNorm,
BSHD-to-BHSD layout conversion, and RoPE against a CPU f32 reference.

Real Qwen decoder stack smoke test:

```bash
ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm \
LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm-7.2.4/lib:${LD_LIBRARY_PATH}" \
cargo run -p qwen3-hip-runtime --bin attention-block-smoke -- \
  /home/flynn/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5
```

This validates `talker.model.layers.0` and `talker.model.layers.1`
sequentially. It covers input RMSNorm, q/k/v projections, q/k RMSNorm, RoPE,
causal QK scores, softmax, value mix, output projection, first residual add,
post-attention RMSNorm, MLP, and second residual add against a CPU f32 reference,
while reusing one set of activation buffers across layers.

Parameterized decoder stack smoke test:

```bash
ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm \
LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm-7.2.4/lib:${LD_LIBRARY_PATH}" \
cargo run -p qwen3-hip-runtime --bin decoder-stack-smoke -- \
  /home/flynn/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5 \
  4
```

This validates the first `N` talker decoder layers with one reusable activation
workspace and a CPU f32 stack reference. The default is 4 layers. Weights are
streamed one layer at a time, so this can validate the full 28-layer talker stack
without retaining all layer weights on the host or device simultaneously.

Persistent decoder stack benchmark:

```bash
ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm \
LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm-7.2.4/lib:${LD_LIBRARY_PATH}" \
cargo run -p qwen3-hip-runtime --bin decoder-stack-bench -- \
  /home/flynn/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5 \
  28 \
  20 \
  3
```

This loads device weights once, reuses compiled kernels and a persistent
activation workspace, then times repeated stack forwards.

Decode-step smoke test:

```bash
ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm \
LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm-7.2.4/lib:${LD_LIBRARY_PATH}" \
cargo run -p qwen3-hip-runtime --bin decode-step-smoke -- \
  /home/flynn/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5
```

This validates one real layer in decode-step mode: two prefix tokens populate a
K/V cache, then a single current token attends over prefix plus current cache and
is compared to the last token from a CPU full-sequence reference.

Persistent decode-step benchmark:

```bash
ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm \
LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm-7.2.4/lib:${LD_LIBRARY_PATH}" \
cargo run --profile timing -p qwen3-hip-runtime --bin decode-step-bench -- \
  /home/flynn/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5 \
  0 \
  2 \
  100 \
  10
```

This loads one layer once, preallocates decode-step workspace and K/V cache,
prefills the cache, then times repeated single-token decode steps.

Persistent decode-step stack benchmark:

```bash
ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm \
LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm-7.2.4/lib:${LD_LIBRARY_PATH}" \
cargo run --profile timing -p qwen3-hip-runtime --bin decode-step-stack-bench -- \
  /home/flynn/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5 \
  28 \
  2 \
  50 \
  5
```

This loads persistent weights and K/V caches for `N` layers, propagates prefix
hidden states through the stack while filling each layer cache, then times
repeated single-token stack decode steps.

Decode-step stack smoke test:

```bash
ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm \
LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm-7.2.4/lib:${LD_LIBRARY_PATH}" \
cargo run -p qwen3-hip-runtime --bin decode-step-stack-smoke -- \
  /home/flynn/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5 \
  2 \
  2
```

This compares persistent stack decode-step output against the last token from a
CPU f32 full-sequence reference over the same first `N` layers.

HIP Graph smoke test:

```bash
ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm \
LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm-7.2.4/lib:${LD_LIBRARY_PATH}" \
cargo run -p qwen3-hip-runtime --bin graph-smoke
```

This captures a HIPRTC kernel launch on a non-default stream, instantiates the
graph, replays it, and validates the result.

HIP Graph rocBLAS smoke test:

```bash
ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm \
LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm-7.2.4/lib:${LD_LIBRARY_PATH}" \
cargo run -p qwen3-hip-runtime --bin graph-sgemm-smoke
```

This captures a rocBLAS SGEMM and a HIPRTC residual-add kernel on the same stream,
instantiates/replays the graph, and validates the result.

Decode-step HIP Graph benchmark:

```bash
ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm \
LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm-7.2.4/lib:${LD_LIBRARY_PATH}" \
cargo run --profile timing -p qwen3-hip-runtime --bin decode-step-graph-bench -- \
  /home/flynn/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5 \
  28 \
  2 \
  100 \
  10
```

This captures one fixed decode-step stack execution, validates graph replay against
same-stream eager decode, then times replay. On the R9700 test system, the fused
decode-step path measured about `6.44 ms` per 28-layer replay versus about
`8.21 ms` before HIP Graph capture and fused QKV/gate-up projections.

CodePredictor benchmark:

```bash
ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm \
LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm-7.2.4/lib:${LD_LIBRARY_PATH}" \
cargo run --profile timing -p qwen3-hip-runtime --bin code-predictor-bench -- \
  /home/flynn/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5 \
  100 \
  10
```

This runs the real `talker.code_predictor` path from a 2-token device prefix
through 15 acoustic groups, including final norm, per-group lm heads, argmax,
embedding lookup, and embedding-sum output. It also checks that repeated runs with
identical inputs produce identical acoustic tokens. Current timing-profile result
on the R9700 test system is about `19.4 ms` per full 15-group CodePredictor call.

CodePredictor Python parity:

```bash
cd python-reference
ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm \
LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm-7.2.4/lib:${LD_LIBRARY_PATH}" \
uv run --no-sync python qwen3_tts_reference.py code-predictor \
  --model /home/flynn/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5 \
  --device cuda:0 \
  --dtype float32 \
  --attn-implementation eager \
  --out out/custom_voice_0p6b_rocm_long/code_predictor_f32

cd ..
ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm \
LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm-7.2.4/lib:${LD_LIBRARY_PATH}" \
cargo run --profile timing -p qwen3-hip-runtime --bin code-predictor-parity -- \
  /home/flynn/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5 \
  python-reference/out/custom_voice_0p6b_rocm_long/code_predictor_f32
```

The current fixture matches exactly: acoustic tokens are identical and
`embedding_sum_max_abs=0`.

Talker prefill/decode-step Python parity:

```bash
ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm \
LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm-7.2.4/lib:${LD_LIBRARY_PATH}" \
cargo run --profile timing -p qwen3-hip-runtime --bin talker-prefill-parity -- \
  /home/flynn/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5 \
  python-reference/out/custom_voice_0p6b_rocm_long/talker_f32_rollout4
```

This validates the HIP talker stack with `talker.model.norm`, `talker.codec_head`,
codec-token suppression, and argmax against the direct Python rollout fixture. It
checks both initial prefill logits and the first cached decode step.

Focused HIP rollout parity:

```bash
ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm \
LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm-7.2.4/lib:${LD_LIBRARY_PATH}" \
cargo run --profile timing -p qwen3-hip-runtime --bin hip-rollout1-parity -- \
  /home/flynn/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5 \
  python-reference/out/custom_voice_0p6b_rocm_long/talker_f32_rollout12
```

This connects HIP talker prefill, HIP semantic embedding lookup, HIP CodePredictor,
device-side step-input assembly, and HIP talker cached decode. It matches the
12-frame direct Python rollout fixture exactly. The talker KV cache must be sized
for `prefill_steps + frame_count`; under-allocating it causes later-frame drift.

HIP rollout benchmark:

```bash
ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm \
LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm-7.2.4/lib:${LD_LIBRARY_PATH}" \
cargo run --profile timing -p qwen3-hip-runtime --bin hip-rollout-bench -- \
  /home/flynn/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5 \
  python-reference/out/custom_voice_0p6b_rocm_long/talker_f32_rollout12 \
  12 \
  10 \
  2
```

This benchmarks the pure-HIP generation loop using fixture-provided prefill,
trailing-text embeddings, and TTS pad embedding. It validates generated codes
against `rollout_codes.npy` when enough fixture frames are available. On the R9700
test system, the 12-frame timing-profile run measured `0.317882 s` mean
generation time for `1.0 s` of audio, or generation RTF `0.317882`.

A longer export requested with `--max-new-tokens 48` stopped after 39 frames. The
same benchmark over those 39 frames measured `1.027077 s` mean generation time
for `3.25 s` of audio, or generation RTF `0.316024`, so the fixture-backed HIP
generation loop scales linearly at the current frame lengths.

Standalone CustomVoice text prep parity:

```bash
ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm \
LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm-7.2.4/lib:${LD_LIBRARY_PATH}" \
cargo run --profile timing -p qwen3-hip-runtime --bin hip-text-prep-parity -- \
  /home/flynn/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5 \
  python-reference/out/custom_voice_0p6b_rocm_long/talker_f32_rollout12 \
  "She said she would be here by noon." \
  Ryan \
  English
```

This validates the HIP runtime's own tokenizer, CustomVoice prompt slicing,
text projection, codec embedding, prefill construction, trailing text embeddings,
and TTS pad embedding without depending on `qwen3-tts`. Current parity against
Python is exact for token IDs, with text/prep tensor max absolute differences on
the order of `2e-6`.

Standalone CustomVoice text-to-code generation:

```bash
ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm \
LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm-7.2.4/lib:${LD_LIBRARY_PATH}" \
cargo run --profile timing -p qwen3-hip-runtime --bin hip-custom-voice-generate -- \
  /home/flynn/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5 \
  "She said she would be here by noon." \
  39 \
  python-reference/out/custom_voice_0p6b_rocm_long/talker_f32_rollout48/rollout_codes.npy \
  Ryan \
  English
```

This path starts from real text inside `qwen3-hip-runtime`, builds CustomVoice
prefill/trailing tensors itself, runs HIP talker + HIP CodePredictor generation,
and validates generated codes. It matches the 12-frame fixture and the 39-frame
fixture exactly.

Standalone codec decode and WAV output:

```bash
cargo run --profile timing -p qwen3-hip-runtime --bin hip-codec-decode -- \
  /home/flynn/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5 \
  python-reference/out/custom_voice_0p6b_rocm_long/talker_codes.npy \
  python-reference/out/custom_voice_0p6b_rocm_long/waveform.npy \
  python-reference/out/custom_voice_0p6b_rocm_long/hip_codec_decode_fixture.wav \
  1.0
```

This uses the copied standalone speech-tokenizer decoder under
`qwen3-hip-runtime`, loading `speech_tokenizer/model.safetensors`. It matches the
existing Burn decoder behavior on the same fixture; both report about
`max_abs=0.0298879`, `mean_abs=0.0006667` against the saved Python high-level
`waveform.npy` fixture.

Standalone CustomVoice text-to-WAV:

```bash
ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm \
LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm-7.2.4/lib:${LD_LIBRARY_PATH}" \
cargo run --profile timing -p qwen3-hip-runtime --bin hip-custom-voice-generate -- \
  /home/flynn/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5 \
  "She said she would be here by noon." \
  12 \
  python-reference/out/custom_voice_0p6b_rocm_long/talker_f32_rollout12/rollout_codes.npy \
  Ryan \
  English \
  python-reference/out/custom_voice_0p6b_rocm_long/hip_custom_voice_standalone_12.wav \
  1.0
```

This performs standalone CustomVoice text preparation, HIP text-to-code
generation, standalone codec waveform decode, and WAV writing from one binary.

Python reference command:

```bash
cd /home/flynn/pocket-tts-rs/python-reference
uv run --no-sync python - <<'PY'
from safetensors import safe_open
import torch

path = "/home/flynn/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5/model.safetensors"
name = "talker.model.layers.0.self_attn.q_proj.weight"
with safe_open(path, framework="pt", device="cpu") as f:
    w = f.get_tensor(name)[:8].float()
input = torch.tensor([((i % 17) - 8) / 9 for i in range(w.shape[1])], dtype=torch.float32)
print([float(x) for x in input @ w.T])
PY
```
