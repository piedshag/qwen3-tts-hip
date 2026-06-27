#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
PY_REF="$ROOT/python-reference/qwen3_tts_reference.py"
OUT_ROOT="${QWEN3_FIXTURE_ROOT:-$ROOT/python-reference/out/custom_voice_0p6b_rocm_long}"
MODEL="${QWEN3_MODEL:-Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice}"
MODEL_DIR="${QWEN3_MODEL_DIR:-/home/flynn/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5}"
PY_SRC="${QWEN3_TTS_PYTHON_SRC:-/home/flynn/Qwen3-TTS}"
TEXT="${QWEN3_TEXT:-She said she would be here by noon.}"
DEVICE="${QWEN3_PY_DEVICE:-cuda:0}"
DTYPE="${QWEN3_PY_DTYPE:-float32}"

export QWEN3_TTS_PYTHON_SRC="$PY_SRC"

cd "$ROOT/python-reference"

uv run "$PY_REF" code-predictor \
  --repo "$PY_SRC" \
  --model "$MODEL" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --local-files-only \
  --out "$OUT_ROOT/code_predictor_f32"

uv run "$PY_REF" custom-voice-talker \
  --repo "$PY_SRC" \
  --model "$MODEL" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --local-files-only \
  --text "$TEXT" \
  --speaker Ryan \
  --language English \
  --max-new-tokens 12 \
  --out "$OUT_ROOT/talker_f32_rollout12"

uv run "$PY_REF" custom-voice-talker \
  --repo "$PY_SRC" \
  --model "$MODEL" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --local-files-only \
  --text "$TEXT" \
  --speaker Ryan \
  --language English \
  --max-new-tokens 48 \
  --out "$OUT_ROOT/talker_f32_rollout48"

uv run "$PY_REF" codec-stages \
  --repo "$PY_SRC" \
  --model-dir "$MODEL_DIR" \
  --codes "$OUT_ROOT/talker_f32_rollout48/rollout_codes.npy" \
  --device "$DEVICE" \
  --dtype "$DTYPE" \
  --local-files-only \
  --out "$OUT_ROOT/codec_stages_rollout39_f32"
