#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
MODEL_DIR="${QWEN3_BASE_MODEL_DIR:-/home/flynn/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-1.7B-Base/snapshots/fd4b254389122332181a7c3db7f27e918eec64e3}"
PROMPT_JSON="${QWEN3_VOICE_CLONE_PROMPT_JSON:-}"
FIXTURE_ROOT="${QWEN3_VOICE_CLONE_FIXTURE_ROOT:-$ROOT/python-reference/out/voice_clone_xvector_1p7b}"
TEXT="${QWEN3_TEXT:-She said she would be here by noon.}"
LANGUAGE="${QWEN3_LANGUAGE:-English}"
MAX_NEW_TOKENS="${QWEN3_MAX_NEW_TOKENS:-12}"
PROFILE="${QWEN3_CARGO_PROFILE:-timing}"
PYTHON="${QWEN3_PYTHON:-$ROOT/python-reference/.venv/bin/python}"

if [[ -z "$PROMPT_JSON" ]]; then
  cat >&2 <<EOF
QWEN3_VOICE_CLONE_PROMPT_JSON is required and must point to an x-vector-only prompt.json artifact.

Example:
  QWEN3_VOICE_CLONE_PROMPT_JSON=/path/to/prompt.json $0
EOF
  exit 2
fi

export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export HIP_PATH="${HIP_PATH:-/opt/rocm}"
export LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm-7.2.4/lib:${LD_LIBRARY_PATH:-}"

cd "$ROOT"

"$PYTHON" python-reference/qwen3_tts_reference.py voice-clone \
  --model "$MODEL_DIR" \
  --device "${QWEN3_PY_DEVICE:-cuda:0}" \
  --dtype "${QWEN3_PY_DTYPE:-float16}" \
  --attn-implementation "${QWEN3_PY_ATTN:-eager}" \
  --out "$FIXTURE_ROOT" \
  --text "$TEXT" \
  --language "$LANGUAGE" \
  --prompt-json "$PROMPT_JSON" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --no-do-sample \
  --repetition-penalty 1.0 \
  --no-subtalker-dosample \
  --subtalker-top-k 50 \
  --subtalker-top-p 1.0 \
  --subtalker-temperature 1.0 \
  --seed 0 \
  --no-non-streaming-mode

FRAMES="$($PYTHON - "$FIXTURE_ROOT/talker_codes.npy" <<'PY'
import sys
import numpy as np

print(np.load(sys.argv[1]).shape[0])
PY
)"

cargo run --profile "$PROFILE" --bin hip-custom-voice-generate -- \
  "$MODEL_DIR" \
  "$TEXT" \
  "$FRAMES" \
  "$FIXTURE_ROOT/talker_codes.npy" \
  Ryan \
  "$LANGUAGE" \
  - \
  1.0 \
  1.0 \
  false \
  50 \
  1.0 \
  1.0 \
  false \
  50 \
  1.0 \
  1.0 \
  0 \
  1 \
  "$PROMPT_JSON"
