#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-quick}"
ROOT="$(git rev-parse --show-toplevel)"
MODEL_DIR="${QWEN3_MODEL_DIR:-/home/flynn/.cache/huggingface/hub/models--Qwen--Qwen3-TTS-12Hz-0.6B-CustomVoice/snapshots/85e237c12c027371202489a0ec509ded67b5e4b5}"
FIXTURE_ROOT="${QWEN3_FIXTURE_ROOT:-$ROOT/python-reference/out/custom_voice_0p6b_rocm_long}"
TEXT="${QWEN3_TEXT:-She said she would be here by noon.}"
PROFILE="${QWEN3_CARGO_PROFILE:-timing}"

export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export HIP_PATH="${HIP_PATH:-/opt/rocm}"
export LD_LIBRARY_PATH="/opt/rocm/lib:/opt/rocm-7.2.4/lib:${LD_LIBRARY_PATH:-}"

cd "$ROOT"

case "$MODE" in
  quick|full) ;;
  *)
    printf 'usage: %s [quick|full]\n' "$0" >&2
    exit 2
    ;;
esac

if [[ ! -f "$FIXTURE_ROOT/talker_f32_rollout12/rollout_codes.npy" || ! -f "$FIXTURE_ROOT/code_predictor_f32/acoustic_tokens.npy" ]]; then
  cat >&2 <<EOF
Missing parity fixtures under:
  $FIXTURE_ROOT

Generate them with:
  ./scripts/qwen3-hip-generate-fixtures.sh

Override paths with QWEN3_FIXTURE_ROOT, QWEN3_MODEL_DIR, QWEN3_MODEL, and QWEN3_TTS_PYTHON_SRC.
EOF
  exit 1
fi

cargo check

cargo run --profile "$PROFILE" --bin hip-text-prep-parity -- \
  "$MODEL_DIR" \
  "$FIXTURE_ROOT/talker_f32_rollout12" \
  "$TEXT" \
  Ryan \
  English

cargo run --profile "$PROFILE" --bin code-predictor-parity -- \
  "$MODEL_DIR" \
  "$FIXTURE_ROOT/code_predictor_f32"

cargo run --profile "$PROFILE" --bin hip-custom-voice-generate -- \
  "$MODEL_DIR" \
  "$TEXT" \
  12 \
  "$FIXTURE_ROOT/talker_f32_rollout12/rollout_codes.npy" \
  Ryan \
  English \
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
  1

if [[ "$MODE" == "full" ]]; then
  cargo run --profile "$PROFILE" --bin hip-custom-voice-generate -- \
    "$MODEL_DIR" \
    "$TEXT" \
    39 \
    "$FIXTURE_ROOT/talker_f32_rollout48/rollout_codes.npy" \
    Ryan \
    English \
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
    1

  cargo run --profile "$PROFILE" --bin hip-codec-initial-parity -- \
    "$MODEL_DIR" \
    "$FIXTURE_ROOT/codec_stages_rollout39_f32"
fi
