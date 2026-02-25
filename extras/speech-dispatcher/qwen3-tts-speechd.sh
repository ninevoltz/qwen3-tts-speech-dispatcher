#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${QWEN_TTS_REPO:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${QWEN_SPEECHD_PYTHON:-/home/john/anaconda3/envs/qwen3-tts/bin/python}"
cd "${REPO_ROOT}"

if [[ -t 0 ]]; then
  TEXT_INPUT="${DATA:-}"
else
  TEXT_INPUT="$(cat)"
  if [[ -z "${TEXT_INPUT}" && -n "${DATA:-}" ]]; then
    TEXT_INPUT="${DATA}"
  fi
fi

if [[ -z "${TEXT_INPUT}" ]]; then
  exit 0
fi

WAV_OUT="$(mktemp --tmpdir qwen3-speechd-XXXXXX.wav)"
cleanup() {
  rm -f "${WAV_OUT}"
}
trap cleanup EXIT

if ! printf '%s' "${TEXT_INPUT}" | \
  "${PYTHON_BIN}" -m qwen_tts.cli.speechd_provider \
    --output "${WAV_OUT}" \
    --voice "${VOICE:-}" \
    --rate "${RATE:-}" \
    --pitch "${PITCH:-}" \
    --language "${LANGUAGE:-auto}"; then
  echo "qwen3-tts-speechd: synthesis timed out or failed." >&2
  exit 1
fi

if [[ -n "${QWEN_SPEECHD_PLAY_CMD:-}" ]]; then
  ${QWEN_SPEECHD_PLAY_CMD} "${WAV_OUT}"
elif command -v paplay >/dev/null 2>&1; then
  paplay "${WAV_OUT}"
elif command -v aplay >/dev/null 2>&1; then
  aplay -q "${WAV_OUT}"
elif command -v play >/dev/null 2>&1; then
  play -q "${WAV_OUT}"
else
  echo "qwen3-tts-speechd: no playback command found (paplay/aplay/play)." >&2
  exit 1
fi
