#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${QWEN_TTS_REPO:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
PYTHON_BIN="${QWEN_SPEECHD_PYTHON:-/home/john/anaconda3/envs/qwen3-tts/bin/python}"
SOCKET_PATH="${QWEN_SPEECHD_SOCKET:-/tmp/qwen3-tts-speechd.sock}"
READY_TIMEOUT_SEC="${QWEN_SPEECHD_READY_TIMEOUT_SEC:-90}"
LOG_FILE="${QWEN_SPEECHD_LOG:-/tmp/qwen3-tts-speechd.log}"
LAST_WAV="${QWEN_SPEECHD_LAST_WAV:-/tmp/qwen3-tts-last.wav}"

cd "${REPO_ROOT}"

if [[ $# -gt 0 ]]; then
  TEXT_INPUT="$*"
else
  TEXT_INPUT="$(cat)"
fi

sanitize_text() {
  "${PYTHON_BIN}" -c '
import sys
from qwen_tts.cli.speechd_text_sanitize import sanitize_speechd_text
sys.stdout.write(sanitize_speechd_text(sys.argv[1]))
' "${1}"
}

TEXT_INPUT="$(sanitize_text "${TEXT_INPUT}")"

if [[ -z "${TEXT_INPUT//[$' \t\r\n']/}" ]]; then
  printf '[%s] empty text\n' "$(date -Is)" >> "${LOG_FILE}" 2>/dev/null || true
  exit 0
fi

WAV_OUT="$(mktemp --tmpdir qwen3-speechd-XXXXXX.wav)"
cleanup() {
  pkill -P $$ >/dev/null 2>&1 || true
  rm -f "${WAV_OUT}"
}
trap cleanup EXIT INT TERM
printf '[%s] invoked text="%s"\n' "$(date -Is)" "${TEXT_INPUT}" >> "${LOG_FILE}" 2>/dev/null || true

run_client() {
  local -a mode_args=()
  local -a instruct_args=()
  if [[ -n "${QWEN_SPEECHD_MODE:-}" ]]; then
    mode_args=(--mode "${QWEN_SPEECHD_MODE}")
  fi
  if [[ -n "${QWEN_SPEECHD_INSTRUCT:-}" ]]; then
    instruct_args=(--instruct "${QWEN_SPEECHD_INSTRUCT}")
  fi

  printf '%s' "${TEXT_INPUT}" | \
    "${PYTHON_BIN}" -m qwen_tts.cli.speechd_client \
      --socket "${SOCKET_PATH}" \
      --output "${WAV_OUT}" \
      --voice "${VOICE:-}" \
      --language "${LANGUAGE:-auto}" \
      "${instruct_args[@]}" \
      "${mode_args[@]}"
}

wait_for_daemon() {
  local tries=$((READY_TIMEOUT_SEC * 4))
  local i
  for ((i=0; i<tries; i++)); do
    if [[ -S "${SOCKET_PATH}" ]] && "${PYTHON_BIN}" - "${SOCKET_PATH}" >/dev/null 2>&1 <<'PY'
import socket, sys
path = sys.argv[1]
s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
s.settimeout(0.2)
try:
    s.connect(path)
    ok = True
except Exception:
    ok = False
finally:
    s.close()
raise SystemExit(0 if ok else 1)
PY
    then
      return 0
    fi
    sleep 0.25
  done
  return 1
}

wait_for_daemon || true

if ! run_client >> "${LOG_FILE}" 2>&1; then
  printf '[%s] run_client failed, restarting daemon and retrying\n' "$(date -Is)" >> "${LOG_FILE}" 2>/dev/null || true
  systemctl --user restart qwen3-tts-daemon.service >/dev/null 2>&1 || true
  wait_for_daemon || true
  if ! run_client >> "${LOG_FILE}" 2>&1; then
    printf '[%s] run_client retry failed; returning success to keep sd_generic healthy\n' "$(date -Is)" >> "${LOG_FILE}" 2>/dev/null || true
    exit 0
  fi
fi

cp -f "${WAV_OUT}" "${LAST_WAV}" >/dev/null 2>&1 || true

play_ok=1
if [[ -n "${QWEN_SPEECHD_PLAY_CMD:-}" ]]; then
  printf '[%s] playback: custom cmd\n' "$(date -Is)" >> "${LOG_FILE}" 2>/dev/null || true
  if ${QWEN_SPEECHD_PLAY_CMD} "${WAV_OUT}" >/dev/null 2>&1; then
    play_ok=0
  fi
fi
if [[ ${play_ok} -ne 0 ]] && command -v paplay >/dev/null 2>&1; then
  printf '[%s] playback: paplay\n' "$(date -Is)" >> "${LOG_FILE}" 2>/dev/null || true
  if paplay "${WAV_OUT}" >/dev/null 2>&1; then
    play_ok=0
  fi
fi
if [[ ${play_ok} -ne 0 ]] && command -v aplay >/dev/null 2>&1; then
  printf '[%s] playback: aplay\n' "$(date -Is)" >> "${LOG_FILE}" 2>/dev/null || true
  if aplay -q "${WAV_OUT}" >/dev/null 2>&1; then
    play_ok=0
  fi
fi
if [[ ${play_ok} -ne 0 ]] && command -v play >/dev/null 2>&1; then
  printf '[%s] playback: play\n' "$(date -Is)" >> "${LOG_FILE}" 2>/dev/null || true
  if play -q "${WAV_OUT}" >/dev/null 2>&1; then
    play_ok=0
  fi
fi
if [[ ${play_ok} -ne 0 ]]; then
  printf '[%s] playback failed for all commands\n' "$(date -Is)" >> "${LOG_FILE}" 2>/dev/null || true
fi
