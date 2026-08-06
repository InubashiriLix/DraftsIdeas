#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: ${0##*/} <progress|confirm|approval|blocked|done> <session-label>" >&2
  exit 64
}

if [[ $# -ne 2 ]]; then
  usage
fi

event="$1"
label="$2"

case "$event" in
  progress) message="有新进展" ;;
  confirm) message="需要确认方案" ;;
  approval) message="需要授权" ;;
  blocked) message="任务受阻，需要处理" ;;
  done) message="任务完成" ;;
  *)
    echo "Unknown notification event: $event" >&2
    exit 64
    ;;
esac

if [[ "$label" != *[![:space:]]* ]]; then
  echo "Session label must not be empty" >&2
  exit 64
fi

noise_state_dir="${NOISE_STATE_DIR:-$HOME/.cache/noise}"
noise_bin="${NOISE_BIN:-$HOME/.local/bin/noise}"

mkdir -p "$noise_state_dir"
touch "$noise_state_dir/armed"

exec 9>"$noise_state_dir/playback.lock"
if ! flock -w 30 9; then
  echo "Timed out waiting for the noise playback lock" >&2
  exit 75
fi

exec "$noise_bin" say "${label}：${message}"
