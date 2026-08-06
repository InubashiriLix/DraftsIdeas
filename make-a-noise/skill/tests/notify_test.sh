#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
notify="$ROOT/scripts/notify.sh"
tmpdir="$(mktemp -d)"
trap 'rm -rf "$tmpdir"' EXIT

fail=0
ok() { echo "ok   - $1"; }
failed() { echo "FAIL - $1"; fail=1; }
skip() { echo "skip - $1"; }

# --- static checks ---
bash -n "$notify"
ok "bash -n scripts/notify.sh"

if command -v shellcheck >/dev/null 2>&1; then
  shellcheck "$notify"
  ok "shellcheck scripts/notify.sh"
else
  skip "shellcheck (not installed)"
fi

export NOISE_BIN=/usr/bin/echo
export NOISE_STATE_DIR="$tmpdir/state"
label="PlantVillage数据集分割任务"

# --- five event texts (announcement: <label>：<message>) ---
expect_text() {
  local event="$1" want="$2" out
  out="$("$notify" "$event" "$label")"
  if [[ "$out" == "say ${label}：${want}" ]]; then
    ok "$event -> $want"
  else
    failed "$event (got: $out)"
  fi
}

expect_text progress "有新进展"
expect_text confirm "需要确认方案"
expect_text approval "需要授权"
expect_text blocked "任务受阻，需要处理"
expect_text done "任务完成"

# --- armed gate auto-created ---
if [[ -f "$NOISE_STATE_DIR/armed" ]]; then
  ok "armed gate auto-created"
else
  failed "armed gate missing"
fi

# --- usage errors: exit 64 ---
expect_exit() {
  local name="$1" want="$2"; shift 2
  set +e
  "$@" >/dev/null 2>&1
  local got=$?
  set -e
  if [[ $got -eq $want ]]; then
    ok "$name"
  else
    failed "$name (exit=$got, want=$want)"
  fi
}

expect_exit "no args -> 64" 64 "$notify"
expect_exit "one arg -> 64" 64 "$notify" done
expect_exit "three args -> 64" 64 "$notify" done "$label" extra
expect_exit "empty label -> 64" 64 "$notify" done ""
expect_exit "whitespace label -> 64" 64 "$notify" done "   "
expect_exit "unknown event -> 64" 64 "$notify" nope "$label"

# --- concurrency: playback serialized via flock ---
(
  exec 9>"$NOISE_STATE_DIR/playback.lock"
  flock 9
  sleep 2
) &
locker=$!
sleep 0.2
start="$(date +%s%N)"
out="$("$notify" done "$label")"
end="$(date +%s%N)"
wait "$locker"
elapsed_ms="$(( (end - start) / 1000000 ))"

if (( elapsed_ms >= 1500 )); then
  ok "waits for lock (serialized, ${elapsed_ms}ms)"
else
  failed "did not wait for lock (${elapsed_ms}ms)"
fi

if [[ "$out" == "say ${label}：任务完成" ]]; then
  ok "content correct after lock wait"
else
  failed "content after lock wait (got: $out)"
fi

# --- summary ---
echo
if [[ "$fail" -eq 0 ]]; then
  echo "All tests passed."
else
  echo "$fail test group(s) failed." >&2
  exit 1
fi
