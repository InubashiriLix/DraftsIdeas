---
name: make-a-noise
description: Notify the user with audible, labeled Chinese TTS (progress, confirmation, approval, blocked, completion) spoken through the local `noise` binary. Use implicitly for every user-visible reply when the global AGENTS.md rule enables persistent sound; also use explicitly when the user asks to be notified with sound, such as "ping me when done", "notify me with sound", "提醒我", "完成后出声". Never fire on your own initiative when sound is not enabled.
---

# make-a-noise

Audible, labeled notifications for the user, spoken through the local `noise` TTS binary (Chinese voice, plays on the user's desktop speakers via PipeWire/ALSA).

## Binary and state

- Binary: `${NOISE_BIN:-$HOME/.local/bin/noise}` (default `~/.local/bin/noise`)
- State: `${NOISE_STATE_DIR:-$HOME/.cache/noise}` (WAV cache, `armed` gate, `playback.lock`)

`NOISE_BIN` and `NOISE_STATE_DIR` may override the locations (useful when the user installed elsewhere or for testing).

## Session label

Generate one stable, short Chinese session label from the project name and the first substantive task, e.g. `PlantVillage数据集分割任务`. The user can replace it at any time; once replaced, use their label for the rest of the session.

## Opt-in gate

Sound is opt-in. The `armed` file is the shared gate:

- The global `AGENTS.md` notification rule is the user's persistent opt-in for every new session. Otherwise, arm before first use:
  ```sh
  mkdir -p "$HOME/.cache/noise" && touch "$HOME/.cache/noise/armed"
  ```
- The wrapper creates `armed` automatically on every call.
- **"本会话静音"** (mute this session): stop invoking the wrapper for the current agent only. Do NOT delete the shared `armed` file.
- **"永久关闭声音通知"** (disable sound permanently): delete `$HOME/.cache/noise/armed`.

## Notify

Before every user-visible assistant message, run exactly once:

```sh
"$HOME/.agents/skills/make-a-noise/scripts/notify.sh" <event> "<session-label>"
```

Choose exactly one event per message, the highest-priority applicable event:

- `done` → 任务完成
- `blocked` → 任务受阻，需要处理
- `approval` → 需要授权
- `confirm` → 需要确认方案
- `progress` → 有新进展

Priority: `done` > `blocked` > `approval` > `confirm` > `progress`.

Announcement format is `<session-label>：<event message>`. The wrapper serializes concurrent agents via `playback.lock` (flock, 30s max wait) so announcements never overlap. If it fails (missing binary, lock timeout), continue the task and report the first failure in text; never let audio block the reply.

## Notes

- Codex has no hook system: enable automatic invocation by copying `examples/AGENTS.md` to `~/.codex/AGENTS.md`, which instructs the agent to call the wrapper before each message.
- Model output is cached under `$HOME/.cache/noise/models/`.
