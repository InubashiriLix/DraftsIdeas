# make-a-noise

Audible, labeled Chinese TTS notifications for agent CLIs (Codex, etc.), spoken through the local `noise` binary.

Each announcement carries a stable session/task label, e.g. “PlantVillage数据集分割任务：需要确认方案”. Five event types:

| event     | message             |
| --------- | ------------------- |
| `progress`| 有新进展             |
| `confirm` | 需要确认方案          |
| `approval`| 需要授权             |
| `blocked` | 任务受阻，需要处理     |
| `done`    | 任务完成             |

Concurrent agents are serialized with `flock` (30s max wait) so announcements never overlap.

## Requirements

- The `noise` binary, default `$HOME/.local/bin/noise` (build from the Rust source and install as `~/.local/bin/noise`).
- `flock` (util-linux) — present on virtually all Linux distros.
- A desktop audio backend (PipeWire/ALSA) for playback.

Locations are overridable: `NOISE_BIN` and `NOISE_STATE_DIR`.

## Install (user-level, any repo)

```sh
mkdir -p ~/.agents/skills
cp -r SKILL.md agents scripts ~/.agents/skills/make-a-noise/
```

## Enable globally (Codex)

Copy the template so every new Codex session auto-notifies:

```sh
cp examples/AGENTS.md ~/.codex/AGENTS.md
```

Then restart or open a new Codex session. The AGENTS.md rule instructs the agent to generate a Chinese session label and call the wrapper before every user-visible message.

## Usage

```sh
~/.agents/skills/make-a-noise/scripts/notify.sh <progress|confirm|approval|blocked|done> "<session-label>"
```

The wrapper auto-creates the shared `armed` gate, serializes playback, and speaks `<session-label>：<事件文案>` via `noise say`.

## Mute / disable

- **本会话静音** — stop calling the wrapper in the current session only; the shared `armed` file is untouched.
- **永久关闭声音通知** — `rm -f ~/.cache/noise/armed`.

## Uninstall

```sh
rm -rf ~/.agents/skills/make-a-noise
# remove this section from ~/.codex/AGENTS.md (or delete the file)
rm -f ~/.cache/noise/armed
```

## Verify

```sh
bash -n scripts/notify.sh tests/notify_test.sh
shellcheck scripts/notify.sh tests/notify_test.sh   # if installed
tests/notify_test.sh
```

Expected: five event texts verified, `armed` created, exit 64 on no/one/three args, empty/whitespace label and unknown event, and serialized playback under lock. Final line: `All tests passed.`

## License

MIT
