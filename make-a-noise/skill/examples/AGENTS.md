# 全局中文语音通知

用户已授权使用 `make-a-noise` 技能。在每个会话中默认启用中文语音通知；除非用户说“本会话静音”或“永久关闭声音通知”，不要再次请求授权。

- 在首个实质任务中生成并记住一个简短、可朗读的中文会话标签：优先用项目名加任务摘要，例如“PlantVillage数据集分割任务”。用户指定标签时立即替换并在本会话沿用。
- 每条面向用户的助手消息前，只运行一次：

  ```sh
  "$HOME/.agents/skills/make-a-noise/scripts/notify.sh" <event> "<session-label>"
  ```

  按优先级选择事件：`done` > `blocked` > `approval` > `confirm` > `progress`。
- `progress` 用于普通回复；提问或等待选择用 `confirm`；请求工具权限用 `approval`；无法继续用 `blocked`；任务真正完成用 `done`。
- 用户说“本会话静音”或“本会话恢复声音”时，仅在当前会话停止或恢复调用脚本，不要影响其他会话，也不要删除共享的 `$HOME/.cache/noise/armed`。用户说“永久关闭声音通知”时，删除 `$HOME/.cache/noise/armed`。
- 若播报失败或锁等待超时，继续正常回复；仅在首次失败时简短说明，音频不得阻塞回复。
