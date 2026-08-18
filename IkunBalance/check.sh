#!/usr/bin/env bash
# IkunBalance — 查询 api.ikuncode.cc 钱包余额
# 依赖: curl, python3
set -euo pipefail

BASE="${IKUN_BASE_URL:-https://api.ikuncode.cc}"
SESSION_COOKIE="${IKUN_SESSION_COOKIE:-}"
ACCESS_TOKEN="${IKUN_ACCESS_TOKEN:-}"

if [[ -z "$SESSION_COOKIE" && -z "$ACCESS_TOKEN" ]]; then
  echo "缺少登录凭据。请设置 IKUN_ACCESS_TOKEN='Bearer ...'，或提供可刷新的 IKUN_SESSION_COOKIE。" >&2
  exit 1
fi

# 既允许粘贴完整请求头值，也允许只粘贴 cookie/token 本体。
if [[ -n "$SESSION_COOKIE" && "$SESSION_COOKIE" != *=* ]]; then
  SESSION_COOKIE="session=$SESSION_COOKIE"
fi
ACCESS_TOKEN="${ACCESS_TOKEN#Bearer }"

# 新版 New API 的 session cookie 不能直接访问用户接口，需要先换取短期
# access token。仍接受 IKUN_ACCESS_TOKEN，方便在 session 暂时不可用时调用。
if [[ -z "$ACCESS_TOKEN" ]]; then
  auth_response=$(curl -sS \
    --connect-timeout 10 \
    --max-time 30 \
    -X POST \
    -H 'Accept: application/json' \
    -H "Cookie: $SESSION_COOKIE" \
    "$BASE/api/user/auth/refresh")

  ACCESS_TOKEN=$(printf '%s' "$auth_response" | python3 -c '
import json
import sys

try:
    response = json.load(sys.stdin)
except (json.JSONDecodeError, UnicodeDecodeError) as error:
    print(f"刷新登录态失败：服务器未返回有效 JSON（{error}）", file=sys.stderr)
    raise SystemExit(1)

data = response.get("data") or {}
token = data.get("access_token")
if not response.get("success") or not token:
    message = response.get("message") or response.get("code") or "session 已过期"
    print(f"刷新登录态失败：{message}", file=sys.stderr)
    print("请从 DevTools 请求头复制 Authorization: Bearer ...，并设置 IKUN_ACCESS_TOKEN。", file=sys.stderr)
    raise SystemExit(1)

print(token)
')
fi

curl_args=(
  -sS
  --connect-timeout 10
  --max-time 30
  -H 'Accept: application/json'
  -H "Authorization: Bearer $ACCESS_TOKEN"
)
if [[ -n "$SESSION_COOKIE" ]]; then
  curl_args+=(-H "Cookie: $SESSION_COOKIE")
fi

response=$(curl "${curl_args[@]}" "$BASE/api/user/self")

printf '%s' "$response" | python3 -c '
import json
import sys

try:
    response = json.load(sys.stdin)
except (json.JSONDecodeError, UnicodeDecodeError) as error:
    print(f"查询余额失败：服务器未返回有效 JSON（{error}）", file=sys.stderr)
    raise SystemExit(1)

if not response.get("success"):
    message = response.get("message") or response.get("code") or "未知错误"
    print(f"查询余额失败：{message}", file=sys.stderr)
    raise SystemExit(1)

user = response.get("data") or {}
try:
    quota = int(user["quota"])
    used_quota = int(user["used_quota"])
except (KeyError, TypeError, ValueError):
    print("查询余额失败：响应缺少 quota/used_quota 字段", file=sys.stderr)
    raise SystemExit(1)

quota_per_unit = 500_000
print("用户:", user.get("username", "-"))
print("分组:", user.get("group", "-"))
print("请求数:", user.get("request_count", 0))
print(f"剩余余额: {quota} quota ≈ ¥{quota / quota_per_unit:.4f}")
print(f"已用额度: {used_quota} quota ≈ ¥{used_quota / quota_per_unit:.4f}")
'
