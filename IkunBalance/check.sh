#!/usr/bin/env bash
# IkunBalance — 爬取 api.ikuncode.cc 钱包余额
# 依赖: curl, python3
# 浏览器请求同时需要 session cookie 与短期 Bearer access token
set -euo pipefail

BASE="https://api.ikuncode.cc"
SESSION_COOKIE="${IKUN_SESSION_COOKIE:-}"
ACCESS_TOKEN="${IKUN_ACCESS_TOKEN:-}"

if [[ -z "$SESSION_COOKIE" ]]; then
  echo "缺少 IKUN_SESSION_COOKIE。请从浏览器 DevTools 的请求头复制 session=..." >&2
  exit 1
fi

if [[ -z "$ACCESS_TOKEN" ]]; then
  echo "缺少 IKUN_ACCESS_TOKEN。请从浏览器 DevTools 的 Authorization: Bearer <token> 中复制 token。" >&2
  exit 1
fi

# 允许直接粘贴完整的 "Bearer <token>"，也允许只提供 token 本身。
ACCESS_TOKEN="${ACCESS_TOKEN#Bearer }"

NEW_API_USER="${IKUN_USER_ID:-28536}"

resp=$(curl -sS \
  -H "Cookie: $SESSION_COOKIE" \
  -H "New-Api-User: $NEW_API_USER" \
  "$BASE/api/user/self")

echo "$resp" | python3 -c "
import sys, json
d = json.load(sys.stdin)
if not d.get('success'):
    print('请求失败:', d.get('message'))
    sys.exit(1)
u = d['data']
quota = u['quota']
used = u['used_quota']
print('用户:', u['username'])
print('分组:', u['group'])
print('请求数:', u['request_count'])
print(f'剩余余额: {quota} quota ≈ \${quota/500000:.4f}')
print(f'已用额度: {used} quota ≈ \${used/500000:.4f}')
"
  -H "Authorization: Bearer $ACCESS_TOKEN" \
