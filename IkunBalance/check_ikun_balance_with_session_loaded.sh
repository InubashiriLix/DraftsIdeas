#!/usr/bin/env bash
# 本文件保留旧文件名作为本地入口；实际逻辑统一由 check.sh 维护。
#
# 新版 New API 通常需要短期 Bearer token。推荐这样运行：
#   IKUN_ACCESS_TOKEN='Bearer ...' ./check_ikun_balance_with_session_loaded.sh
#
# 如果 session cookie 仍可刷新，也可以只设置：
#   IKUN_SESSION_COOKIE='session=...' ./check_ikun_balance_with_session_loaded.sh
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/check.sh" "$@"
