# IkunBalance

爬取 [api.ikuncode.cc](https://api.ikuncode.cc) 的钱包余额。

## 用法

```bash
chmod +x check.sh
IKUN_SESSION_COOKIE='session=你的sessioncookie' ./check.sh
```

可选:`IKUN_USER_ID` 覆盖用户 id(默认 `28536`)。

## 说明

- New API 后端要求 `New-Api-User: <用户id>` 头,id 从登录 cookie 里解出。
- 余额在 `/api/user/self` 的 `quota` 字段,单位 500000 quota = $1。
- session cookie 会过期,过期后从浏览器 DevTools → Network → `/wallet` 请求头里复制新的 `session` cookie。
