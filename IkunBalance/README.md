# IkunBalance

爬取 [api.ikuncode.cc](https://api.ikuncode.cc) 的钱包余额。

## 用法

```bash
chmod +x check.sh
IKUN_ACCESS_TOKEN='Bearer 你的短期token' ./check.sh
```

也可以同时提供 session cookie；脚本会优先使用传入的 token：

```bash
IKUN_SESSION_COOKIE='session=你的cookie' \
IKUN_ACCESS_TOKEN='Bearer 你的短期token' \
./check.sh
```

如果 session cookie 仍能刷新短期 token，也可只设置
`IKUN_SESSION_COOKIE`。可选用 `IKUN_BASE_URL` 覆盖服务地址。

## 说明

- 新版后台使用短期 `Authorization: Bearer ...` token；旧的
  `New-Api-User` 请求头已不再需要。
- 脚本在没有显式 token 时，会尝试通过
  `POST /api/user/auth/refresh` 用 session cookie 换取 token。
- 余额仍在 `/api/user/self` 的 `quota` 字段；当前站点配置为
  `500000 quota = ¥1`。
- `/api/subscription/self` 返回订阅方案和订阅额度，不是钱包余额。
- Bearer token 与 session cookie 都是登录凭据，不要提交到 Git。
