# SuperScript/SubScript Generator

Pure front-end tool for turning text into superscript or subscript characters. No backend required; everything runs in the browser.

## Features

- Instant, offline conversion in the browser (no network calls)
- Superscript and subscript modes
- Clean UI with copy-to-clipboard
- Launch via simple static server with configurable host/port

## Run Locally

1. Start the static server (defaults to `127.0.0.1:8000`):
```bash
./serve.sh
```

2. Use custom host/port if needed:
```bash
./serve.sh 0.0.0.0 3000
./serve.sh 192.168.1.50 8080
```

3. Open your browser at `http://<host>:<port>` and start converting text.

> The script relies on `python3 -m http.server`, which is available on most systems by default. If `python3` is missing, install it or serve `index.html` with any static server of your choice.

## Supported Characters

### Superscript
- Numbers: 0-9
- Lowercase: a-z (most letters)
- Uppercase: A-Z (limited support)
- Symbols: +, -, =, (, )

### Subscript
- Numbers: 0-9
- Lowercase: a, e, h, i, j, k, l, m, n, o, p, r, s, t, u, v, x
- Symbols: +, -, =, (, )

> Uppercase letters without a dedicated subscript/superscript glyph will fall back to the lowercase equivalent when available (e.g., M → ₘ, H → ₕ).
