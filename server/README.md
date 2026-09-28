# Karma Chat — Server & API

The Karma AI model runs **here on the server**. Clients (the website, the Android APK, or any API user) only send chat requests over the internet — no model is ever shipped to them.

## API

CORS is enabled — any website or app can call this API.

### `GET /health`
```json
{"ok": true, "params": 6083328, "tools": ["calculator", "clock", "dice", "wikipedia", "duckduckgo"]}
```

### `POST /chat`
```json
{
  "messages": [
    {"role": "user", "text": "who is Virat Kohli"},
    {"role": "bot",  "text": "..."},
    {"role": "user", "text": "and what is 23*7?"}
  ],
  "temp": 0.75,
  "topk": 40,
  "max_new": 160
}
```
Response:
```json
{"reply": "161", "source": "calculator"}
```

`source` tells you where the answer came from: `model` (the Karma transformer), `wikipedia`, `duckduckgo`, `calculator`, `clock`, or `dice`.

Quick test:
```bash
curl https://YOUR-SERVER/health
curl -X POST https://YOUR-SERVER/chat -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","text":"what time is it"}]}'
```

## Deploy on a Hugging Face Space (free, ~16 GB RAM)

1. Create a free account at https://huggingface.co
2. **New Space** → name it (e.g. `karma-chat`) → SDK: **Docker** → Space hardware: free CPU
3. Upload all files from this package (`app.py`, `tools.py`, `model.py`, `Dockerfile`, `requirements.txt`, `static/`) — "Add file → Upload files