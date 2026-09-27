# TinyGPT Chat — server edition

A 983,488-parameter transformer chatbot trained from scratch on DailyDialog, running **on the server**.
The web page talks to the model over the internet (`POST /chat`) — nothing runs on the client device.

## Run locally

```bash
cp ../model.bin ../config.json .
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

Open http://localhost:7860

## Deploy

- **Hugging Face Spaces (free)**: create a Docker Space, upload this folder's files (with `model.bin` + `config.json`) — the README metadata header is included in the repo root release zip.
- **Anywhere with Docker**: `docker build -t tinygpt-server . && docker run -p 7860:7860 tinygpt-server`

## API

```bash
curl -X POST http://localhost:7860/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","text":"hello, how are you?"}]}'
```

Files: `app.py` (FastAPI server), `model.py` (transformer), `static/index.html` (chat client), `Dockerfile`.
