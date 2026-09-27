---
title: TinyGPT Chat
emoji: 🤖
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Real transformer chatbot + live web tools, running server-side
---

# TinyGPT Chat — server edition with live tools

A transformer chatbot trained from scratch on DailyDialog + UltraChat, running **on the server**,
with a real tool layer: factual questions get answered from **live web sources**
(Wikipedia / DuckDuckGo), plus calculator, clock and dice tools.
The web page talks to the model over the internet (`POST /chat`) — nothing heavy runs on the client.

## Tools (server-side, rule-routed)

| Ask something like...            | Answered by            |
|----------------------------------|------------------------|
| "what is 23*7+5"                 | 🧮 calculator (exact)   |
| "what time is it" / "aaj koto baje" | 🕰 clock (IST)       |
| "who is Virat Kohli"              | 🌐 Wikipedia (live)     |
| "tell me about the Taj Mahal"     | 🌐 Wikipedia (live)     |
| anything else                     | 🤖 the model (chat)     |

## Run locally

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

Open http://localhost:7860

## API

```bash
curl -X POST http://localhost:7860/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","text":"who is Albert Einstein"}]}'
# {"reply":"Albert Einstein was ... (source: Wikipedia)","source":"wikipedia"}
```

Files: `app.py` (FastAPI server), `tools.py` (tool layer), `model.py` (transformer),
`model.bin` (weights, float32 flat), `config.json` (architecture + vocab),
`static/index.html` (chat client), `Dockerfile`.
