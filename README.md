# Karma Chat — a real transformer AI, online

A from-scratch AI chatbot. **The model lives on a server; clients are thin apps that talk to its API.** No model weights are public.

- **Karma** = the AI (trained from scratch, GPT-style decoder-only transformer)
- Server: FastAPI (`server/`) — the model + live web tools (calculator, clock, dice, Wikipedia, DuckDuckGo) + chat UI
- Clients: this website (API client), the Android APK (thin client, ~42 KB), or any app that calls the API
- Model privacy: weights are kept on the server / in a private Hugging Face repo — never shipped to clients

## Try it

1. Deploy the server (see `server/README.md` — one free Hugging Face Space)
2. Open https://samratbarman1013-commits.github.io/tinygpt-chat/ and enter your server address (⋮ → Server URL), or install the APK (see releases) and do the same
3. Chat — needs internet, everything runs on your server

## The API

```
GET  /health   → {"ok": true, "params": ..., "tools": [...]}
POST /chat     → {"reply": "...", "source": "model|wikipedia|duckduckgo|calculator|clock|dice"}
```

CORS is open — any website or app can use it. Full docs and deployment guide: [`server/README.md`](server/README.md).

## History

Started as an experiment to train a real transformer from scratch and run it fully in the browser: 1M → 3.25M → 6M parameters, then scaled to 100M (chunked training on GitHub Actions). Now architecture is client–server so the model stays private.

- v1.x — on-device models (1M / 3.25M / 6M), offline APK + PWA (older releases removed)
- v2.0 — 100M-parameter model (private; deployed on the server only)
- v3.0 — the API update: thin-client APK + website, model no longer shipped
