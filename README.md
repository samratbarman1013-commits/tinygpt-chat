# TinyGPT Chat — a real 1M-parameter transformer chatbot

A complete, from-scratch AI chatbot that runs **100% on your device** — no server, no API keys, no internet needed after the first load.

- **983,488 parameters** (0.98M) decoder-only transformer, GPT-style architecture
- Trained from scratch (random init) on the DailyDialog dataset of real human conversations
- Runs in the browser via a **pure-JavaScript inference engine** (`engine.js`) — no ONNX, no WebAssembly, no ML libraries
- Ships as an installable PWA (works offline) and as a native Android APK (WebView wrapper, model bundled inside)

## Architecture

| Component | Value |
|---|---|
| Parameters | 983,488 (~1M) |
| Layers | 2 transformer blocks |
| Model dimension | 192 |
| Attention heads | 6 (head dim 32) |
| Feed-forward | 800 |
| Context window | 256 tokens |
| Tokenizer | character-level (vocab 102) |
| Output head | tied with input embeddings |
| Positional encoding | learned |

Training: AdamW, lr 1e-3 with warmup + cosine decay, batch 16 × 256 tokens, 3000 steps on ~10,700 DailyDialog multi-turn conversations (~5.6M characters).

## How it works

1. `train.py` — trains the transformer in PyTorch and exports weights to a flat `model.bin` (float32) + `config.json`
2. `engine.js` — a dependency-free JS class implementing the exact same forward pass (embeddings → 2× [LayerNorm → causal self-attention → residual → LayerNorm → GELU MLP → residual] → LayerNorm → tied output head) with a KV cache, temperature and top-k sampling
3. `index.html` — chat UI, loads `model.bin` and runs inference entirely client-side

## Run it

Open the GitHub Pages site: model downloads once (≈4 MB), then everything runs offline.

Or install the Android APK — the model is bundled inside the app.

## Train it yourself

Everything needed to reproduce the model is in this repo:

```bash
pip install torch
mkdir -p data && curl -sL -o train.zip "https://huggingface.co/datasets/roskoN/dailydialog/resolve/main/train.zip" && unzip train.zip -d data
python train.py
```

The GitHub Actions workflow `.github/workflows/build-model.yml` does exactly this on GitHub's servers on demand.

## Honest expectations

1M parameters is ~1/200,000th of GPT-4-class models. TinyGPT speaks simple, sometimes odd English — it was trained on one CPU for about an hour. It demonstrates the full transformer stack end-to-end, not chat quality.
