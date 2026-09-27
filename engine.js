/* TinyGPT inference engine — pure JS, no dependencies.
 * Runs a 1M-parameter decoder-only transformer fully on-device.
 * Weight layout (one flat Float32Array) matches export in train.py:
 *   wte[V*D], wpe[BLOCK*D], per layer i:
 *     ln1_g[D], ln1_b[D], attn_w[3D*D], attn_b[3D], proj_w[D*D], proj_b[D],
 *     ln2_g[D], ln2_b[D], fc1_w[F*D], fc1_b[F], fc2_w[D*F], fc2_b[D],
 *   then lnf_g[D], lnf_b[D]
 * torch Linear weight is [out, in] -> W[o*in + j].
 */
class TinyGPT {
  constructor(config, weights) {
    this.cfg = config;
    this.w = weights;
    const c = config, D = c.d;
    this.stoi = {};
    c.itos.forEach((ch, i) => { this.stoi[ch] = i; });
    let off = 0;
    const take = (n) => { const o = off; off += n; return o; };
    this.o = {};
    this.o.wte = take(c.vocab * D);
    this.o.wpe = take(c.block * D);
    this.o.layers = [];
    for (let i = 0; i < c.n_layers; i++) {
      this.o.layers.push({
        ln1g: take(D), ln1b: take(D),
        attnW: take(3 * D * D), attnB: take(3 * D),
        projW: take(D * D), projB: take(D),
        ln2g: take(D), ln2b: take(D),
        fc1W: take(c.ffn * D), fc1B: take(c.ffn),
        fc2W: take(D * c.ffn), fc2B: take(D),
      });
    }
    this.o.lnfg = take(D); this.o.lnfb = take(D);
    this.reset();
  }

  reset() {
    const c = this.cfg;
    this.kc = new Float32Array(c.n_layers * c.block * c.d); // keys cache
    this.vc = new Float32Array(c.n_layers * c.block * c.d); // values cache
    this.n = 0; // tokens cached
  }

  encode(s) {
    const ids = [];
    for (const ch of s) {
      const id = this.stoi[ch];
      if (id === undefined) continue; // drop OOV chars
      ids.push(id);
    }
    return ids;
  }

  _layernorm(x, off, g, b, D) {
    const w = this.w;
    let mu = 0;
    for (let j = 0; j < D; j++) mu += x[j];
    mu /= D;
    let varr = 0;
    for (let j = 0; j < D; j++) { const d = x[j] - mu; varr += d * d; }
    varr /= D;
    const inv = 1 / Math.sqrt(varr + 1e-5);
    for (let j = 0; j < D; j++) x[j] = (x[j] - mu) * inv * w[g + j] + w[b + j];
  }

  _gelu(x) {
    return 0.5 * x * (1 + Math.tanh(0.7978845608 * (x + 0.044715 * x * x * x)));
  }

  /* Feed one token id; appends to KV cache; keeps hidden state in this.lastX. */
  _forwardToken(id) {
    const c = this.cfg, D = c.d, F = c.ffn, H = c.n_heads, hd = D / H, w = this.w, o = this.o;
    const T = this.n;
    if (T >= c.block) throw new Error("context overflow");
    const x = new Float32Array(D);
    for (let j = 0; j < D; j++) x[j] = w[o.wte + id * D + j] + w[o.wpe + T * D + j];

    const qkv = new Float32Array(3 * D);
    const q = new Float32Array(D), k = new Float32Array(D), v = new Float32Array(D);

    for (let li = 0; li < c.n_layers; li++) {
      const L = o.layers[li];
      // --- attention block ---
      const h1 = Float32Array.from(x);
      this._layernorm(h1, 0, L.ln1g, L.ln1b, D);
      for (let out = 0; out < 3 * D; out++) {
        let s = w[L.attnB + out];
        const base = L.attnW + out * D;
        for (let j = 0; j < D; j++) s += h1[j] * w[base + j];
        qkv[out] = s;
      }
      for (let j = 0; j < D; j++) { q[j] = qkv[j]; k[j] = qkv[D + j]; v[j] = qkv[2 * D + j]; }
      // append k,v to cache
      const kbase = li * c.block * D + T * D;
      for (let j = 0; j < D; j++) { this.kc[kbase + j] = k[j]; this.vc[kbase + j] = v[j]; }
      // causal attention over 0..T (inclusive of current)
      const att = new Float32Array(D);
      for (let h = 0; h < H; h++) {
        const scores = new Float32Array(T + 1);
        let maxs = -Infinity;
        for (let t = 0; t <= T; t++) {
          let s = 0;
          const kb = li * c.block * D + t * D + h * hd;
          for (let j = 0; j < hd; j++) s += q[h * hd + j] * this.kc[kb + j];
          s /= Math.sqrt(hd);
          scores[t] = s;
          if (s > maxs) maxs = s;
        }
        let sum = 0;
        for (let t = 0; t <= T; t++) { scores[t] = Math.exp(scores[t] - maxs); sum += scores[t]; }
        for (let t = 0; t <= T; t++) {
          const p = scores[t] / sum;
          const vb = li * c.block * D + t * D + h * hd;
          for (let j = 0; j < hd; j++) att[h * hd + j] += p * this.vc[vb + j];
        }
      }
      const proj = new Float32Array(D);
      for (let out2 = 0; out2 < D; out2++) {
        let s = w[L.projB + out2];
        const base = L.projW + out2 * D;
        for (let j = 0; j < D; j++) s += att[j] * w[base + j];
        proj[out2] = s;
      }
      for (let j = 0; j < D; j++) x[j] += proj[j];
      // --- MLP block ---
      const h2 = Float32Array.from(x);
      this._layernorm(h2, 0, L.ln2g, L.ln2b, D);
      const f1 = new Float32Array(F);
      for (let out3 = 0; out3 < F; out3++) {
        let s = w[L.fc1B + out3];
        const base = L.fc1W + out3 * D;
        for (let j = 0; j < D; j++) s += h2[j] * w[base + j];
        f1[out3] = this._gelu(s);
      }
      for (let out4 = 0; out4 < D; out4++) {
        let s = w[L.fc2B + out4];
        const base = L.fc2W + out4 * F;
        for (let j = 0; j < F; j++) s += f1[j] * w[base + j];
        x[out4] += s;
      }
    }
    // final norm, keep as hidden state for logits
    this._layernorm(x, 0, o.lnfg, o.lnfb, D);
    this.lastX = x;
    this.n = T + 1;
  }

  _logits(x) {
    const c = this.cfg, D = c.d, w = this.w;
    const logits = new Float32Array(c.vocab);
    for (let i = 0; i < c.vocab; i++) {
      let s = 0;
      const base = this.o.wte + i * D;
      for (let j = 0; j < D; j++) s += x[j] * w[base + j];
      logits[i] = s;
    }
    return logits;
  }

  _sampleFrom(logits, temp, topk) {
    const V = this.cfg.vocab;
    if (temp <= 0.01) {
      let best = 0;
      for (let i = 1; i < V; i++) if (logits[i] > logits[best]) best = i;
      return best;
    }
    const scaled = new Float32Array(V);
    let maxs = -Infinity;
    for (let i = 0; i < V; i++) { scaled[i] = logits[i] / temp; if (scaled[i] > maxs) maxs = scaled[i]; }
    let sum = 0;
    for (let i = 0; i < V; i++) { scaled[i] = Math.exp(scaled[i] - maxs); sum += scaled[i]; }
    for (let i = 0; i < V; i++) scaled[i] /= sum;
    if (topk > 0 && topk < V) {
      const idx = Array.from({ length: V }, (_, i) => i);
      idx.sort((a, b) => scaled[b] - scaled[a]);
      const keep = new Set(idx.slice(0, topk));
      let s2 = 0;
      for (let i = 0; i < V; i++) { if (!keep.has(i)) scaled[i] = 0; else s2 += scaled[i]; }
      for (let i = 0; i < V; i++) scaled[i] /= s2;
    }
    let r = Math.random(), acc = 0;
    for (let i = 0; i < V; i++) { acc += scaled[i]; if (r <= acc) return i; }
    return V - 1;
  }

  /* Generate a bot reply. history = [{role:'user'|'bot', text}]. */
  reply(history, opts = {}) {
    const maxNew = opts.maxNew || 160;
    const temp = opts.temp ?? 0.75;
    const topk = opts.topk ?? 40;
    const eosId = this.stoi[this.cfg.eos];
    const nlId = this.stoi["\n"];
    this.reset();
    let prompt = this.cfg.bos;
    for (const m of history) {
      prompt += (m.role === "user" ? "User: " : "Bot: ") + m.text.trim() + "\n";
    }
    prompt += "Bot:";
    const ids = this.encode(prompt);
    if (ids.length === 0) ids.push(0);
    // keep only the most recent context that fits in the block, leaving room to generate
    const budget = Math.max(16, this.cfg.block - maxNew - 2);
    const start = Math.max(0, ids.length - budget);
    for (let i = start; i < ids.length; i++) this._forwardToken(ids[i]);
    let out = "";
    for (let t = 0; t < maxNew; t++) {
      const logits = this._logits(this.lastX);
      const id = this._sampleFrom(logits, temp, topk);
      const ch = this.cfg.itos[id];
      if (id === eosId) break;
      if (ch === "\n") break; // reply ends at end of Bot line
      out += ch;
      if (this.n + 1 >= this.cfg.block) break;
      this._forwardToken(id);
      if (out.endsWith("User:")) { out = out.slice(0, -6); break; }
    }
    return out.trim();
  }
}

if (typeof module !== "undefined") module.exports = { TinyGPT };
