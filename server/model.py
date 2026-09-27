"""TinyGPT model definition — identical architecture to the trained checkpoint."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d, n_heads, ffn):
        super().__init__()
        self.n_heads = n_heads
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.Linear(d, 3 * d, bias=True)
        self.proj = nn.Linear(d, d, bias=True)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, ffn, bias=True)
        self.fc2 = nn.Linear(ffn, d, bias=True)

    def forward(self, x):
        B, T, C = x.shape
        n_heads, hd = self.n_heads, C // self.n_heads
        h = self.ln1(x)
        qkv = self.attn(h)
        q, k, v = qkv.split(C, dim=2)
        q = q.view(B, T, n_heads, hd).transpose(1, 2)
        k = k.view(B, T, n_heads, hd).transpose(1, 2)
        v = v.view(B, T, n_heads, hd).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.proj(y)
        h = self.ln2(x)
        x = x + self.fc2(F.gelu(self.fc1(h)))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab, d, n_layers, n_heads, ffn, block):
        super().__init__()
        self.wte = nn.Embedding(vocab, d)
        self.wpe = nn.Embedding(block, d)
        self.blocks = nn.ModuleList([Block(d, n_heads, ffn) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d)

    def forward(self, idx):
        B, T = idx.shape
        x = self.wte(idx) + self.wpe(torch.arange(T))
        for b in self.blocks:
            x = b(x)
        x = self.ln_f(x)
        return x @ self.wte.weight.T
