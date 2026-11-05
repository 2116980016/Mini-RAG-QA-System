# mini_gpt.py  ── 从零实现字符级 mini-GPT
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- 超参 ----------------
@dataclass
class Config:
    vocab_size: int = 0          # 由 tokenizer 决定
    block_size: int = 128        # 上下文长度
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------- 1. 字符级 tokenizer ----------------
class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(chars)

    def encode(self, s: str):
        return [self.stoi[c] for c in s if c in self.stoi]

    def decode(self, ids):
        return "".join(self.itos[int(i)] for i in ids)


# ---------------- 2. 单头自注意力（带 causal mask）----------------
class Head(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        head_dim = cfg.n_embd // cfg.n_head
        self.key   = nn.Linear(cfg.n_embd, head_dim, bias=False)
        self.query = nn.Linear(cfg.n_embd, head_dim, bias=False)
        self.value = nn.Linear(cfg.n_embd, head_dim, bias=False)
        # 下三角 mask 作为 buffer，不参与梯度
        self.register_buffer(
            "tril", torch.tril(torch.ones(cfg.block_size, cfg.block_size))
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)    # (B,T,hd)
        q = self.query(x)
        v = self.value(x)

        # 缩放点积注意力
        wei = q @ k.transpose(-2, -1) / math.sqrt(k.size(-1))  # (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        return wei @ v     # (B,T,hd)


# ---------------- 3. 多头注意力 ----------------
class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.heads = nn.ModuleList([Head(cfg) for _ in range(cfg.n_head)])
        self.proj  = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


# ---------------- 4. FFN ----------------
class FeedForward(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


# ---------------- 5. Transformer Block（Pre-LN）----------------
class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.n_embd)
        self.attn = MultiHeadAttention(cfg)
        self.ln2  = nn.LayerNorm(cfg.n_embd)
        self.ffn  = FeedForward(cfg)

    def forward(self, x):
        # Pre-LN：先归一化再进子层，残差保持恒等路径
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ---------------- 6. Mini-GPT ----------------
class MiniGPT(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop    = nn.Dropout(cfg.dropout)
        self.blocks  = nn.Sequential(*[Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f    = nn.LayerNorm(cfg.n_embd)
        self.head    = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # weight tying：输入 embedding 与输出投影共享权重
        self.head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size

        tok = self.tok_emb(idx)                                 # (B,T,C)
        pos = self.pos_emb(torch.arange(T, device=idx.device))  # (T,C)
        x = self.drop(tok + pos)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)                                   # (B,T,vocab)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        return logits, loss

    # ---------- 采样生成：temperature / top-k / top-p ----------
    @torch.no_grad()
    def generate(self, idx, max_new_tokens,
                 temperature=1.0, top_k=None, top_p=None):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-8)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = F.softmax(sorted_logits, dim=-1)
                cumprobs = probs.cumsum(dim=-1)
                # 去掉“累积概率（不含自己）已超过 top_p”的 token
                mask = cumprobs - probs > top_p
                sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
                logits = torch.full_like(logits, float("-inf"))
                logits.scatter_(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=-1)
        return idx


# ---------------- 7. 数据 & 训练 ----------------
def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)


def train(text: str, steps: int = 15000, batch_size: int = 32, lr: float = 3e-4):
    tok = CharTokenizer(text)
    cfg = Config(vocab_size=tok.vocab_size)
    model = MiniGPT(cfg).to(cfg.device)

    data = torch.tensor(tok.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data, val_data = data[:n], data[n:]

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    model.train()
    for step in range(steps):
        xb, yb = get_batch(train_data, cfg.block_size, batch_size, cfg.device)
        _, loss = model(xb, yb)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % 200 == 0 or step == steps - 1:
            model.eval()
            with torch.no_grad():
                xv, yv = get_batch(val_data, cfg.block_size, batch_size, cfg.device)
                _, vloss = model(xv, yv)
            model.train()
            print(f"[step {step:4d}] train {loss.item():.3f} | val {vloss.item():.3f}")

    return model, tok, cfg


def sample(model, tok, cfg, prompt="", n=200, **kwargs):
    model.eval()
    idx = torch.tensor([tok.encode(prompt) or [0]], dtype=torch.long, device=cfg.device)
    out = model.generate(idx, max_new_tokens=n, **kwargs)
    return tok.decode(out[0].tolist())


# ---------------- 8. Demo ----------------
if __name__ == "__main__":
    # 换成你自己的语料；中文语料建议先在字表里加标点/换行
    with open("corpus.txt", "r", encoding="utf-8") as f:
        text = f.read()

    model, tok, cfg = train(text, steps=15000)

    print("\n--- greedy (T=0.01) ---")
    print(sample(model, tok, cfg, prompt="贾宝玉", n=150, temperature=0.01))

    print("\n--- T=0.8, top_k=40 ---")
    print(sample(model, tok, cfg, prompt="贾宝玉", n=150, temperature=0.8, top_k=40))

    print("\n--- T=1.0, top_p=0.9 ---")
    print(sample(model, tok, cfg, prompt="贾宝玉", n=150, temperature=1.0, top_p=0.9))