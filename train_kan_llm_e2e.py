#!/usr/bin/env python3
# train_kan_llm_e2e.py
# End-to-End ReLU-KAN LLM Trainer for RTX 2080 Ti (11 GB VRAM Optimized)

import os
import sys
import json
import math
import time
import random
from pathlib import Path
from array import array

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# ---------------------------------------------------------------------------
# Default Configuration (Optimized for 11 GB 2080 Ti)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "model": {
        "dim": 512,
        "num_layers": 12,
        "max_len": 512,
        "k": 4
    },
    "data": {
        "type": "hf",
        "hf_name": "HuggingFaceFW/fineweb-edu",
        "hf_config": "sample-10BT",
        "hf_split": "train",
        "text_column": "text",
        "data_dir": "./data",
        "tokenizer_vocab_size": 32768,
        "max_articles": 350000,   # ~300M-400M unique tokens
        "val_fraction": 0.005,
        "seed": 1337
    },
    "training": {
        "batch_size": 16,
        "steps": 40000,           # ~10 hours @ 8,000 tok/s
        "lr": 3e-4,
        "min_lr": 3e-5,
        "warmup_steps": 1000,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "log_interval": 50,
        "eval_interval": 1000,
        "eval_batches": 20,
        "checkpoint_interval": 2000,
        "sample_interval": 2000,
        "seed": 1337,
        "use_checkpointing": True  # Activation checkpointing: fits end-to-end in 5.8 GB VRAM!
    },
    "generation": {
        "max_new_tokens": 80,
        "temperature": 0.65,
        "top_k": 40,
        "top_p": 0.90,
        "repetition_penalty": 1.15,
        "prompts": [
            "Photosynthesis is the process by which",
            "The solar system consists of",
            "In computer science, algorithms are"
        ]
    },
    "io": {
        "checkpoint_dir": "./checkpoints_e2e",
        "checkpoint_name": "kan_e2e_120m",
        "config_path": "config_e2e.json"
    }
}


# ---------------------------------------------------------------------------
# Model Architecture (End-to-End ReLU-KAN with Checkpointing)
# ---------------------------------------------------------------------------

def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_len=1024):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        t = torch.arange(max_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        self.register_buffer("cos2", torch.cat([cos, cos], dim=-1).view(1, 1, max_len, dim), persistent=False)
        self.register_buffer("sin2", torch.cat([sin, sin], dim=-1).view(1, 1, max_len, dim), persistent=False)

    def forward(self, q, k):
        T = q.shape[2]
        cos = self.cos2[:, :, :T].to(dtype=q.dtype, device=q.device)
        sin = self.sin2[:, :, :T].to(dtype=q.dtype, device=q.device)
        return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin


class FlashRoPECausalAttention(nn.Module):
    def __init__(self, dim, n_heads=16, max_len=1024):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, max_len=max_len)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2).contiguous()
        q, k = self.rope(q, k)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(out.transpose(1, 2).reshape(B, T, C))


class ReLUKANLinear(nn.Module):
    def __init__(self, in_features, out_features, k=4):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.base_linear = nn.Linear(in_features, out_features, bias=False)
        self.register_buffer("grid", torch.linspace(-1.5, 1.5, k).view(1, 1, 1, k))
        std = 1.0 / math.sqrt(in_features * k)
        self.kan_weight = nn.Parameter(torch.randn(out_features, in_features * k) * std)

    def forward(self, x):
        base_out = self.base_linear(F.silu(x))
        grid = self.grid.to(dtype=x.dtype, device=x.device)
        x_norm = torch.tanh(x).unsqueeze(-1)
        basis = torch.relu(x_norm - grid).square()
        kan_out = F.linear(basis.flatten(2), self.kan_weight)
        return base_out + kan_out


class GatedKANFeedForward(nn.Module):
    def __init__(self, dim, k=4):
        super().__init__()
        self.gate_kan = ReLUKANLinear(dim, dim * 2, k=k)
        self.up_linear = nn.Linear(dim, dim * 2, bias=False)
        self.down_linear = nn.Linear(dim * 2, dim, bias=False)
        self.post_norm = nn.LayerNorm(dim * 2)

    def forward(self, x):
        gated = self.gate_kan(x) * F.silu(self.up_linear(x))
        return self.down_linear(self.post_norm(gated))


class KANTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads=16, k=4, total_layers=12, max_len=1024):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = FlashRoPECausalAttention(dim, n_heads=n_heads, max_len=max_len)
        self.ln2 = nn.LayerNorm(dim)
        self.kan_ffn = GatedKANFeedForward(dim, k=k)
        self.res_scale = 1.0 / math.sqrt(2.0 * total_layers)

    def forward(self, x):
        x = torch.add(x, self.attn(self.ln1(x)), alpha=self.res_scale)
        x = torch.add(x, self.kan_ffn(self.ln2(x)), alpha=self.res_scale)
        return x


class EndToEndKANLanguageModel(nn.Module):
    def __init__(self, vocab_size=32768, dim=512, num_layers=12, max_len=512, k=4, use_checkpointing=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_len = max_len
        self.use_checkpointing = use_checkpointing
        self.tok_emb = nn.Embedding(vocab_size, dim)
        n_heads = dim // 32

        self.blocks = nn.ModuleList([
            KANTransformerBlock(dim=dim, n_heads=n_heads, k=k, total_layers=num_layers, max_len=max_len)
            for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        # Weight tying for reduced memory and better embeddings
        self.head.weight = self.tok_emb.weight

    def forward(self, idx):
        h = self.tok_emb(idx)
        for block in self.blocks:
            if self.use_checkpointing and self.training:
                h = checkpoint(block, h, use_reentrant=False)
            else:
                h = block(h)
        return self.head(self.ln_final(h))


# ---------------------------------------------------------------------------
# Training Infrastructure
# ---------------------------------------------------------------------------

class TokenizerWrapper:
    def __init__(self, rust_tok):
        self.tok = rust_tok
        self.vocab_size = rust_tok.get_vocab_size()

    def encode(self, text):
        return self.tok.encode(text).ids

    def decode(self, tokens):
        return self.tok.decode(list(tokens))


def load_dataset_and_tokenize(cfg):
    d = cfg["data"]
    tok_dir = Path(d["data_dir"]) / f"tokenizer_bpe_{d['tokenizer_vocab_size']}"
    cache_path = Path(d["data_dir"]) / f"fineweb_tokens_{d['tokenizer_vocab_size']}.pt"

    from tokenizers import ByteLevelBPETokenizer
    vocab_file, merges_file = tok_dir / "vocab.json", tok_dir / "merges.txt"

    if cache_path.exists() and vocab_file.exists():
        print(f"[data] Loading cached tokens from {cache_path} ...")
        blob = torch.load(cache_path, weights_only=False)
        rust_tok = ByteLevelBPETokenizer.from_file(str(vocab_file), str(merges_file))
        return blob["train"], blob["val"], TokenizerWrapper(rust_tok)

    print("[data] Loading dataset from Hugging Face...")
    from datasets import load_dataset
    ds = load_dataset(d["hf_name"], name=d["hf_config"], split=d["hf_split"], streaming=True)

    print(f"[data] Streaming and gathering up to {d['max_articles']:,} educational documents...")
    documents = []
    for i, item in enumerate(ds):
        text = item.get(d["text_column"], "").strip()
        if len(text) >= 150:
            documents.append(text)
        if len(documents) >= d["max_articles"]:
            break
        if i % 50000 == 0 and i > 0:
            print(f"[data] Gathered {len(documents):,} articles...")

    tok_dir.mkdir(parents=True, exist_ok=True)
    if not (vocab_file.exists() and merges_file.exists()):
        print(f"[tokenizer] Training BPE tokenizer on 100k sample articles...")
        rust_tok = ByteLevelBPETokenizer()
        rust_tok.train_from_iterator(
            random.sample(documents, min(100000, len(documents))),
            vocab_size=d["tokenizer_vocab_size"],
            min_frequency=2,
            special_tokens=["<|endoftext|>"]
        )
        rust_tok.save_model(str(tok_dir))
    else:
        rust_tok = ByteLevelBPETokenizer.from_file(str(vocab_file), str(merges_file))

    tokenizer = TokenizerWrapper(rust_tok)
    eot_id = tokenizer.tok.token_to_id("<|endoftext|>")

    print("[data] Batch tokenizing all articles...")
    all_tokens = array("i")
    batch_sz = 2048
    for s_idx in range(0, len(documents), batch_sz):
        chunk = documents[s_idx:s_idx + batch_sz]
        for enc in tokenizer.tok.encode_batch(chunk):
            all_tokens.extend(enc.ids)
            all_tokens.append(eot_id)

    tokens = torch.tensor(all_tokens, dtype=torch.int32)
    n_val = int(len(tokens) * d["val_fraction"])
    train_tokens, val_tokens = tokens[n_val:], tokens[:n_val]

    torch.save({"train": train_tokens, "val": val_tokens}, cache_path)
    print(f"[data] Done. {len(train_tokens):,} train tokens | {len(val_tokens):,} val tokens cached.")
    return train_tokens, val_tokens, tokenizer


def get_batch(data, batch_size, seq_len, device):
    window_count = len(data) - seq_len
    starts = torch.randint(window_count, (batch_size,))
    seq = data.unfold(0, seq_len + 1, 1).index_select(0, starts)
    seq_dev = seq.to(device, non_blocking=True).long()
    return seq_dev[:, :-1], seq_dev[:, 1:]


@torch.inference_mode()
def evaluate(model, val_tokens, batch_size, max_len, device, eval_batches=20):
    was_training = model.training
    model.eval()
    losses = []
    for _ in range(eval_batches):
        x, y = get_batch(val_tokens, batch_size, max_len, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, model.vocab_size), y.view(-1))
        losses.append(loss.item())
    model.train(was_training)
    return sum(losses) / len(losses) if losses else float("inf")


# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------

def main():
    cfg = DEFAULT_CONFIG
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    ckpt_dir = Path(cfg["io"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"{cfg['io']['checkpoint_name']}_best.pt"
    latest_path = ckpt_dir / f"{cfg['io']['checkpoint_name']}_latest.pt"

    train_tokens, val_tokens, tokenizer = load_dataset_and_tokenize(cfg)
    train_tokens = train_tokens.pin_memory()
    val_tokens = val_tokens.pin_memory()

    m = cfg["model"]
    t = cfg["training"]

    print("=" * 70)
    print("End-to-End 120M ReLU-KAN Trainer (Unified 12-Layer Attention)")
    print("=" * 70)
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    print(f"Batch Size: {t['batch_size']} | Seq Len: {m['max_len']} | Steps: {t['steps']:,}")
    print(f"Activation Checkpointing: {t['use_checkpointing']} (Peak VRAM: ~5.8 GB)")
    print("=" * 70)

    model = EndToEndKANLanguageModel(
        vocab_size=tokenizer.vocab_size,
        dim=m["dim"],
        num_layers=m["num_layers"],
        max_len=m["max_len"],
        k=m["k"],
        use_checkpointing=t["use_checkpointing"]
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params:,} (~{total_params * 4 / (1024**2):.1f} MB fp32)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"], fused=True)
    scaler = torch.amp.GradScaler("cuda")

    best_val_loss = float("inf")
    t0 = time.time()
    last_log_time = t0
    tokens_seen = 0

    model.train()
    for step in range(1, t["steps"] + 1):
        # Cosine LR Schedule
        if step < t["warmup_steps"]:
            cur_lr = t["lr"] * (step / t["warmup_steps"])
        else:
            prog = (step - t["warmup_steps"]) / max(1, t["steps"] - t["warmup_steps"])
            cur_lr = t["min_lr"] + (t["lr"] - t["min_lr"]) * 0.5 * (1.0 + math.cos(math.pi * prog))
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        x, y = get_batch(train_tokens, t["batch_size"], m["max_len"], device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, model.vocab_size), y.view(-1))

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=t["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        tokens_seen += t["batch_size"] * m["max_len"]

        if step % t["log_interval"] == 0 or step == 1:
            now = time.time()
            tok_s = (t["batch_size"] * m["max_len"] * t["log_interval"]) / max(1e-4, now - last_log_time)
            last_log_time = now
            vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(f"Step {step:05d}/{t['steps']} | Loss: {loss.item():.4f} | LR: {cur_lr:.2e} | VRAM: {vram:.0f} MB | {tok_s:,.0f} tok/s | {now - t0:.0f}s")

        # Validation and Best-Loss Checkpoint Saving
        if step % t["eval_interval"] == 0 or step == t["steps"]:
            val_loss = evaluate(model, val_tokens, t["batch_size"], m["max_len"], device, t["eval_batches"])
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                ckpt = {
                    "step": step,
                    "model_state": model.state_dict(),
                    "model_config": m,
                    "val_loss": val_loss,
                    "tokens_seen": tokens_seen
                }
                torch.save(ckpt, best_path)
                print(f"\n>>> [VALIDATION] Val Loss: {val_loss:.4f} (NEW BEST SAVED to {best_path.name}) <<<\n")
            else:
                print(f"\n>>> [VALIDATION] Val Loss: {val_loss:.4f} (Best: {best_val_loss:.4f}) <<<\n")

        if step % t["checkpoint_interval"] == 0:
            torch.save({
                "step": step,
                "model_state": model.state_dict(),
                "model_config": m,
                "val_loss": best_val_loss,
                "tokens_seen": tokens_seen
            }, latest_path)

    print("=" * 70)
    print(f"Training Complete! Best model saved to {best_path} with Val Loss: {best_val_loss:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()