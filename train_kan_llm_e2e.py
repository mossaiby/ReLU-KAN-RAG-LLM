#!/usr/bin/env python3
# train_kan_llm_e2e.py
# 124M End-to-End ReLU-KAN LLM Trainer (1024 Context, FineWeb-Edu, 11 GB VRAM Optimized)

import os
import sys
import json
import math
import time
import random
import argparse
import copy
from pathlib import Path
from array import array

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# ---------------------------------------------------------------------------
# Default Configuration (24-Hour 124M Marathon on 11 GB 2080 Ti)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "model": {
        "dim": 640,
        "num_layers": 14,
        "max_len": 1024,
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
        "max_articles": 700000,    # ~650M-700M unique tokens
        "val_fraction": 0.005,
        "seed": 1337
    },
    "training": {
        "batch_size": 8,           # Micro-batch size
        "grad_accum_steps": 2,     # Effective batch size = 16 (16,384 tokens/step)
        "steps": 40000,            # 40k steps = ~655M tokens (~24.2 hours)
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
        "use_checkpointing": True  # Activation checkpointing: fits in ~4.8 GB VRAM
    },
    "generation": {
        "max_new_tokens": 90,
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
        "checkpoint_name": "kan_e2e_124m",
        "config_path": "config_e2e.json"
    }
}


# ---------------------------------------------------------------------------
# CLI Argument Parsing & Config Loader
# ---------------------------------------------------------------------------

def deep_merge(base, override):
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def parse_args():
    p = argparse.ArgumentParser(description="124M End-to-End ReLU-KAN LLM Trainer")
    p.add_argument("--config", type=str, default=None, help="Path to config_e2e.json")
    p.add_argument("--save-config", action="store_true", default=False, help="Save effective config back to JSON")
    p.add_argument("--fresh", action="store_true", default=False, help="Ignore existing checkpoints and start clean")
    p.add_argument("--dim", type=int, default=None)
    p.add_argument("--layers", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--min-lr", type=float, default=None)
    p.add_argument("--warmup-steps", type=int, default=None)
    p.add_argument("--max-articles", type=int, default=None)
    p.add_argument("--eval-interval", type=int, default=None)
    p.add_argument("--checkpoint-interval", type=int, default=None)
    p.add_argument("--sample-interval", type=int, default=None)
    p.add_argument("--checkpoint-dir", type=str, default=None)
    p.add_argument("--checkpoint-name", type=str, default=None)
    return p.parse_args()


def load_config():
    args = parse_args()
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg_path = Path(args.config or DEFAULT_CONFIG["io"]["config_path"])

    if cfg_path.exists():
        with open(cfg_path) as f:
            file_cfg = json.load(f)
        cfg = deep_merge(DEFAULT_CONFIG, file_cfg)
        config_existed = True
    else:
        config_existed = False

    cli_map = {
        "dim": ("model", "dim"),
        "layers": ("model", "num_layers"),
        "seq_len": ("model", "max_len"),
        "batch_size": ("training", "batch_size"),
        "grad_accum": ("training", "grad_accum_steps"),
        "steps": ("training", "steps"),
        "lr": ("training", "lr"),
        "min_lr": ("training", "min_lr"),
        "warmup_steps": ("training", "warmup_steps"),
        "max_articles": ("data", "max_articles"),
        "eval_interval": ("training", "eval_interval"),
        "checkpoint_interval": ("training", "checkpoint_interval"),
        "sample_interval": ("training", "sample_interval"),
        "checkpoint_dir": ("io", "checkpoint_dir"),
        "checkpoint_name": ("io", "checkpoint_name"),
    }

    ns = vars(args)
    for arg_k, (sec, key) in cli_map.items():
        v = ns.get(arg_k)
        if v is not None:
            cfg[sec][key] = v

    out_path = Path(cfg["io"]["config_path"])
    if not config_existed or args.save_config:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(cfg, f, indent=2)
        print(f"[config] Effective config saved to {out_path}")

    return cfg, args.fresh


# ---------------------------------------------------------------------------
# Model Architecture (End-to-End 124M ReLU-KAN)
# ---------------------------------------------------------------------------

def _rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_len=2048):
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
    def __init__(self, dim, n_heads=10, max_len=2048):
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
    def __init__(self, dim, n_heads=10, k=4, total_layers=14, max_len=2048):
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
    def __init__(self, vocab_size=32768, dim=640, num_layers=14, max_len=1024, k=4, use_checkpointing=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_len = max_len
        self.use_checkpointing = use_checkpointing
        self.tok_emb = nn.Embedding(vocab_size, dim)
        n_heads = dim // 64  # head_dim = 64 (optimal for Tensor Cores)

        self.blocks = nn.ModuleList([
            KANTransformerBlock(dim=dim, n_heads=n_heads, k=k, total_layers=num_layers, max_len=max_len * 2)
            for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        # Weight tying for reduced VRAM and improved embeddings
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
# Tokenizer & Data Loading
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
    cache_path = Path(d["data_dir"]) / f"fineweb_tokens_{d['tokenizer_vocab_size']}_len{d['max_articles']}.pt"

    from tokenizers import ByteLevelBPETokenizer
    vocab_file, merges_file = tok_dir / "vocab.json", tok_dir / "merges.txt"

    if cache_path.exists() and vocab_file.exists():
        print(f"[data] Loading cached tokens from {cache_path} ...")
        blob = torch.load(cache_path, weights_only=False)
        rust_tok = ByteLevelBPETokenizer.from_file(str(vocab_file), str(merges_file))
        return blob["train"], blob["val"], TokenizerWrapper(rust_tok)

    print("[data] Streaming dataset from Hugging Face...")
    from datasets import load_dataset
    ds = load_dataset(d["hf_name"], name=d["hf_config"], split=d["hf_split"], streaming=True)

    print(f"[data] Gathering up to {d['max_articles']:,} educational documents...")
    documents = []
    for i, item in enumerate(ds):
        text = item.get(d["text_column"], "").strip()
        if len(text) >= 150:
            documents.append(text)
        if len(documents) >= d["max_articles"]:
            break
        if i % 100000 == 0 and i > 0:
            print(f"[data]   Gathered {len(documents):,} articles...")

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
    batch_sz = 4096
    for s_idx in range(0, len(documents), batch_sz):
        chunk = documents[s_idx:s_idx + batch_sz]
        for enc in tokenizer.tok.encode_batch(chunk):
            all_tokens.extend(enc.ids)
            all_tokens.append(eot_id)
        if s_idx % 80000 == 0 and s_idx > 0:
            print(f"[data]   Tokenized {s_idx:,} / {len(documents):,} articles ({len(all_tokens):,} tokens)...")

    tokens = torch.tensor(all_tokens, dtype=torch.int32)
    n_val = int(len(tokens) * d["val_fraction"])
    train_tokens, val_tokens = tokens[n_val:], tokens[:n_val]

    torch.save({"train": train_tokens, "val": val_tokens}, cache_path)
    print(f"[data] Tokenization complete: {len(train_tokens):,} train | {len(val_tokens):,} val tokens cached.")
    return train_tokens, val_tokens, tokenizer


def get_batch(data, batch_size, seq_len, device):
    window_count = len(data) - seq_len
    starts = torch.randint(window_count, (batch_size,))
    seq = data.unfold(0, seq_len + 1, 1).index_select(0, starts)
    seq_dev = seq.to(device, non_blocking=True).long()
    # Explicit .contiguous() prevents any view stride errors
    return seq_dev[:, :-1].contiguous(), seq_dev[:, 1:].contiguous()


@torch.inference_mode()
def evaluate(model, val_tokens, batch_size, max_len, device, eval_batches=20):
    was_training = model.training
    model.eval()
    losses = []
    for _ in range(eval_batches):
        x, y = get_batch(val_tokens, batch_size, max_len, device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), y.reshape(-1))
        losses.append(loss.item())
    model.train(was_training)
    return sum(losses) / len(losses) if losses else float("inf")


@torch.inference_mode()
def generate_sample(model, tokenizer, prompt, max_new_tokens=90, temperature=0.65, top_k=40, repetition_penalty=1.15, device="cuda"):
    was_training = model.training
    model.eval()
    tokens = tokenizer.encode(prompt)
    if not tokens:
        tokens = [0]
    input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    eot_id = tokenizer.tok.token_to_id("<|endoftext|>")

    for _ in range(max_new_tokens):
        idx_cond = input_ids[:, -model.max_len:]
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(idx_cond)
        next_logits = logits[0, -1, :].clone().float()

        # Repetition penalty
        for prev_tok in set(input_ids[0].tolist()[-15:]):
            if next_logits[prev_tok] > 0:
                next_logits[prev_tok] /= repetition_penalty
            else:
                next_logits[prev_tok] *= repetition_penalty

        next_logits = next_logits / max(temperature, 1e-4)
        if top_k > 0:
            v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits[next_logits < v[-1]] = -float("Inf")

        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        token_id = next_token.item()

        if token_id == eot_id:
            break

        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

    model.train(was_training)
    return tokenizer.decode(input_ids[0].tolist()).strip()


# ---------------------------------------------------------------------------
# Main Training Loop
# ---------------------------------------------------------------------------

def main():
    cfg, fresh = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    m = cfg["model"]
    t = cfg["training"]

    ckpt_dir = Path(cfg["io"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"{cfg['io']['checkpoint_name']}_best.pt"
    latest_path = ckpt_dir / f"{cfg['io']['checkpoint_name']}_latest.pt"

    train_tokens, val_tokens, tokenizer = load_dataset_and_tokenize(cfg)
    train_tokens = train_tokens.pin_memory()
    val_tokens = val_tokens.pin_memory()

    grad_accum = t.get("grad_accum_steps", 1)
    effective_batch = t["batch_size"] * grad_accum
    tokens_per_step = effective_batch * m["max_len"]

    print("=" * 70)
    print("124M End-to-End ReLU-KAN Marathon Trainer (1024 Context)")
    print("=" * 70)
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    print(f"Context: {m['max_len']} | Micro-Batch: {t['batch_size']} | Grad Accum: {grad_accum} (Effective: {effective_batch})")
    print(f"Tokens/Step: {tokens_per_step:,} | Total Steps: {t['steps']:,} (~{tokens_per_step * t['steps'] / 1e6:.1f}M tokens)")
    print(f"Activation Checkpointing: {t['use_checkpointing']} (Peak VRAM: ~4.8 GB)")
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

    start_step = 1
    best_val_loss = float("inf")
    tokens_seen = 0

    # Resume from checkpoint if present
    if not fresh and latest_path.exists():
        print(f"\n[resume] Loading checkpoint from {latest_path} ...")
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scaler.load_state_dict(ckpt["scaler_state"])
        start_step = ckpt.get("step", 0) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        tokens_seen = ckpt.get("tokens_seen", 0)
        print(f"[resume] Resuming at Step {start_step} (Best Val Loss: {best_val_loss:.4f})\n")

    t0 = time.time()
    last_log_time = t0
    accum_loss = 0.0

    model.train()
    for step in range(start_step, t["steps"] + 1):
        # Cosine LR Schedule
        if step < t["warmup_steps"]:
            cur_lr = t["lr"] * (step / t["warmup_steps"])
        else:
            prog = (step - t["warmup_steps"]) / max(1, t["steps"] - t["warmup_steps"])
            cur_lr = t["min_lr"] + (t["lr"] - t["min_lr"]) * 0.5 * (1.0 + math.cos(math.pi * prog))
        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0

        # Gradient accumulation loop
        for _ in range(grad_accum):
            x, y = get_batch(train_tokens, t["batch_size"], m["max_len"], device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), y.reshape(-1)) / grad_accum
            scaler.scale(loss).backward()
            step_loss += loss.item() * grad_accum

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=t["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        tokens_seen += tokens_per_step
        accum_loss += step_loss

        # Logging
        if step % t["log_interval"] == 0 or step == start_step:
            now = time.time()
            interval_tokens = tokens_per_step * (t["log_interval"] if step > start_step else 1)
            tok_s = interval_tokens / max(1e-4, now - last_log_time)
            last_log_time = now
            avg_loss = accum_loss / (t["log_interval"] if step > start_step else 1)
            accum_loss = 0.0
            vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(f"Step {step:05d}/{t['steps']} | Loss: {avg_loss:.4f} | LR: {cur_lr:.2e} | VRAM: {vram:.0f} MB | {tok_s:,.0f} tok/s | {now - t0:.0f}s")

        # Periodic Validation
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

        # Periodic Text Generation Samples
        if step % t["sample_interval"] == 0:
            print("\n" + "-" * 55)
            print(f"--- Generation Samples at Step {step} ---")
            for p in cfg["generation"]["prompts"]:
                sample_out = generate_sample(
                    model, tokenizer, p,
                    max_new_tokens=cfg["generation"]["max_new_tokens"],
                    temperature=cfg["generation"]["temperature"],
                    top_k=cfg["generation"]["top_k"],
                    repetition_penalty=cfg["generation"]["repetition_penalty"],
                    device=device
                )
                print(f">>> {sample_out}\n")
            print("-" * 55 + "\n")

        # Periodic Latest Checkpoint
        if step % t["checkpoint_interval"] == 0 or step == t["steps"]:
            torch.save({
                "step": step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "model_config": m,
                "val_loss": best_val_loss,
                "tokens_seen": tokens_seen
            }, latest_path)

    print("=" * 70)
    print(f"Training Complete! Best model saved to {best_path} with Val Loss: {best_val_loss:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()