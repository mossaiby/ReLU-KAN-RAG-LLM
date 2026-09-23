#!/usr/bin/env python3
# train_kan_llm_e2e.py
# 124M End-to-End ReLU-KAN LLM Trainer (Power-Cut Protected & Atomic Checkpointing)

import os
import sys
import json
import math
import time
import random
import argparse
import copy
import gc
from pathlib import Path
from array import array

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

# ---------------------------------------------------------------------------
# Default Configuration (Batch-32 Large Scale Continual Training)
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
        "max_articles": 2500000,
        "val_fraction": 0.005,
        "seed": 1337
    },
    "training": {
        "batch_size": 16,          # Fast 16-microbatch mode (~14.5k tok/s)
        "grad_accum_steps": 2,     # Effective batch = 32 (32,768 tokens/step)
        "steps": 100000,           # Full Chinchilla target (~2.57B tokens)
        "lr": 3e-4,
        "min_lr": 3e-5,
        "continuation_lr": 1.8e-4,
        "warmup_steps": 1000,
        "weight_decay": 1e-4,
        "grad_clip": 1.0,
        "log_interval": 50,
        "eval_interval": 1000,
        "eval_batches": 20,
        "checkpoint_interval": 2000,
        "sample_interval": 2000,
        "seed": 1337,
        "use_checkpointing": True
    },
    "generation": {
        "max_new_tokens": 90,
        "temperature": 0.65,
        "top_k": 40,
        "top_p": 0.90,
        "repetition_penalty": 1.20,
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
# Atomic Torch Saver (Immune to Sudden Power Outages)
# ---------------------------------------------------------------------------

def atomic_torch_save(obj, target_path, retries=5):
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_name(f"{target_path.name}.tmp")

    if tmp_path.exists():
        try:
            tmp_path.unlink()
        except OSError:
            pass

    # Save to temporary file first
    torch.save(obj, tmp_path)

    # Force OS flush to physical disk sectors
    try:
        with open(tmp_path, "rb+") as f:
            os.fsync(f.fileno())
    except OSError:
        pass

    # Atomic rename/replace
    for attempt in range(retries):
        try:
            os.replace(tmp_path, target_path)
            return
        except OSError:
            time.sleep(0.2 * (attempt + 1))

    try:
        os.replace(tmp_path, target_path)
    except Exception as e:
        print(f"[warning] Could not atomically rename {tmp_path} to {target_path}: {e}")


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
    p.add_argument("--resume-from", type=str, default=None, help="Path to specific checkpoint to resume from")
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

    return cfg, args.fresh, args.resume_from


# ---------------------------------------------------------------------------
# Model Architecture (Hardened ReLU-KAN)
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
        self.post_norm = nn.LayerNorm(dim * 2, eps=1e-5)

    def forward(self, x):
        g1 = self.gate_kan(x)
        g2 = F.silu(self.up_linear(x))
        gated = self.post_norm(g1.float() * g2.float())
        return self.down_linear(gated.to(dtype=x.dtype))


class KANTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads=10, k=4, total_layers=14, max_len=2048):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim, eps=1e-5)
        self.attn = FlashRoPECausalAttention(dim, n_heads=n_heads, max_len=max_len)
        self.ln2 = nn.LayerNorm(dim, eps=1e-5)
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
        n_heads = dim // 64

        self.blocks = nn.ModuleList([
            KANTransformerBlock(dim=dim, n_heads=n_heads, k=k, total_layers=num_layers, max_len=max_len * 2)
            for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(dim, eps=1e-5)
        self.head = nn.Linear(dim, vocab_size, bias=False)

        # Weight tying
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
    train_bin = Path(d["data_dir"]) / f"fineweb_train_{d['tokenizer_vocab_size']}_len{d['max_articles']}.bin"
    val_bin = Path(d["data_dir"]) / f"fineweb_val_{d['tokenizer_vocab_size']}_len{d['max_articles']}.bin"

    from tokenizers import ByteLevelBPETokenizer
    vocab_file, merges_file = tok_dir / "vocab.json", tok_dir / "merges.txt"

    if train_bin.exists() and val_bin.exists() and vocab_file.exists():
        print(f"[data] Found cached binary tokens on disk:")
        print(f"       Train: {train_bin}")
        print(f"       Val  : {val_bin}")
        rust_tok = ByteLevelBPETokenizer.from_file(str(vocab_file), str(merges_file))
        tokenizer = TokenizerWrapper(rust_tok)
        train_mmap = np.memmap(train_bin, dtype=np.int32, mode="r")
        val_mmap = np.memmap(val_bin, dtype=np.int32, mode="r")
        print(f"[data] Memory-mapped {len(train_mmap):,} train | {len(val_mmap):,} val tokens (RAM: ~0 MB).")
        return train_mmap, val_mmap, tokenizer

    from datasets import load_dataset
    print("[data] Streaming FineWeb-Edu dataset from Hugging Face...")
    ds_stream = load_dataset(d["hf_name"], name=d["hf_config"], split=d["hf_split"], streaming=True)

    tok_dir.mkdir(parents=True, exist_ok=True)
    if not (vocab_file.exists() and merges_file.exists()):
        print("[tokenizer] Gathering 50,000 sample articles to train BPE tokenizer...")
        sample_docs = []
        for item in ds_stream:
            text = item.get(d["text_column"], "").strip()
            if len(text) >= 150:
                sample_docs.append(text)
            if len(sample_docs) >= 50000:
                break
        print(f"[tokenizer] Training BPE tokenizer on {len(sample_docs):,} sample articles...")
        rust_tok = ByteLevelBPETokenizer()
        rust_tok.train_from_iterator(
            sample_docs,
            vocab_size=d["tokenizer_vocab_size"],
            min_frequency=2,
            special_tokens=["<|endoftext|>"]
        )
        rust_tok.save_model(str(tok_dir))
        del sample_docs
        gc.collect()
    else:
        print(f"[tokenizer] Loading existing tokenizer from {tok_dir} ...")
        rust_tok = ByteLevelBPETokenizer.from_file(str(vocab_file), str(merges_file))

    tokenizer = TokenizerWrapper(rust_tok)
    eot_id = tokenizer.tok.token_to_id("<|endoftext|>")

    print(f"[data] Streaming, tokenizing, and writing up to {d['max_articles']:,} articles directly to disk...")
    ds_stream = load_dataset(d["hf_name"], name=d["hf_config"], split=d["hf_split"], streaming=True)
    Path(d["data_dir"]).mkdir(parents=True, exist_ok=True)
    val_interval = int(1.0 / max(d["val_fraction"], 1e-5))

    chunk_articles = []
    articles_collected = 0
    total_train_tokens = 0
    total_val_tokens = 0
    t0 = time.time()

    with open(train_bin, "wb") as f_train, open(val_bin, "wb") as f_val:
        for item in ds_stream:
            text = item.get(d["text_column"], "").strip()
            if len(text) < 150:
                continue

            chunk_articles.append(text)
            articles_collected += 1

            if len(chunk_articles) >= 4000:
                encs = tokenizer.tok.encode_batch(chunk_articles)
                train_chunk = array("i")
                val_chunk = array("i")

                for doc_i, enc in enumerate(encs):
                    target_arr = val_chunk if (doc_i % val_interval == 0) else train_chunk
                    target_arr.extend(enc.ids)
                    target_arr.append(eot_id)

                if train_chunk:
                    f_train.write(train_chunk.tobytes())
                    total_train_tokens += len(train_chunk)
                if val_chunk:
                    f_val.write(val_chunk.tobytes())
                    total_val_tokens += len(val_chunk)

                f_train.flush()
                f_val.flush()
                chunk_articles = []

                if articles_collected % 50000 == 0 or articles_collected == d["max_articles"]:
                    rate = (total_train_tokens + total_val_tokens) / max(1e-4, time.time() - t0)
                    print(f"[data]   {articles_collected:,} / {d['max_articles']:,} articles written | "
                          f"{total_train_tokens + total_val_tokens:,} tokens ({rate:,.0f} tok/s) | RAM < 350 MB")

            if articles_collected >= d["max_articles"]:
                break

        if chunk_articles:
            encs = tokenizer.tok.encode_batch(chunk_articles)
            train_chunk = array("i")
            val_chunk = array("i")
            for doc_i, enc in enumerate(encs):
                target_arr = val_chunk if (doc_i % val_interval == 0) else train_chunk
                target_arr.extend(enc.ids)
                target_arr.append(eot_id)
            if train_chunk:
                f_train.write(train_chunk.tobytes())
                total_train_tokens += len(train_chunk)
            if val_chunk:
                f_val.write(val_chunk.tobytes())
                total_val_tokens += len(val_chunk)
            f_train.flush()
            f_val.flush()
            del chunk_articles
            gc.collect()

    print(f"[data] Done. Written to disk: {total_train_tokens:,} train | {total_val_tokens:,} val tokens.")
    train_mmap = np.memmap(train_bin, dtype=np.int32, mode="r")
    val_mmap = np.memmap(val_bin, dtype=np.int32, mode="r")
    return train_mmap, val_mmap, tokenizer


def get_batch(data, batch_size, seq_len, device):
    window_count = len(data) - seq_len
    starts = torch.randint(window_count, (batch_size,)).tolist()
    batch = np.empty((batch_size, seq_len + 1), dtype=np.int32)
    for i, s in enumerate(starts):
        batch[i] = data[s : s + seq_len + 1]
    seq_dev = torch.from_numpy(batch).to(device, non_blocking=True).long()
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
        if not (torch.isnan(loss) or torch.isinf(loss)):
            losses.append(loss.item())
    model.train(was_training)
    return sum(losses) / len(losses) if losses else float("inf")


@torch.inference_mode()
def generate_sample(model, tokenizer, prompt, max_new_tokens=90, temperature=0.65, top_k=40, repetition_penalty=1.20, device="cuda"):
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

        if torch.isnan(next_logits).any():
            break

        for prev_tok in set(input_ids[0].tolist()[-20:]):
            if next_logits[prev_tok] > 0:
                next_logits[prev_tok] /= repetition_penalty
            else:
                next_logits[prev_tok] *= repetition_penalty

        next_logits = next_logits / max(temperature, 1e-4)
        if top_k > 0:
            v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
            next_logits[next_logits < v[-1]] = -float("Inf")

        probs = F.softmax(next_logits, dim=-1)
        if torch.isnan(probs).any() or probs.sum() == 0:
            break

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
    cfg, fresh, resume_from = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    m = cfg["model"]
    t = cfg["training"]

    random.seed(t["seed"])
    torch.manual_seed(t["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(t["seed"])

    ckpt_dir = Path(cfg["io"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / f"{cfg['io']['checkpoint_name']}_best.pt"
    latest_path = ckpt_dir / f"{cfg['io']['checkpoint_name']}_latest.pt"

    train_tokens, val_tokens, tokenizer = load_dataset_and_tokenize(cfg)

    grad_accum = t.get("grad_accum_steps", 1)
    effective_batch = t["batch_size"] * grad_accum
    tokens_per_step = effective_batch * m["max_len"]

    print("=" * 70)
    print("124M End-to-End ReLU-KAN Trainer (Batch-32 Mode)")
    print("=" * 70)
    print(f"Device: {device} ({torch.cuda.get_device_name(0)})")
    print(f"Context: {m['max_len']} | Micro-Batch: {t['batch_size']} | Grad Accum: {grad_accum} (Effective: {effective_batch})")
    print(f"Tokens/Step: {tokens_per_step:,} | Target Steps: {t['steps']:,} (~{tokens_per_step * t['steps'] / 1e6:.1f}M tokens)")
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
    scaler = torch.amp.GradScaler("cuda", init_scale=32768.0)

    start_step = 1
    best_val_loss = float("inf")
    tokens_seen = 0

    target_resume = None
    if resume_from:
        target_resume = Path(resume_from)
    elif not fresh and latest_path.exists():
        target_resume = latest_path

    is_continuation_run = False
    continuation_start_step = 1

    if target_resume and target_resume.exists():
        print(f"\n[resume] Checking checkpoint: {target_resume} ...")
        try:
            ckpt = torch.load(target_resume, map_location=device, weights_only=False)
        except Exception as e:
            print(f"[resume] WARNING: Could not load '{target_resume.name}' ({e})! Power outage may have interrupted a write.")
            if best_path.exists() and target_resume != best_path:
                print(f"[resume] Recovering from healthy best checkpoint: {best_path} ...")
                ckpt = torch.load(best_path, map_location=device, weights_only=False)
            else:
                raise RuntimeError(f"Failed to load checkpoint and no valid backup found: {e}")

        # Check if checkpoint contains NaNs
        has_nan = any(torch.isnan(p).any() for p in ckpt["model_state"].values())
        if has_nan:
            print(f"[resume] WARNING: '{target_resume.name}' contains NaN weights!")
            if best_path.exists():
                print(f"[resume] Recovering from healthy best checkpoint: {best_path} ...")
                ckpt = torch.load(best_path, map_location=device, weights_only=False)
            else:
                raise RuntimeError("Checkpoint contains NaNs and no clean best.pt was found.")

        model.load_state_dict(ckpt["model_state"])
        if "optimizer_state" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            except Exception:
                print("[resume] Optimizer state skipped.")
        if "scaler_state" in ckpt:
            try:
                scaler.load_state_dict(ckpt["scaler_state"])
            except Exception:
                pass

        start_step = ckpt.get("step", 0) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        tokens_seen = ckpt.get("tokens_seen", 0)

        if "rng_state" in ckpt:
            torch.set_rng_state(ckpt["rng_state"].cpu())
        if "cuda_rng_state" in ckpt and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([s.cpu() for s in ckpt["cuda_rng_state"]])
        if "py_rng_state" in ckpt:
            random.setstate(ckpt["py_rng_state"])

        if start_step >= 39000:
            is_continuation_run = True
            continuation_start_step = start_step
            print(f"[resume] Continuing completed run from Step {start_step}. Activating smooth re-warming scheduler.")

        print(f"[resume] Successfully restored clean state. Resuming at Step {start_step} (Best Val Loss: {best_val_loss:.4f})\n")

    t0 = time.time()
    last_log_time = t0
    accum_loss = 0.0

    rewarm_steps = 500
    peak_continuation_lr = t.get("continuation_lr", 1.8e-4)

    model.train()
    for step in range(start_step, t["steps"] + 1):
        if is_continuation_run:
            step_offset = step - continuation_start_step
            if step_offset < rewarm_steps:
                cur_lr = t["min_lr"] + (peak_continuation_lr - t["min_lr"]) * (step_offset / rewarm_steps)
            else:
                rem_prog = (step_offset - rewarm_steps) / max(1, (t["steps"] - continuation_start_step - rewarm_steps))
                rem_prog = min(1.0, max(0.0, rem_prog))
                cur_lr = t["min_lr"] + (peak_continuation_lr - t["min_lr"]) * 0.5 * (1.0 + math.cos(math.pi * rem_prog))
        else:
            if step < t["warmup_steps"]:
                cur_lr = t["lr"] * (step / t["warmup_steps"])
            else:
                prog = (step - t["warmup_steps"]) / max(1, t["steps"] - t["warmup_steps"])
                cur_lr = t["min_lr"] + (t["lr"] - t["min_lr"]) * 0.5 * (1.0 + math.cos(math.pi * prog))

        for pg in optimizer.param_groups:
            pg["lr"] = cur_lr

        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        skip_step = False

        for _ in range(grad_accum):
            x, y = get_batch(train_tokens, t["batch_size"], m["max_len"], device)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
                loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), y.reshape(-1))

            if torch.isnan(loss) or torch.isinf(loss):
                skip_step = True
                break

            loss_scaled = loss / grad_accum
            scaler.scale(loss_scaled).backward()
            step_loss += loss.item() / grad_accum

        if skip_step:
            print(f"\n[shield] Non-finite loss detected at step {step}; skipping batch without updating weights.")
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            continue

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=t["grad_clip"])

        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            print(f"\n[shield] Non-finite grad norm ({grad_norm}) at step {step}; skipping step and halving scale.")
            optimizer.zero_grad(set_to_none=True)
            scaler.update()
            continue

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
            is_best = (val_loss < best_val_loss) and not math.isnan(val_loss)
            if is_best:
                best_val_loss = val_loss
                ckpt = {
                    "step": step,
                    "model_state": model.state_dict(),
                    "model_config": m,
                    "val_loss": val_loss,
                    "tokens_seen": tokens_seen
                }
                atomic_torch_save(ckpt, best_path)
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

        # Periodic Latest Checkpoint (Atomic Save)
        if step % t["checkpoint_interval"] == 0 or step == t["steps"]:
            atomic_torch_save({
                "step": step,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "model_config": m,
                "val_loss": best_val_loss,
                "tokens_seen": tokens_seen,
                "rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                "py_rng_state": random.getstate()
            }, latest_path)

    print("=" * 70)
    print(f"Training Complete! Best model saved to {best_path} with Val Loss: {best_val_loss:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()