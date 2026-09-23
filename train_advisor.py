#!/usr/bin/env python3
# train_advisor.py
# Pre-Flight Diagnostic, Cache Inspector & Resilient Training Advisor

import os
import sys
import json
import math
from pathlib import Path
import numpy as np
import torch

def calc_model_params(dim, num_layers, k, vocab_size=32768, head_dim=64):
    tok_emb = vocab_size * dim
    n_heads = dim // head_dim
    attn = dim * (dim * 3) + dim * dim
    gate_kan = dim * (dim * 2) + (dim * 2) * (dim * k)
    up = dim * (dim * 2)
    down = (dim * 2) * dim
    ffn = gate_kan + up + down
    lns = dim * 4
    block = attn + ffn + lns
    return tok_emb + num_layers * block

def estimate_vram(dim, layers, k, max_len, batch_size, vocab_size=32768):
    total_params = calc_model_params(dim, layers, k, vocab_size)
    param_mb = total_params * 4 / (1024**2)
    grad_mb = param_mb
    adam_mb = param_mb * 2
    act_ckpt_mb = (batch_size * max_len * dim * 2 / (1024**2)) * layers
    recompute_mb = 250
    logits_mb = (batch_size * max_len * vocab_size * 2) / (1024**2)
    ce_mb = (batch_size * max_len * vocab_size * 4 * 2) / (1024**2)
    total_peak = param_mb + grad_mb + adam_mb + act_ckpt_mb + recompute_mb + logits_mb + ce_mb
    return total_params, param_mb, adam_mb, total_peak

def main():
    cfg_path = Path("config_e2e.json")
    if not cfg_path.exists():
        from train_kan_llm_e2e import DEFAULT_CONFIG
        cfg = DEFAULT_CONFIG
        print("[advisor] Using DEFAULT_CONFIG (config_e2e.json not found).")
    else:
        with open(cfg_path) as f:
            cfg = json.load(f)
        print(f"[advisor] Loaded configuration from {cfg_path}")

    m = cfg["model"]
    t = cfg["training"]
    d = cfg["data"]

    device_count = torch.cuda.device_count()
    has_cuda = torch.cuda.is_available()
    gpu_name = torch.cuda.get_device_name(0) if has_cuda else "None (CPU)"
    vram_total_mb = torch.cuda.get_device_properties(0).total_memory / (1024**2) if has_cuda else 0

    total_params, param_mb, adam_mb, est_peak_vram = estimate_vram(
        m["dim"], m["num_layers"], m["k"], m["max_len"], t["batch_size"], d["tokenizer_vocab_size"]
    )

    grad_accum = t.get("grad_accum_steps", 1)
    effective_batch = t["batch_size"] * grad_accum
    tokens_per_step = effective_batch * m["max_len"]
    total_tokens_planned = t["steps"] * tokens_per_step

    # Inspect on-disk binary token cache
    data_dir = Path(d["data_dir"])
    train_bins = list(data_dir.glob("fineweb_train_*.bin"))
    cached_tokens_count = 0
    cache_file_str = "None found"

    if train_bins:
        newest_cache = max(train_bins, key=os.path.getmtime)
        cached_tokens_count = os.path.getsize(newest_cache) // 4
        cache_file_str = f"{newest_cache.name} ({cached_tokens_count:,} tokens / ~{cached_tokens_count / 1e9:.2f}B)"

    # Inspect checkpoint progress with power-cut corruption resilience
    ckpt_dir = Path(cfg["io"]["checkpoint_dir"])
    ckpt_latest = ckpt_dir / f"{cfg['io']['checkpoint_name']}_latest.pt"
    ckpt_best = ckpt_dir / f"{cfg['io']['checkpoint_name']}_best.pt"

    completed_steps = 0
    tokens_seen = 0
    best_val_loss = float("inf")
    active_ckpt_str = "None"
    ckpt_meta = None

    if ckpt_latest.exists():
        try:
            ckpt_meta = torch.load(ckpt_latest, map_location="cpu", weights_only=False)
            active_ckpt_str = f"{ckpt_latest.name} (Healthy)"
        except Exception as e:
            print(f"\n[advisor warning] '{ckpt_latest.name}' is corrupted ({e}) from the power outage.")
            if ckpt_best.exists():
                try:
                    ckpt_meta = torch.load(ckpt_best, map_location="cpu", weights_only=False)
                    active_ckpt_str = f"{ckpt_best.name} (Healthy Fallback)"
                    print(f"[advisor warning] Recovered using healthy backup: '{ckpt_best.name}'\n")
                except Exception as e_best:
                    print(f"[advisor error] Both latest and best failed to load: {e_best}\n")

    elif ckpt_best.exists():
        try:
            ckpt_meta = torch.load(ckpt_best, map_location="cpu", weights_only=False)
            active_ckpt_str = f"{ckpt_best.name} (Healthy)"
        except Exception as e:
            print(f"[advisor error] Could not load '{ckpt_best.name}': {e}")

    if ckpt_meta is not None:
        completed_steps = ckpt_meta.get("step", 0)
        tokens_seen = ckpt_meta.get("tokens_seen", 0)
        best_val_loss = ckpt_meta.get("val_loss", float("inf"))

    remaining_steps = max(0, t["steps"] - completed_steps)
    remaining_tokens = remaining_steps * tokens_per_step
    hours_at_14500 = (remaining_tokens / 14500) / 3600

    print("\n" + "=" * 75)
    print("           RELU-KAN LLM PRE-FLIGHT TRAINING ADVISOR")
    print("=" * 75)

    print(f"Hardware Detected      : {gpu_name} ({vram_total_mb:,.0f} MB physical VRAM)")
    print(f"Model Configuration    : dim={m['dim']} | layers={m['num_layers']} | k={m['k']} | context={m['max_len']}")
    print(f"Total Parameters       : {total_params:,} ({total_params/1e6:.1f}M params, tied weights)")
    print(f"Estimated Peak VRAM    : ~{est_peak_vram:,.0f} MB ({est_peak_vram/1024:.2f} GB)")
    print(f"VRAM Headroom Available: ~{vram_total_mb - est_peak_vram:,.0f} MB")

    print("-" * 75)
    print(f"Active Checkpoint      : {active_ckpt_str}")
    if completed_steps > 0:
        print(f"Progress So Far        : Step {completed_steps:,} / {t['steps']:,} (Best Val Loss: {best_val_loss:.4f})")
        print(f"Tokens Consumed        : {tokens_seen:,} (~{tokens_seen / 1e6:.1f}M tokens)")

    print("-" * 75)
    print(f"Disk Token Cache Found : {cache_file_str}")
    if cached_tokens_count > 0:
        unseen_tokens = max(0, cached_tokens_count - tokens_seen)
        print(f"Unseen Tokens Remaining: {unseen_tokens:,} (~{unseen_tokens / 1e9:.2f}B fresh tokens available on disk)")

    print("-" * 75)
    print(f"Batch Configuration    : {t['batch_size']} micro-batch × {grad_accum} grad-accum = {effective_batch} effective batch")
    print(f"Tokens Per Step        : {tokens_per_step:,} tokens/step")
    print(f"Remaining Steps        : {remaining_steps:,} steps ({remaining_tokens / 1e6:.1f}M fresh tokens)")
    print(f"Estimated Time to Done : ~{hours_at_14500:.1f} hours (@ 14,500 tok/s)")
    print("=" * 75)

    print(f"\nRecommended command to run:")
    print(f"  python train_kan_llm_e2e.py --batch-size {t['batch_size']} --grad-accum {grad_accum} --steps {t['steps']} --save-config\n")

if __name__ == "__main__":
    main()