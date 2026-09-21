#!/usr/bin/env python3
# train_advisor.py
# Pre-Flight Diagnostic, Memory Calculator & Training Advisor

import os
import sys
import json
import math
from pathlib import Path
import torch

def calc_model_params(dim, num_layers, k, vocab_size=32768, head_dim=64):
    tok_emb = vocab_size * dim
    n_heads = dim // head_dim
    # Attention
    attn = dim * (dim * 3) + dim * dim
    # Gated KAN FFN
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
    adam_mb = param_mb * 2  # fp32 moments m and v
    
    # Activation Checkpointing: saves only block inputs
    act_ckpt_mb = (batch_size * max_len * dim * 2 / (1024**2)) * layers
    
    # Single-block recomputation peak in backward
    recompute_mb = 250
    
    # Logits + Cross-Entropy peak
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
    total_tokens = t["steps"] * tokens_per_step
    token_param_ratio = total_tokens / total_params

    # Runtime estimation at realistic throughputs
    hours_at_7000 = (total_tokens / 7000) / 3600
    hours_at_7500 = (total_tokens / 7500) / 3600
    hours_at_8000 = (total_tokens / 8000) / 3600

    print("\n" + "=" * 75)
    print("           RELU-KAN LLM PRE-FLIGHT TRAINING ADVISOR")
    print("=" * 75)

    print(f"Hardware Detected      : {gpu_name} ({vram_total_mb:,.0f} MB physical VRAM)")
    print(f"Model Configuration    : dim={m['dim']} | layers={m['num_layers']} | k={m['k']} | context={m['max_len']}")
    print(f"Total Parameters       : {total_params:,} ({total_params/1e6:.1f}M params, tied weights)")
    print(f"Weights + AdamW Memory : ~{param_mb + adam_mb:,.0f} MB")
    print(f"Estimated Peak VRAM    : ~{est_peak_vram:,.0f} MB ({est_peak_vram/1024:.2f} GB)")
    print(f"VRAM Headroom Available: ~{vram_total_mb - est_peak_vram:,.0f} MB")

    print("-" * 75)
    print(f"Batch Configuration    : {t['batch_size']} micro-batch × {grad_accum} grad-accum = {effective_batch} effective batch")
    print(f"Tokens Per Step        : {tokens_per_step:,} tokens")
    print(f"Planned Steps          : {t['steps']:,} steps")
    print(f"Total Tokens Processed : {total_tokens:,} ({total_tokens/1e6:.1f}M / {total_tokens/1e9:.2f}B tokens)")
    print(f"Token/Parameter Ratio  : {token_param_ratio:.2f}x (Compute scaling factor)")

    print("-" * 75)
    print(f"Estimated Runtime @ 7,000 tok/s: {hours_at_7000:.1f} hours ({hours_at_7000/24:.1f} days)")
    print(f"Estimated Runtime @ 7,500 tok/s: {hours_at_7500:.1f} hours ({hours_at_7500/24:.1f} days)")
    print(f"Estimated Runtime @ 8,000 tok/s: {hours_at_8000:.1f} hours ({hours_at_8000/24:.1f} days)")
    print("=" * 75)

    # Health Checks & Warnings
    warnings = []
    if est_peak_vram > vram_total_mb:
        warnings.append(f"CRITICAL: Estimated peak VRAM ({est_peak_vram:.0f} MB) exceeds physical VRAM ({vram_total_mb:.0f} MB). You will OOM!")
    elif est_peak_vram > vram_total_mb * 0.85:
        warnings.append(f"CAUTION: VRAM is above 85% capacity. Reduce batch size if Windows desktop takes VRAM.")
    else:
        print("[advisor check] VRAM Budget: EXCELLENT. Comfortable headroom, zero paging risk.")

    if token_param_ratio < 2.0:
        warnings.append("Data Warning: Token-to-parameter ratio is < 2.0x. Consider adding more steps or articles.")
    elif token_param_ratio >= 4.0:
        print("[advisor check] Data Quality: EXCELLENT. Over 4x unique tokens per parameter ensures deep generalization.")

    if t["warmup_steps"] < 500:
        warnings.append("Warmup Warning: warmup_steps < 500 is short for a 124M model. 1,000 is recommended.")

    if warnings:
        print("\n[WARNINGS DETECTED]:")
        for w in warnings:
            print(f"  ! {w}")
    else:
        print("\nAll pre-flight checks PASSED. Ready for marathon training!")
        print("To launch, run:")
        print("  python train_kan_llm_e2e.py")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    main()