#!/usr/bin/env python3
# finetune_rag_e2e.py
# 124M End-to-End ReLU-KAN SFT with 4x Answer-Loss Amplification

import time
import math
import random
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer

from train_kan_llm_e2e import EndToEndKANLanguageModel, TokenizerWrapper

# ---------------------------------------------------------------------------
# Setup & Paths
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_IN = Path("./checkpoints_e2e/kan_e2e_124m_best.pt")
CKPT_OUT = Path("./checkpoints_e2e/kan_e2e_124m_rag_instruct.pt")
TOK_DIR = Path("./data/tokenizer_bpe_32768")

BATCH_SIZE = 8
STEPS = 1500        # ~9 minutes on RTX 2080 Ti
LR = 3.5e-5
MIN_LR = 5e-6
MAX_LEN = 1024
ANSWER_WEIGHT = 4.0  # 4x gradient penalty on answer tokens
EVAL_INTERVAL = 100

# 1. Load Tokenizer & Base Model
print(f"[init] Using device: {DEVICE}")
print(f"[init] Loading 124M base checkpoint from {CKPT_IN} ...")
rust_tok = ByteLevelBPETokenizer.from_file(str(TOK_DIR / "vocab.json"), str(TOK_DIR / "merges.txt"))
tokenizer = TokenizerWrapper(rust_tok)
eot_id = tokenizer.tok.token_to_id("<|endoftext|>") or 0

ckpt = torch.load(CKPT_IN, map_location=DEVICE, weights_only=False)
m_cfg = ckpt["model_config"]

model = EndToEndKANLanguageModel(
    vocab_size=tokenizer.vocab_size,
    dim=m_cfg["dim"],
    num_layers=m_cfg["num_layers"],
    max_len=m_cfg["max_len"],
    k=m_cfg["k"],
    use_checkpointing=True
).to(DEVICE)

model.load_state_dict(ckpt["model_state"])
print("[init] Base model loaded. Preparing instruction data...")

# ---------------------------------------------------------------------------
# 2. Prepare SQuAD with Token-Level Loss Weights
# ---------------------------------------------------------------------------
print("[data] Loading 'rajpurkar/squad' train and validation splits...")
raw_train = load_dataset("rajpurkar/squad", split="train")
raw_val = load_dataset("rajpurkar/squad", split="validation")

def process_split(ds, limit=16000):
    pairs = []
    for item in ds:
        ctx = item["context"].strip()
        q = item["question"].strip()
        ans = item["answers"]["text"][0].strip() if item["answers"]["text"] else ""
        if not ans or len(ans) > 80:
            continue
        pairs.append((ctx, q, ans))
        if len(pairs) >= limit:
            break
    return pairs

train_pairs = process_split(raw_train, limit=16000)
val_pairs = process_split(raw_val, limit=800)

random.seed(42)
random.shuffle(train_pairs)
print(f"[data] Prepared {len(train_pairs):,} train | {len(val_pairs):,} val pairs.")


class WeightedQADataset(Dataset):
    def __init__(self, pairs, tokenizer, max_len=1024, ans_weight=4.0):
        self.samples = []
        for ctx, q, ans in pairs:
            prompt_str = f"Context: {ctx}\nQuestion: {q}\nAnswer:"
            # Trailing space included in answer string to align with BPE leading-space tokens
            answer_str = f" {ans}<|endoftext|>"

            p_tokens = tokenizer.encode(prompt_str)
            a_tokens = tokenizer.encode(answer_str)

            if len(p_tokens) + len(a_tokens) > max_len:
                p_tokens = p_tokens[:max_len - len(a_tokens)]

            input_ids = p_tokens + a_tokens
            targets = input_ids[1:]

            # Token weights: 1.0 on Context/Question, 4.0 on Answer tokens!
            weights = [1.0] * (len(p_tokens) - 1) + [ans_weight] * len(a_tokens)

            self.samples.append((
                torch.tensor(input_ids[:-1], dtype=torch.long),
                torch.tensor(targets, dtype=torch.long),
                torch.tensor(weights, dtype=torch.float32)
            ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    max_len = max(len(x) for x, _, _ in batch)
    x_batch = torch.full((len(batch), max_len), eot_id, dtype=torch.long)
    y_batch = torch.full((len(batch), max_len), -100, dtype=torch.long)
    w_batch = torch.zeros(len(batch), max_len, dtype=torch.float32)

    for i, (x, y, w) in enumerate(batch):
        x_batch[i, :len(x)] = x
        y_batch[i, :len(y)] = y
        w_batch[i, :len(w)] = w

    return x_batch.to(DEVICE), y_batch.to(DEVICE), w_batch.to(DEVICE)


train_loader = DataLoader(WeightedQADataset(train_pairs, tokenizer, MAX_LEN, ANSWER_WEIGHT),
                          batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(WeightedQADataset(val_pairs, tokenizer, MAX_LEN, ANSWER_WEIGHT),
                        batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# ---------------------------------------------------------------------------
# 3. Validation Routine
# ---------------------------------------------------------------------------
@torch.inference_mode()
def evaluate_val_loss(model, val_loader, max_batches=25):
    was_training = model.training
    model.eval()
    losses = []
    for i, (x, y, w) in enumerate(val_loader):
        if i >= max_batches:
            break
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(x)
            raw_loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), y.reshape(-1), reduction="none")
            weighted = (raw_loss * w.reshape(-1)).sum() / w.reshape(-1).clamp(min=1.0).sum()
        if not (torch.isnan(weighted) or torch.isinf(weighted)):
            losses.append(weighted.item())
    model.train(was_training)
    return sum(losses) / len(losses) if losses else float("inf")


# ---------------------------------------------------------------------------
# 4. End-to-End Fine-Tuning Loop
# ---------------------------------------------------------------------------
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scaler = torch.amp.GradScaler("cuda", init_scale=32768.0)

model.train()
data_iter = iter(train_loader)
best_val_loss = float("inf")

print("=" * 75)
print(f"Starting 124M SFT with 4x Answer Amplification ({STEPS} steps, ~9 mins)")
print("=" * 75)

t0 = time.time()
running_train_loss = 0.0

for step in range(1, STEPS + 1):
    try:
        x, y, w = next(data_iter)
    except StopIteration:
        data_iter = iter(train_loader)
        x, y, w = next(data_iter)

    # Cosine learning rate decay across steps
    progress = step / STEPS
    cur_lr = MIN_LR + (LR - MIN_LR) * 0.5 * (1.0 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = cur_lr

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = model(x)
        raw_loss = F.cross_entropy(logits.reshape(-1, model.vocab_size), y.reshape(-1), reduction="none")
        loss = (raw_loss * w.reshape(-1)).sum() / w.reshape(-1).clamp(min=1.0).sum()

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()

    running_train_loss += loss.item()

    # Periodic Validation & Best Model Checkpoint
    if step % EVAL_INTERVAL == 0 or step == STEPS:
        avg_train = running_train_loss / EVAL_INTERVAL
        running_train_loss = 0.0

        val_loss = evaluate_val_loss(model, val_loader, max_batches=25)
        is_best = val_loss < best_val_loss

        if is_best:
            best_val_loss = val_loss
            ckpt_save = {
                "step": step,
                "model_state": model.state_dict(),
                "model_config": m_cfg,
                "val_loss": val_loss
            }
            torch.save(ckpt_save, CKPT_OUT)
            saved_str = "[SAVED NEW BEST]"
        else:
            saved_str = ""

        print(f"Step {step:04d}/{STEPS} | Train: {avg_train:.4f} | Val: {val_loss:.4f} (Best: {best_val_loss:.4f}) {saved_str} | {time.time()-t0:.0f}s")

if not CKPT_OUT.exists():
    ckpt_save = {
        "step": STEPS,
        "model_state": model.state_dict(),
        "model_config": m_cfg,
        "val_loss": best_val_loss
    }
    torch.save(ckpt_save, CKPT_OUT)

print("=" * 75)
print(f"[done] SFT Complete! Best model saved to {CKPT_OUT} with Val Loss: {best_val_loss:.4f}")
print("=" * 75)