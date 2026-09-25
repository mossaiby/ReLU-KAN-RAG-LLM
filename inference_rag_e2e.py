#!/usr/bin/env python3
# rag_inference.py
# 124.2M End-to-End ReLU-KAN + RAG Inference Engine (Sharpened Query Battery)

import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import ByteLevelBPETokenizer
from sentence_transformers import SentenceTransformer
import faiss

# Import your new 124M architecture
from train_kan_llm_e2e import EndToEndKANLanguageModel, TokenizerWrapper

# ---------------------------------------------------------------------------
# Configuration & Paths
# ---------------------------------------------------------------------------

CKPT_DIR = Path("./checkpoints_e2e")
DATA_DIR = Path("./data")

CKPT_INSTRUCT = CKPT_DIR / "kan_e2e_124m_rag_instruct.pt"
CKPT_BEST = CKPT_DIR / "kan_e2e_124m_best.pt"
TOK_DIR = DATA_DIR / "tokenizer_bpe_32768"

# ---------------------------------------------------------------------------
# 1. Device & Model Initialization
# ---------------------------------------------------------------------------

def init_system():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[system] Using device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # 1. Load Tokenizer
    vocab_file = TOK_DIR / "vocab.json"
    merges_file = TOK_DIR / "merges.txt"
    if not vocab_file.exists() or not merges_file.exists():
        raise FileNotFoundError(f"Tokenizer files not found in {TOK_DIR}")

    rust_tok = ByteLevelBPETokenizer.from_file(str(vocab_file), str(merges_file))
    tokenizer = TokenizerWrapper(rust_tok)
    print(f"[tokenizer] Loaded vocab size: {tokenizer.vocab_size:,}")

    # 2. Select Checkpoint (Instruct first, fallback to base)
    if CKPT_INSTRUCT.exists():
        ckpt_path = CKPT_INSTRUCT
        print(f"[model] Loading fine-tuned 124M Instruct checkpoint: {ckpt_path}")
    elif CKPT_BEST.exists():
        ckpt_path = CKPT_BEST
        print(f"[model] Instruct model not found. Falling back to 124M base checkpoint: {ckpt_path}")
    else:
        raise FileNotFoundError(f"No checkpoint found in {CKPT_DIR}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    m_cfg = ckpt.get("model_config", {})

    # 3. Instantiate 124M KAN Model
    model = EndToEndKANLanguageModel(
        vocab_size=tokenizer.vocab_size,
        dim=m_cfg.get("dim", 640),
        num_layers=m_cfg.get("num_layers", 14),
        max_len=m_cfg.get("max_len", 1024),
        k=m_cfg.get("k", 4),
        use_checkpointing=False
    ).to(device)

    model.load_state_dict(ckpt.get("model_state", ckpt))
    model.eval()
    print("[model] 124.2M End-to-End ReLU-KAN model ready in memory.")
    return model, tokenizer, device


# ---------------------------------------------------------------------------
# 2. Knowledge Base & Vector Database (FAISS + MiniLM)
# ---------------------------------------------------------------------------

DEFAULT_DOCUMENTS = [
    "Photosynthesis is the biological process used by green plants and algae to synthesize nutrients from carbon dioxide and water using sunlight, generating oxygen as a byproduct.",
    "The solar system formed approximately 4.6 billion years ago from the gravitational collapse of a giant interstellar molecular cloud. It consists of the Sun and eight planets.",
    "In computer science, an algorithm is a finite sequence of rigorous, step-by-step instructions typically used to solve a specific class of problems or perform computation.",
    "Mitochondria are membrane-bound cell organelles that generate most of the chemical energy needed to power the cell's biochemical reactions, stored in adenosine triphosphate (ATP).",
    "The Treaty of Versailles was the primary peace treaty produced by the Paris Peace Conference at the end of World War I, officially signed on June 28, 1919."
]


class VectorRetriever:
    def __init__(self, documents, device="cpu"):
        print("[retriever] Loading all-MiniLM-L6-v2 embedding model...")
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        self.documents = list(documents)

        # Build FAISS Index with cosine similarity
        embeddings = self.embedder.encode(self.documents, convert_to_numpy=True, normalize_embeddings=True)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype(np.float32))
        print(f"[retriever] FAISS index built with {len(self.documents)} knowledge chunks.")

    def add_document(self, text):
        emb = self.embedder.encode([text], convert_to_numpy=True, normalize_embeddings=True)
        self.index.add(emb.astype(np.float32))
        self.documents.append(text)

    def retrieve(self, query, top_k=1):
        q_emb = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        scores, indices = self.index.search(q_emb.astype(np.float32), top_k)
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.documents):
                results.append(self.documents[idx])
        return results


# ---------------------------------------------------------------------------
# 3. Instruct-Aligned RAG Generation Loop
# ---------------------------------------------------------------------------

@torch.inference_mode()
def answer_rag_query(query, model, tokenizer, retriever, device,
                     repetition_penalty=1.15, max_new_tokens=40):
    # 1. Retrieve the most relevant chunk
    retrieved_chunks = retriever.retrieve(query, top_k=1)
    if not retrieved_chunks:
        return "No relevant context found.", ""
    context = retrieved_chunks[0].strip()

    # 2. Strict prompt format matching SFT (NO trailing space after Answer:)
    prompt = f"Context: {context}\nQuestion: {query.strip()}\nAnswer:"

    tokens = tokenizer.encode(prompt)

    # 1024-token context window guardrail
    if len(tokens) > 950:
        truncated_context = context[:len(context) // 2]
        prompt = f"Context: {truncated_context}...\nQuestion: {query.strip()}\nAnswer:"
        tokens = tokenizer.encode(prompt)

    input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    eot_id = tokenizer.tok.token_to_id("<|endoftext|>")

    generated_tokens = []

    # 3. Autoregressive Greedy Generation with Repetition Suppression
    for _ in range(max_new_tokens):
        curr_len = input_ids.shape[1]
        if curr_len >= model.max_len:
            break

        idx_cond = input_ids[:, -model.max_len:]
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(idx_cond)

        next_logits = logits[0, -1, :].clone().float()

        # Repetition penalty: suppresses looping the same entity
        if repetition_penalty != 1.0 and len(generated_tokens) > 0:
            for prev_tok in set(generated_tokens[-12:]):
                if next_logits[prev_tok] > 0:
                    next_logits[prev_tok] /= repetition_penalty
                else:
                    next_logits[prev_tok] *= repetition_penalty

        # Greedy choice
        next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
        token_id = next_token.item()
        tok_str = tokenizer.decode([token_id])

        # Clean stopping rules
        if token_id == eot_id or "<|" in tok_str:
            break
        if "\n" in tok_str and len(generated_tokens) > 0:
            break

        generated_tokens.append(token_id)
        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

    raw_answer = tokenizer.decode(generated_tokens).strip()
    clean_answer = raw_answer.split("<|")[0].strip()
    return clean_answer, context


# ---------------------------------------------------------------------------
# 4. Main Verification Battery (De-Cluttered Triggers) & Interactive Shell
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("124.2M End-to-End ReLU-KAN + RAG Inference Engine (1024 Context)")
    print("=" * 70)

    model, tokenizer, device = init_system()
    retriever = VectorRetriever(DEFAULT_DOCUMENTS, device="cpu")

    print("\n--- Running De-Cluttered Verification Checks ---")
    test_queries = [
        "What is generated as a byproduct of photosynthesis?",
        "When was the Treaty of Versailles signed?",
        "What does an algorithm perform?",
        "In what molecule is the energy generated by mitochondria stored?",
        "What does the solar system consist of?"
    ]

    for q in test_queries:
        ans, ctx = answer_rag_query(q, model, tokenizer, retriever, device)
        print(f"\n[Question]: {q}")
        print(f"[Retrieved]: {ctx}")
        print(f"[Answer]   : {ans}")

    print("\n" + "=" * 70)
    print("Interactive Mode Ready! Type your question, or 'exit' to quit.")
    print("=" * 70)

    while True:
        try:
            user_query = input("\nQuestion: ").strip()
            if not user_query:
                continue
            if user_query.lower() in ("exit", "quit", "q"):
                print("Exiting RAG session.")
                break

            ans, ctx = answer_rag_query(user_query, model, tokenizer, retriever, device)
            print(f"Answer: {ans}")
            print(f"(Source Context: {ctx})")

        except KeyboardInterrupt:
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()