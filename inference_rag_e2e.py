#!/usr/bin/env python3
# inference_rag_e2e.py
# 124.2M End-to-End ReLU-KAN + RAG Engine (Context-Grounded & Candidate-Verified)

import os
import sys
import re
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import ByteLevelBPETokenizer
from sentence_transformers import SentenceTransformer
import faiss

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

    vocab_file = TOK_DIR / "vocab.json"
    merges_file = TOK_DIR / "merges.txt"
    if not vocab_file.exists() or not merges_file.exists():
        raise FileNotFoundError(f"Tokenizer files not found in {TOK_DIR}")

    rust_tok = ByteLevelBPETokenizer.from_file(str(vocab_file), str(merges_file))
    tokenizer = TokenizerWrapper(rust_tok)
    print(f"[tokenizer] Loaded vocab size: {tokenizer.vocab_size:,}")

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
        clean_text = text.strip()
        if not clean_text:
            return
        emb = self.embedder.encode([clean_text], convert_to_numpy=True, normalize_embeddings=True)
        self.index.add(emb.astype(np.float32))
        self.documents.append(clean_text)
        print(f"[retriever] Successfully added new document (Total indexed: {len(self.documents)}).")

    def retrieve(self, query, top_k=1):
        q_emb = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        scores, indices = self.index.search(q_emb.astype(np.float32), top_k)
        results = []
        for idx in indices[0]:
            if 0 <= idx < len(self.documents):
                results.append(self.documents[idx])
        return results


# ---------------------------------------------------------------------------
# 3. Instruct-Aligned RAG Generation Loop (With Span Grounding)
# ---------------------------------------------------------------------------

def extract_valid_context_tokens(context, tokenizer):
    """Extracts all token IDs that can validly begin any word in the retrieved context."""
    # Split text into clean words/symbols
    words = re.findall(r"[A-Za-z0-9]+|[(),.!?]", context)
    valid_ids = set()
    for w in words:
        # Word with standard BPE leading space
        ids_space = tokenizer.encode(" " + w)
        if ids_space:
            valid_ids.add(ids_space[0])
        # Word without leading space
        ids_raw = tokenizer.encode(w)
        if ids_raw:
            valid_ids.add(ids_raw[0])
    return valid_ids


@torch.inference_mode()
def answer_rag_query(query, model, tokenizer, retriever, device,
                     repetition_penalty=1.15, max_new_tokens=40, debug=False, ground_to_context=True):
    retrieved_chunks = retriever.retrieve(query, top_k=1)
    if not retrieved_chunks:
        return "No relevant context found.", ""
    context = retrieved_chunks[0].strip()

    prompt = f"Context: {context}\nQuestion: {query.strip()}\nAnswer:"

    tokens = tokenizer.encode(prompt)
    if len(tokens) > 950:
        truncated_context = context[:len(context) // 2]
        prompt = f"Context: {truncated_context}...\nQuestion: {query.strip()}\nAnswer:"
        tokens = tokenizer.encode(prompt)

    # 1. Identify question tokens to suppress echoing
    raw_q_tokens = tokenizer.encode(query.strip())
    suppress_tokens = set()
    for tid in raw_q_tokens:
        tok_text = tokenizer.decode([tid]).strip().lower()
        if len(tok_text) > 2 and tok_text not in {
            "what", "when", "where", "which", "who", "the", "was", "did",
            "does", "in", "on", "at", "is", "are", "for", "from", "of", "to"
        }:
            suppress_tokens.add(tid)

    # 2. Extract valid context starting tokens for span grounding
    valid_context_tokens = extract_valid_context_tokens(context, tokenizer) if ground_to_context else set()

    input_ids = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    eot_id = tokenizer.tok.token_to_id("<|endoftext|>")

    generated_tokens = []
    top_candidates = []
    first_token_prob = 1.0

    for gen_step in range(max_new_tokens):
        curr_len = input_ids.shape[1]
        if curr_len >= model.max_len:
            break

        idx_cond = input_ids[:, -model.max_len:]
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(idx_cond)

        next_logits = logits[0, -1, :].clone().float()

        # Echo suppression on first 3 tokens
        if gen_step < 3 and suppress_tokens:
            for s_id in suppress_tokens:
                next_logits[s_id] -= 6.0

        # Repetition suppression on recently generated tokens
        if repetition_penalty != 1.0 and len(generated_tokens) > 0:
            for prev_tok in set(generated_tokens[-12:]):
                if next_logits[prev_tok] > 0:
                    next_logits[prev_tok] /= repetition_penalty
                else:
                    next_logits[prev_tok] *= repetition_penalty

        probs = F.softmax(next_logits, dim=-1)

        # Inspect top-5 candidate tokens on token 0
        if gen_step == 0:
            top_p, top_i = torch.topk(probs, 5)
            first_token_prob = top_p[0].item()
            if debug:
                for p_val, i_val in zip(top_p.tolist(), top_i.tolist()):
                    top_candidates.append(f"{tokenizer.decode([i_val])!r} ({p_val:.1%})")

        # Step 0: Apply Context Span Grounding (prioritize candidates that exist in the text)
        if gen_step == 0 and ground_to_context and valid_context_tokens:
            # Check the top 5 candidates; if an out-of-context token won, look for the first valid context word
            top_candidates_ids = torch.topk(next_logits, 10).indices.tolist()
            chosen_token_id = top_candidates_ids[0]
            for cand_id in top_candidates_ids:
                if cand_id in valid_context_tokens:
                    chosen_token_id = cand_id
                    break
            token_id = chosen_token_id
        else:
            next_token = torch.argmax(next_logits, dim=-1, keepdim=True)
            token_id = next_token.item()

        tok_str = tokenizer.decode([token_id])

        if token_id == eot_id or "<|" in tok_str:
            break
        if "\n" in tok_str and len(generated_tokens) > 0:
            break

        generated_tokens.append(token_id)
        next_token_tensor = torch.tensor([[token_id]], dtype=torch.long, device=device)
        input_ids = torch.cat([input_ids, next_token_tensor], dim=1)

    raw_answer = tokenizer.decode(generated_tokens).strip()
    clean_answer = raw_answer.split("<|")[0].strip()

    if debug and top_candidates:
        conf_label = "HIGH" if first_token_prob >= 0.20 else ("MEDIUM" if first_token_prob >= 0.08 else "LOW")
        debug_str = f"\n  [Confidence: {conf_label} ({first_token_prob:.1%}) | Candidates: {', '.join(top_candidates)}]"
        clean_answer = f"{clean_answer}{debug_str}"

    return clean_answer, context


# ---------------------------------------------------------------------------
# 4. Main Verification Battery & Interactive Shell
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("124.2M End-to-End ReLU-KAN + RAG Inference Engine (1024 Context)")
    print("=" * 70)

    model, tokenizer, device = init_system()
    retriever = VectorRetriever(DEFAULT_DOCUMENTS, device="cpu")

    print("\n--- Running Automated Verification Checks ---")
    test_queries = [
        "When was the Treaty of Versailles signed?",
        "What does an algorithm perform?",
        "What does the solar system consist of?",
        "What treaty was produced by the Paris Peace Conference?",
        "What gas is generated as a byproduct of photosynthesis?"
    ]

    for q in test_queries:
        ans, ctx = answer_rag_query(q, model, tokenizer, retriever, device)
        print(f"\n[Question]: {q}")
        print(f"[Retrieved]: {ctx}")
        print(f"[Answer]   : {ans}")

    print("\n" + "=" * 70)
    print("Interactive Mode Ready!")
    print("Commands:")
    print("  /add <text>     - Paste and index a new paragraph into FAISS")
    print("  /list           - List all currently indexed documents")
    print("  /debug on/off   - Show top-5 candidate token probabilities")
    print("  /ground on/off  - Toggle Context Span Grounding (Default: ON)")
    print("  exit            - Quit the session")
    print("=" * 70)

    debug_mode = True
    grounding = True

    while True:
        try:
            user_input = input("\nQuestion: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "q"):
                print("Exiting RAG session.")
                break

            if user_input.lower() == "/debug on":
                debug_mode = True
                print("[debug] Top-5 token candidate inspector enabled.")
                continue
            elif user_input.lower() == "/debug off":
                debug_mode = False
                print("[debug] Top-5 token candidate inspector disabled.")
                continue

            if user_input.lower() == "/ground on":
                grounding = True
                print("[ground] Context span grounding enabled (filters out-of-context tokens).")
                continue
            elif user_input.lower() == "/ground off":
                grounding = False
                print("[ground] Context span grounding disabled.")
                continue

            if user_input.startswith("/add "):
                new_doc = user_input[5:].strip()
                retriever.add_document(new_doc)
                continue

            if user_input.lower() == "/list":
                print("\nCurrently Indexed Documents:")
                for i, doc in enumerate(retriever.documents, 1):
                    print(f"  [{i}] {doc[:100]}...")
                continue

            ans, ctx = answer_rag_query(user_input, model, tokenizer, retriever, device,
                                        debug=debug_mode, ground_to_context=grounding)
            print(f"Answer: {ans}")
            print(f"(Source Context: {ctx})")

        except KeyboardInterrupt:
            print("\nExiting.")
            break


if __name__ == "__main__":
    main()