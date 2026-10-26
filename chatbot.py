import os
import subprocess
from typing import List, Tuple
from torch.utils.data import DataLoader

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from dataset import DialogueDataset
from transformers import (
    AutoTokenizer,
    Gemma3ForCausalLM,
    BitsAndBytesConfig,
)

# torch.cuda.empty_cache()
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ───────────────────────── CONFIG ─────────────────────────
DATA_PATH      = "data/child_multimodal_dataset.json"
REBUILD_INDEX  = False  # True → force rebuild

_EMBED_MODEL_ID = "GanymedeNil/text2vec-large-chinese"
_INDEX_FILE     = "rag/text2vec.faiss"
_DOCS_FILE      = "rag/text2vec.npy"
_GEMMA_ID       = "google/gemma-3-4b-it"
# ──────────────────────────────────────────────────────────

def proxy():
    """Load HTTP/HTTPS proxy env vars from /etc/network_turbo."""
    result = subprocess.run(
        'bash -c "source /etc/network_turbo && env | grep proxy"',
        shell=True,
        capture_output=True,
        text=True
    )
    for line in result.stdout.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            os.environ[k] = v
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

def collate_text(batch):
    """Return the 'dialog_item' dicts for each sample in the batch."""
    return [item["dialog_item"] for item in batch]


def load_dialogues(file_path: str) -> List[str]:
    """
    Load each CHILD utterance as a standalone document with no role tags.
    """
    docs: List[str] = []
    dataset = DialogueDataset(file_path)
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_text,
    )

    for convo in loader:
        utt = convo[0]
        if utt["role"] == "child":
            docs.append(utt["text"].strip())

    print(f"Loaded {len(docs)} child utterances.")
    return docs


def get_embedder(device: str = "cuda") -> SentenceTransformer:
    print(f"Embedder: Using device {device}")
    return SentenceTransformer(_EMBED_MODEL_ID, device=device)


def embed_texts(
    embedder: SentenceTransformer,
    texts: List[str],
    batch_size: int = 64
) -> np.ndarray:
    return embedder.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def build_or_load_faiss(
    docs: List[str],
    embedder: SentenceTransformer,
    rebuild: bool = False,
) -> Tuple[faiss.IndexFlatIP, List[str]]:
    """
    Builds a FAISS IndexFlatIP on the embedded docs, or loads it if already saved.
    """
    if (
        not rebuild
        and os.path.exists(_INDEX_FILE)
        and os.path.exists(_DOCS_FILE)
    ):
        index = faiss.read_index(_INDEX_FILE)
        cached_docs = np.load(_DOCS_FILE, allow_pickle=True).tolist()
        print("Loaded existing FAISS index.")
        return index, cached_docs

    vecs = embed_texts(embedder, docs)
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)

    os.makedirs(os.path.dirname(_INDEX_FILE), exist_ok=True)
    faiss.write_index(index, _INDEX_FILE)
    np.save(_DOCS_FILE, np.array(docs, dtype=object))
    print("Built and saved new FAISS index.")
    return index, docs


def load_gemma_quantized(model_id: str = _GEMMA_ID) -> Tuple[pipeline, str]:
    """
    Loads Gemma-3 chat model via HF pipeline in bfloat16 on CUDA.
    """
    # cfg = BitsAndBytesConfig(
    #     load_in_8bit=True,
    #     llm_int8_threshold=6.0,
    #     # llm_int8_enable_fp32_cpu_offload=True
    # )
    # model = Gemma3ForCausalLM.from_pretrained(
    #     _GEMMA_ID, quantization_config=cfg, torch_dtype=torch.bfloat16,
    #     device_map="auto"
    # ).eval()
    # tok = AutoTokenizer.from_pretrained(_GEMMA_ID)
    
    # chat = pipeline("text-generation", model=model, tokenizer=tok)

    chat = pipeline(
        "text-generation",
        model=model_id,
        device="cuda",
        torch_dtype=torch.bfloat16,
    )
    return chat, "cuda"


def rag_answer(
    query: str,
    index: faiss.IndexFlatIP,
    docs: List[str],
    embedder: SentenceTransformer,
    chat_pipe,
    k: int = 5,
    max_new: int = 256,
) -> str:
    # ── 1) Retrieve top-k snippets ───────────────────────────────────────────
    q_vec = embed_texts(embedder, [query])
    scores, indices = index.search(q_vec, k)
    snippets = [docs[i] for i in indices[0]]
    context = "\n".join(f"检索内容 {j+1}: {snip}" for j, snip in enumerate(snippets))

    # Debug print
    print("[DEBUG] 检索到的上下文：\n" + context + "\n")

    # ── 2) Build Chinese prompts ──────────────────────────────────────────
    system_prompt = (
        "你是一位面向家长的专属助理，仅在家长询问其孩子的状况时回答问题；"
        "如果提问与孩子状况无关，请回复：“对不起，我无法回答这个问题。”"
        "不要猜测、编造或产生幻觉；所有回答中均以“您的孩子”作主语。"
    )
    user_prompt = (
        "以下是从孩子对话中提取的独立片段：\n\n"
        f"{context}\n\n"
        "请仅基于上述片段，回答以下有关“孩子状况”的问题：\n"
        f"{query}"
    )

    messages = [[
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user",   "content": [{"type": "text", "text": user_prompt}]},
    ]]

    # ── 3) Generate with Gemma-3 ────────────────────────────────────────────
    out = chat_pipe(text_inputs=messages, max_new_tokens=max_new, top_p=None, top_k=None, do_sample=False)

    turns = out[0][0]['generated_text']

    # find the assistant turn
    for turn in turns:
        if turn.get("role") == "assistant":
            content = turn["content"]
            return content.strip()
    return ""


def initialize_system(
    data_path: str,
    rebuild_index: bool = False,
) -> Tuple[
    faiss.IndexFlatIP,
    List[str],
    SentenceTransformer,
    pipeline,
]:
    """
    Full startup: proxy, load & index corpus, load Gemma-3.
    """
    proxy()
    docs       = load_dialogues(data_path)
    embedder   = get_embedder()
    index, docs = build_or_load_faiss(docs, embedder, rebuild_index)
    chat_pipe, device = load_gemma_quantized()
    print(f"Gemma-3 running on {device}")
    return index, docs, embedder, chat_pipe


def ask_parent(
    query: str,
    index: faiss.IndexFlatIP,
    docs: List[str],
    embedder: SentenceTransformer,
    chat_pipe,
) -> str:
    """
    One-shot RAG call: retrieve + generate.
    """
    return rag_answer(query, index, docs, embedder, chat_pipe)


if __name__ == "__main__":
    idx, docs, embedder, pipe = initialize_system(DATA_PATH, REBUILD_INDEX)

    print("\nRAG ready. Ask away (type 'exit' to quit).\n")
    while True:
        q = input("Parent> ")
        if q.lower() in {"exit", "quit"}:
            break
        ans = ask_parent(q, idx, docs, embedder, pipe)
        print("Assistant>", ans, "\n")
