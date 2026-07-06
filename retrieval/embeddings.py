"""BGE-M3 dense encoder + BGE-reranker-v2-m3 cross-encoder. Both cached locally."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

DENSE_MODEL = "BAAI/bge-m3"
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
DENSE_DIM = 1024


class DenseEncoder:
    def __init__(self, model_name: str = DENSE_MODEL, device: str | None = None, batch_size: int = 16):
        from transformers import AutoModel, AutoTokenizer

        self.device = device or os.environ.get("LAW_INDEX_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.batch_size = batch_size

    @torch.inference_mode()
    def encode(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), DENSE_DIM), dtype=np.float32)
        for i in range(0, len(texts), self.batch_size):
            batch = self.tokenizer(
                texts[i : i + self.batch_size], padding=True, truncation=True, max_length=512, return_tensors="pt"
            ).to(self.device)
            outputs = self.model(**batch)
            hidden = outputs.last_hidden_state[:, 0]
            hidden = torch.nn.functional.normalize(hidden, p=2, dim=1)
            out[i : i + self.batch_size] = hidden.cpu().numpy()
        return out

    @torch.inference_mode()
    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]


class Reranker:
    def __init__(self, model_name: str = RERANK_MODEL, device: str | None = None, batch_size: int = 16):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = device or os.environ.get("LAW_INDEX_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device).eval()
        self.batch_size = batch_size

    @torch.inference_mode()
    def rerank(self, query: str, docs: list[str]) -> np.ndarray:
        scores = np.zeros(len(docs), dtype=np.float32)
        for i in range(0, len(docs), self.batch_size):
            batch_docs = docs[i : i + self.batch_size]
            pairs = [[query, d] for d in batch_docs]
            inputs = self.tokenizer(
                pairs, padding=True, truncation=True, max_length=512, return_tensors="pt"
            ).to(self.device)
            logits = self.model(**inputs).logits.squeeze(-1)
            scores[i : i + self.batch_size] = logits.cpu().numpy()
        return scores


def save_dense(emb: np.ndarray, uids: list[str], runs_dir: Path | str) -> None:
    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    np.save(runs_dir / "law_dense.npy", emb.astype(np.float32))
    import json

    (runs_dir / "law_dense_meta.json").write_text(json.dumps(uids, ensure_ascii=False))


def load_dense(runs_dir: Path | str) -> tuple[np.ndarray, list[str]]:
    runs_dir = Path(runs_dir)
    import json

    emb = np.load(runs_dir / "law_dense.npy")
    uids = json.loads((runs_dir / "law_dense_meta.json").read_text())
    return emb, uids


def dense_search(query: str, emb: np.ndarray, uids: list[str], encoder: DenseEncoder, top_k: int = 30) -> list[tuple[str, float]]:
    q = encoder.encode_one(query)
    scores = emb @ q
    idx = np.argsort(-scores)[:top_k]
    return [(uids[i], float(scores[i])) for i in idx]
