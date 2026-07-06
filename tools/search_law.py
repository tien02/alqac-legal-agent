#!/usr/bin/env python3
"""Hybrid law article search: BM25 + BGE-M3 + RRF + BGE-reranker-v2-m3.

CLI:
    python tools/search_law.py --query "bồi thường thiệt hại" --top-k 5
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval.embeddings import DenseEncoder, Reranker, dense_search, load_dense  # noqa: E402
from retrieval.law_index import BM25LawIndex  # noqa: E402

RUNS_DIR = ROOT / "runs"


def rrf_fuse(bm25_hits: list[tuple[str, float]], dense_hits: list[tuple[str, float]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for rank, (uid, _) in enumerate(bm25_hits):
        scores[uid] = scores.get(uid, 0.0) + 1.0 / (k + rank + 1)
    for rank, (uid, _) in enumerate(dense_hits):
        scores[uid] = scores.get(uid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def main(query: str, top_k: int = 5, bm25_top: int = 30, dense_top: int = 30, fuse_top: int = 20) -> list[dict]:
    bm25_idx = BM25LawIndex.load(RUNS_DIR)
    emb, uids = load_dense(RUNS_DIR)
    encoder = DenseEncoder()

    bm25_hits = [(a.uid, s) for a, s in bm25_idx.search(query, top_k=bm25_top)]
    dense_hits = dense_search(query, emb, uids, encoder, top_k=dense_top)
    fused = rrf_fuse(bm25_hits, dense_hits)[:fuse_top]
    if not fused:
        return []

    reranker = Reranker()
    uid_to_article = {a.uid: a for a in bm25_idx.articles}
    cand_articles = [uid_to_article[uid] for uid, _ in fused]
    cand_texts = [a.content_Article for a in cand_articles]
    rr_scores = reranker.rerank(query, cand_texts)

    ranked = sorted(zip(cand_articles, rr_scores, strict=False), key=lambda x: x[1], reverse=True)[:top_k]
    return [
        {
            "law_id": a.law_id,
            "article_number": a.article_number,
            "content_Article": a.content_Article,
            "score": float(s),
        }
        for a, s in ranked
    ]


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--query", required=True)
    p.add_argument("--top-k", type=int, default=5)
    args = p.parse_args()
    out = main(args.query, top_k=args.top_k)
    print(json.dumps(out, ensure_ascii=False, indent=2))
