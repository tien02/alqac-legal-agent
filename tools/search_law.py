#!/usr/bin/env python3
"""Hybrid law article search: BM25 + BGE-M3 + RRF + BGE-reranker-v2-m3.

CLI:
    python tools/search_law.py --query "bồi thường thiệt hại" --case-id case_4101 --top-k 5

Output (numbered plaintext, model-facing):
    1: <article content>...
    2: <article content>...

Each numbered entry maps to a stable (law_id, article_number, aid) tuple stored
in runs/law_refs.db, scoped per case_id. The model cites by integer only.
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
from retrieval.law_refs_registry import LawRefsRegistry  # noqa: E402

RUNS_DIR = ROOT / "runs"


def rrf_fuse(bm25_hits: list[tuple[str, float]], dense_hits: list[tuple[str, float]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for rank, (uid, _) in enumerate(bm25_hits):
        scores[uid] = scores.get(uid, 0.0) + 1.0 / (k + rank + 1)
    for rank, (uid, _) in enumerate(dense_hits):
        scores[uid] = scores.get(uid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


def main(
    query: str,
    case_id: str | None = None,
    top_k: int = 5,
    bm25_top: int = 30,
    dense_top: int = 30,
    fuse_top: int = 20,
    law_ids: list[str] | None = None,
) -> list[dict]:
    bm25_idx = BM25LawIndex.load(RUNS_DIR)
    emb, uids = load_dense(RUNS_DIR)
    encoder = DenseEncoder()

    law_id_set = set(law_ids) if law_ids else None

    bm25_hits = [
        (a.uid, s)
        for a, s in bm25_idx.search(query, top_k=bm25_top, law_ids=law_id_set)
    ]
    dense_hits = dense_search(query, emb, uids, encoder, top_k=dense_top)
    if law_id_set:
        uid_to_article = {a.uid: a for a in bm25_idx.articles}
        dense_hits = [
            (uid, s) for uid, s in dense_hits
            if uid in uid_to_article and uid_to_article[uid].law_id in law_id_set
        ]
    fused = rrf_fuse(bm25_hits, dense_hits)[:fuse_top]
    if not fused:
        return []

    reranker = Reranker()
    uid_to_article = {a.uid: a for a in bm25_idx.articles}
    cand_articles = [uid_to_article[uid] for uid, _ in fused]
    cand_texts = [a.content_Article for a in cand_articles]
    rr_scores = reranker.rerank(query, cand_texts)

    ranked = sorted(zip(cand_articles, rr_scores, strict=False), key=lambda x: x[1], reverse=True)[:top_k]

    registry = LawRefsRegistry() if case_id else None

    out: list[dict] = []
    for a, s in ranked:
        ref_id: int | None = None
        if registry is not None:
            ref = registry.assign(case_id, a.law_id, a.article_number)
            ref_id = ref.ref_id
        out.append({
            "ref_id": ref_id,
            "content_Article": a.content_Article,
            "score": float(s),
        })
    return out


def format_stdout(items: list[dict], query: str) -> str:
    if not items:
        return f"[no articles found for query: {query}]\n"
    lines = [f"# Query: {query}", ""]
    for item in items:
        rid = item.get("ref_id")
        content = item.get("content_Article", "").strip()
        header = f"{rid}:" if rid is not None else "-"
        lines.append(f"{header} {content}")
        lines.append("")
    lines.append("(Cite these articles by ref_id only, e.g. law_refs: [1, 3])")
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--query", required=True)
    p.add_argument("--case-id", default=None)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--law-id", action="append", default=None,
                   help="filter to specific law_id(s); repeat for multiple")
    p.add_argument("--json", action="store_true", help="emit raw JSON instead of numbered text")
    args = p.parse_args()
    out = main(args.query, case_id=args.case_id, top_k=args.top_k, law_ids=args.law_id)
    if args.json:
        print(json.dumps(out, ensure_ascii=False, indent=2))
    else:
        print(format_stdout(out, args.query))
