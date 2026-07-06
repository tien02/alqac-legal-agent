#!/usr/bin/env python3
"""One-shot: build BM25 + BGE-M3 dense index over corpus_law_pub.json."""
from __future__ import annotations

import time
from pathlib import Path

from retrieval.embeddings import DenseEncoder, save_dense
from retrieval.law_index import build_from_corpus

DATA_DEFAULT = Path(__file__).resolve().parent.parent / "Data" / "corpus_law_pub.json"
RUNS_DEFAULT = Path(__file__).resolve().parent.parent / "runs"


def main(data_path: Path = DATA_DEFAULT, runs_dir: Path = RUNS_DEFAULT) -> None:
    print(f"[1/3] Building BM25 from {data_path} ...")
    bm25_idx = build_from_corpus(data_path, runs_dir)
    print(f"     indexed {len(bm25_idx.articles)} articles across {len({a.law_id for a in bm25_idx.articles})} laws")

    print(f"[2/3] Encoding {len(bm25_idx.articles)} articles with BGE-M3 ...")
    t0 = time.time()
    encoder = DenseEncoder()
    contents = [a.content_Article for a in bm25_idx.articles]
    emb = encoder.encode(contents)
    uids = [a.uid for a in bm25_idx.articles]
    save_dense(emb, uids, runs_dir)
    print(f"     dense shape={emb.shape} dtype={emb.dtype} elapsed={time.time() - t0:.1f}s")

    print(f"[3/3] Saved index to {runs_dir}")
    for name in ("law_bm25.pkl", "law_meta.json", "law_dense.npy", "law_dense_meta.json"):
        p = runs_dir / name
        print(f"     - {name} ({p}, exists={p.exists()})")

    from retrieval.embeddings import dense_search

    hits = dense_search("bồi thường thiệt hại do xâm phạm sức khỏe", emb, uids, encoder, top_k=5)
    print("     smoke dense_search top-5:", hits[:3])


if __name__ == "__main__":
    import typer

    typer.run(lambda data: main(Path(data), RUNS_DEFAULT))
