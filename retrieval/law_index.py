"""BM25 index over corpus_law_pub.json articles."""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi
from underthesea import word_tokenize

_PUNCT = ".,;:!?\"'`()[]{}|="


@dataclass(frozen=True)
class LawArticle:
    law_id: str
    aid: int               # corpus-internal ID (e.g. 53354)
    article_number: int    # legal article number (e.g. 584) = idx_in_law + 1
    content_Article: str

    @property
    def uid(self) -> str:
        return f"{self.law_id}::{self.aid}"


def tokenize_vi(text: str) -> list[str]:
    """Underthesea word-segmentation. Merges Vietnamese compound words ("hợp đồng" -> "hợp_đồng")."""
    toks = word_tokenize(text.lower(), format="text").split()
    return [t.strip(_PUNCT) for t in toks if len(t.strip(_PUNCT)) > 1]


class BM25LawIndex:
    def __init__(self, bm25: BM25Okapi, articles: list[LawArticle]):
        self.bm25 = bm25
        self.articles = articles
        self.uid_to_idx = {a.uid: i for i, a in enumerate(articles)}

    def search(
        self, query: str, top_k: int = 30, law_ids: set[str] | None = None,
    ) -> list[tuple[LawArticle, float]]:
        scores = self.bm25.get_scores(tokenize_vi(query))
        indexed = list(enumerate(scores))
        if law_ids:
            indexed = [(i, s) for i, s in indexed if self.articles[i].law_id in law_ids]
        ranked = sorted(indexed, key=lambda x: x[1], reverse=True)[:top_k]
        return [(self.articles[i], float(s)) for i, s in ranked if s > 0]

    @classmethod
    def load(cls, runs_dir: Path | str) -> "BM25LawIndex":
        runs_dir = Path(runs_dir)
        with open(runs_dir / "law_bm25.pkl", "rb") as f:
            bm25 = pickle.load(f)
        import json

        articles = [LawArticle(**a) for a in json.loads((runs_dir / "law_meta.json").read_text())]
        return cls(bm25, articles)


def build_from_corpus(corpus_path: Path | str, runs_dir: Path | str) -> BM25LawIndex:
    """Build BM25 from Data/corpus_law_pub.json and persist to runs_dir."""
    import json

    corpus = json.loads(Path(corpus_path).read_text())
    articles: list[LawArticle] = []
    for law in corpus:
        law_id = law["law_id"]
        for idx_in_law, art in enumerate(law["content"], start=1):
            articles.append(
                LawArticle(
                    law_id=law_id,
                    aid=int(art["aid"]),
                    article_number=idx_in_law,
                    content_Article=art["content_Article"],
                )
            )
    corpus_tokens = [tokenize_vi(a.content_Article) for a in articles]
    bm25 = BM25Okapi(corpus_tokens)

    runs_dir = Path(runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    with open(runs_dir / "law_bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)
    (runs_dir / "law_meta.json").write_text(
        json.dumps(
            [
                {
                    "law_id": a.law_id,
                    "aid": a.aid,
                    "article_number": a.article_number,
                    "content_Article": a.content_Article,
                }
                for a in articles
            ],
            ensure_ascii=False,
        )
    )
    return BM25LawIndex(bm25, articles)
