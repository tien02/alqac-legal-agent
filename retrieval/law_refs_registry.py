"""Per-case law-reference registry.

Assigns a stable integer `ref_id` per (case_id, law_id, article_number) tuple.
Model only cites the small integer; the (law_id, aid) pair is stored server-side
so the model literally cannot fabricate a citation.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = ROOT / "Data" / "corpus_law_pub.json"
DEFAULT_DB = ROOT / "runs" / "law_refs.db"

_lock = threading.Lock()


def _build_anum_to_aid() -> dict[tuple[str, int], int]:
    corpus = json.loads(CORPUS_PATH.read_text())
    m: dict[tuple[str, int], int] = {}
    for law in corpus:
        law_id = law["law_id"]
        for idx, art in enumerate(law["content"], start=1):
            m[(law_id, idx)] = int(art["aid"])
    return m


_ANUM_TO_AID: dict[tuple[str, int], int] = _build_anum_to_aid()


def article_aid(law_id: str, article_number: int) -> int | None:
    return _ANUM_TO_AID.get((str(law_id).strip(), int(article_number)))


@dataclass(frozen=True)
class LawRef:
    ref_id: int
    law_id: str
    article_number: int
    aid: int | None


class LawRefsRegistry:
    def __init__(self, db_path: Path | str = DEFAULT_DB):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with _lock, sqlite3.connect(self.db_path) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS refs (
                    case_id TEXT NOT NULL,
                    ref_id INTEGER NOT NULL,
                    law_id TEXT NOT NULL,
                    article_number INTEGER NOT NULL,
                    aid INTEGER,
                    ts REAL,
                    PRIMARY KEY (case_id, ref_id)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS refs_unique_tuple
                    ON refs(case_id, law_id, article_number);
                """
            )
            conn.commit()

    def assign(self, case_id: str, law_id: str, article_number: int) -> LawRef:
        """Return existing ref_id for the (case_id, law_id, article_number) tuple
        or allocate a new one. Idempotent — same input returns same ref_id."""
        law_id = str(law_id).strip()
        article_number = int(article_number)
        aid = article_aid(law_id, article_number)
        with _lock, sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT ref_id FROM refs WHERE case_id=? AND law_id=? AND article_number=?",
                (case_id, law_id, article_number),
            ).fetchone()
            if row:
                return LawRef(ref_id=int(row[0]), law_id=law_id,
                              article_number=article_number, aid=aid)
            next_id = int(
                (conn.execute("SELECT COALESCE(MAX(ref_id), 0) FROM refs WHERE case_id=?",
                              (case_id,)).fetchone() or [0])[0]
            ) + 1
            conn.execute(
                "INSERT INTO refs(case_id, ref_id, law_id, article_number, aid, ts) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (case_id, next_id, law_id, article_number, aid, time.time()),
            )
            conn.commit()
            return LawRef(ref_id=next_id, law_id=law_id,
                          article_number=article_number, aid=aid)

    def lookup(self, case_id: str, ref_id: int) -> LawRef | None:
        with _lock, sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT ref_id, law_id, article_number, aid FROM refs "
                "WHERE case_id=? AND ref_id=?",
                (case_id, int(ref_id)),
            ).fetchone()
        if not row:
            return None
        return LawRef(ref_id=int(row[0]), law_id=str(row[1]),
                      article_number=int(row[2]),
                      aid=int(row[3]) if row[3] is not None else None)

    def reset_case(self, case_id: str) -> None:
        with _lock, sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM refs WHERE case_id=?", (case_id,))
            conn.commit()
