"""Per-case budget tracker. Reads SQLite log produced by case_api_client.

LLatrieval verify-then-continue is in eval/run_dev_set.py via STOP/CONTINUE side-channel;
this module exposes stats + penalty formula only.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CaseStats:
    case_id: str
    n_calls: int
    n_unique_chunks: int
    gold_segments_estimate: int


def api_efficiency_penalty(n_unique_chunks: int, gold_segments: int) -> float:
    """ALQAC penalty: full credit at N<=2G, linear decay to 0 at N=5G."""
    if n_unique_chunks <= 2 * gold_segments:
        return 0.0
    if n_unique_chunks >= 5 * gold_segments:
        return 1.0
    return (n_unique_chunks - 2 * gold_segments) / (3 * gold_segments)


class BudgetTracker:
    def __init__(self, db_path: Path | str = "runs/case_calls.db"):
        self.db_path = str(db_path)

    def stats(self, case_id: str, gold_segments_estimate: int | None = None) -> CaseStats:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT
                  COUNT(*) AS n_calls,
                  COUNT(DISTINCT chunk_id) AS n_unique
                FROM calls WHERE case_id = ?
                """,
                (case_id,),
            ).fetchone()
        n_calls = int(row[0] or 0)
        n_unique = int(row[1] or 0)
        if gold_segments_estimate is None:
            gold_segments_estimate = max(1, 5)
        return CaseStats(
            case_id=case_id,
            n_calls=n_calls,
            n_unique_chunks=n_unique,
            gold_segments_estimate=gold_segments_estimate,
        )

    def penalty(self, case_id: str, gold_segments_estimate: int | None = None) -> float:
        s = self.stats(case_id, gold_segments_estimate)
        return api_efficiency_penalty(s.n_unique_chunks, s.gold_segments_estimate)
