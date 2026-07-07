"""ALQAC 2026 case-evidence API client.

Endpoint: POST https://alqac-api.ngrok.pro/retrieve
Auth: X-API-Key header.
Rate limit: 1 request / 5s (server enforces, returns 429 on excess).
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://alqac-api.ngrok.pro/retrieve"
DEFAULT_TIMEOUT = 30.0
RATE_LIMIT_SECONDS = 5.0
DEFAULT_GOLD_ESTIMATE = 10
MAX_CALLS_PER_CASE = 5 * DEFAULT_GOLD_ESTIMATE

_runs_lock = threading.Lock()


@dataclass(frozen=True)
class CaseChunk:
    chunk_id: str
    text: str
    score: float


class CaseApiClient:
    def __init__(
        self,
        api_key: str | None = None,
        runs_dir: Path | str = "runs",
        rate_limit: float = RATE_LIMIT_SECONDS,
        max_calls_per_case: int = MAX_CALLS_PER_CASE,
    ):
        self.api_key = api_key or os.environ.get("ALQAC_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("ALQAC_API_KEY not set. Put it in .env or export it.")
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.runs_dir / "case_calls.db"
        self.rate_limit = rate_limit
        self.max_calls_per_case = max_calls_per_case
        self._init_db()
        self._last_call_ts = 0.0
        self._lock = threading.Lock()

    def _init_db(self) -> None:
        with _runs_lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS calls (
                    case_id  TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    query    TEXT,
                    ts       REAL NOT NULL,
                    score    REAL,
                    PRIMARY KEY (case_id, chunk_id, query)
                )
                """
            )
            conn.commit()

    def _count_unique(self, case_id: str) -> int:
        with _runs_lock, sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(DISTINCT chunk_id) FROM calls WHERE case_id = ?", (case_id,)
            ).fetchone()
            return int(row[0] or 0)

    def _record(self, case_id: str, chunk: CaseChunk, query: str) -> None:
        with _runs_lock, sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO calls(case_id, chunk_id, query, ts, score) VALUES (?, ?, ?, ?, ?)",
                (case_id, chunk.chunk_id, query, time.time(), chunk.score),
            )
            conn.commit()

    def retrieve(self, query: str, case_id: str, *, force: bool = False) -> CaseChunk:
        n_unique = self._count_unique(case_id)
        if not force and n_unique >= self.max_calls_per_case:
            return CaseChunk(
                chunk_id="BUDGET_EXHAUSTED",
                text=f"Per-case budget exhausted: {n_unique} unique chunks (cap={self.max_calls_per_case}).",
                score=0.0,
            )

        with self._lock:
            wait = self.rate_limit - (time.time() - self._last_call_ts)
            if wait > 0:
                time.sleep(wait)
            self._last_call_ts = time.time()

        resp = httpx.post(
            API_URL,
            json={"query": query, "case_id": case_id},
            headers={"X-API-Key": self.api_key, "Content-Type": "application/json"},
            timeout=DEFAULT_TIMEOUT,
        )
        if resp.status_code == 429:
            time.sleep(self.rate_limit)
            return self.retrieve(query, case_id, force=force)
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results") or []
        top = results[0] if results else {}
        chunk = CaseChunk(
            chunk_id=str(top.get("chunk_id", "")),
            text=str(top.get("text", "")),
            score=float(top.get("score", 0.0)),
        )
        self._record(case_id, chunk, query)
        return chunk

    def reset_case(self, case_id: str) -> None:
        """Clear call log for a case. Call before each fresh run."""
        with _runs_lock, sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM calls WHERE case_id = ?", (case_id,))
            conn.commit()
