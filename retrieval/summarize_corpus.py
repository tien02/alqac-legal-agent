#!/usr/bin/env python3
"""Summarize every article in Data/corpus_law_pub.json via Qwen3.5 (llama.cpp).

Idempotent: resumes an existing runs/corpus_summarized.json if present, skipping
articles that already have a `summary` field.

Parallelizes across both llama.cpp endpoints (:8001, :8002) so both GPUs work.

Output shape (same as input plus `summary`):
    [
      {
        "law_id": "...",
        "content": [
          {"aid": 270, "content_Article": "...", "summary": "..."},
          ...
        ]
      }
    ]
"""
from __future__ import annotations

import json
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = ROOT / "Data" / "corpus_law_pub.json"
OUT_PATH = ROOT / "runs" / "corpus_summarized.json"

ENDPOINTS = [
    "http://localhost:8001/v1/chat/completions",
    "http://localhost:8002/v1/chat/completions",
]
API_KEY = "local-dev-key"
MODEL = "qwen3.5-9b-gguf"

SYSTEM = (
    "Bạn là chuyên gia luật Việt Nam. Đọc điều luật và tóm tắt bằng 1-2 câu "
    "tiếng Việt. Nêu rõ: (1) chủ đề pháp lý cốt lõi (ví dụ: bồi thường thiệt "
    "hại, án phí, thẩm quyền, hợp đồng, quyền sử dụng đất, thi hành án); "
    "(2) đối tượng áp dụng hoặc điều kiện chính; (3) từ khóa quan trọng. "
    "KHÔNG viết lời mở đầu — trả về TRỰC TIẾP câu tóm tắt."
)

_lock = threading.Lock()


def summarize_one(client: httpx.Client, endpoint: str, content: str) -> str:
    r = client.post(
        endpoint,
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": f"Điều luật:\n{content}\n\nTóm tắt:"},
            ],
            "max_tokens": 200,
            "temperature": 0.2,
            "top_p": 0.9,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        headers={"Authorization": f"Bearer {API_KEY}"},
        timeout=60.0,
    )
    r.raise_for_status()
    data = r.json()
    return (data["choices"][0]["message"].get("content") or "").strip()


def load_or_init() -> list[dict]:
    """Return corpus with existing summaries preserved."""
    corpus = json.loads(CORPUS_PATH.read_text())
    if OUT_PATH.exists():
        try:
            prior = json.loads(OUT_PATH.read_text())
            prior_lookup: dict[tuple[str, int], str] = {}
            for law in prior:
                for art in law.get("content", []):
                    s = str(art.get("summary") or "").strip()
                    if s:
                        prior_lookup[(law["law_id"], int(art["aid"]))] = s
            for law in corpus:
                for art in law["content"]:
                    key = (law["law_id"], int(art["aid"]))
                    if key in prior_lookup:
                        art["summary"] = prior_lookup[key]
            print(f"resumed {len(prior_lookup)} prior summaries")
        except Exception as e:
            print(f"[warn] could not load prior summaries: {e}")
    return corpus


def pending_articles(corpus: list[dict]) -> list[tuple[int, int, str]]:
    """(law_idx, art_idx, content) for articles without summary."""
    out = []
    for li, law in enumerate(corpus):
        for ai, art in enumerate(law["content"]):
            if not str(art.get("summary") or "").strip():
                out.append((li, ai, str(art["content_Article"])))
    return out


def worker(job_q: "queue.Queue", corpus: list[dict], endpoint: str, done: dict) -> None:
    with httpx.Client() as client:
        while True:
            try:
                li, ai, content = job_q.get(timeout=1.0)
            except queue.Empty:
                return
            try:
                s = summarize_one(client, endpoint, content)
            except Exception as e:
                s = ""
                print(f"  [err] {endpoint} law={li} art={ai}: {e}", flush=True)
            with _lock:
                corpus[li]["content"][ai]["summary"] = s
                done["n"] += 1
                if done["n"] % 25 == 0:
                    OUT_PATH.write_text(json.dumps(corpus, ensure_ascii=False, indent=2))
                    print(f"  progress {done['n']}/{done['total']}", flush=True)
            job_q.task_done()


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    corpus = load_or_init()
    pending = pending_articles(corpus)
    total = sum(len(l["content"]) for l in corpus)
    print(f"corpus size: {total} articles, pending: {len(pending)}")

    if not pending:
        OUT_PATH.write_text(json.dumps(corpus, ensure_ascii=False, indent=2))
        print(f"all summarized — wrote {OUT_PATH}")
        return

    q: queue.Queue = queue.Queue()
    for item in pending:
        q.put(item)

    done = {"n": total - len(pending), "total": total}

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(ENDPOINTS)) as pool:
        for ep in ENDPOINTS:
            pool.submit(worker, q, corpus, ep, done)
    q.join()

    OUT_PATH.write_text(json.dumps(corpus, ensure_ascii=False, indent=2))
    print(f"done: {done['n']}/{done['total']} in {time.time()-t0:.0f}s -> {OUT_PATH}")


if __name__ == "__main__":
    main()
