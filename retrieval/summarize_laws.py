#!/usr/bin/env python3
"""Build a compact taxonomy of the 18 laws in Data/corpus_law_pub.json.

For each law, ask Qwen3.5 for a 3-4 sentence description covering:
  - Scope / subject matter (what area of law)
  - Typical disputes it applies to
  - 5-10 keywords

Output goes to runs/law_taxonomy.json — consumed by the agent's Phase 1
(decompose which laws apply to a case) and by tools/search_law.py --law-id
scoped retrieval.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = ROOT / "Data" / "corpus_law_pub.json"
OUT_PATH = ROOT / "runs" / "law_taxonomy.json"

ENDPOINTS = [
    "http://localhost:8001/v1/chat/completions",
    "http://localhost:8002/v1/chat/completions",
]
API_KEY = "local-dev-key"
MODEL = "qwen3.5-9b-gguf"

SYSTEM = (
    "Bạn là chuyên gia luật Việt Nam. Đọc law_id + trích một vài điều luật "
    "đầu tiên, rồi mô tả bộ luật này bằng tiếng Việt gồm 3-4 câu ngắn: "
    "(1) phạm vi & chủ đề pháp lý chính; (2) các loại tranh chấp / vấn đề "
    "thường viện dẫn bộ luật này; (3) 5-10 từ khoá quan trọng (cách nhau "
    "bằng dấu phẩy). Trả về JSON KHÔNG có lời mở đầu, đúng format:\n"
    "{\"description\": \"...\", \"keywords\": [\"...\", \"...\"]}"
)


def build_prompt(law_id: str, articles: list[dict]) -> str:
    head = "\n\n".join(
        f"[Điều {i+1}] {a['content_Article'][:400]}"
        for i, a in enumerate(articles[:6])
    )
    return f"law_id: {law_id}\nsố điều: {len(articles)}\n\nTrích 6 điều đầu:\n{head}"


def summarize_law(client: httpx.Client, endpoint: str, law_id: str, articles: list[dict]) -> dict:
    r = client.post(
        endpoint,
        json={
            "model": MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": build_prompt(law_id, articles)},
            ],
            "max_tokens": 400,
            "temperature": 0.2,
            "top_p": 0.9,
            "chat_template_kwargs": {"enable_thinking": False},
        },
        headers={"Authorization": f"Bearer {API_KEY}"},
        timeout=90.0,
    )
    r.raise_for_status()
    text = (r.json()["choices"][0]["message"].get("content") or "").strip()
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return {"description": text, "keywords": []}


def main() -> None:
    corpus = json.loads(CORPUS_PATH.read_text())
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    existing: dict[str, dict] = {}
    if OUT_PATH.exists():
        try:
            for e in json.loads(OUT_PATH.read_text()):
                existing[e["law_id"]] = e
            print(f"resumed {len(existing)} law summaries")
        except Exception as e:
            print(f"[warn] resume failed: {e}")

    out: list[dict] = []
    with httpx.Client() as client:
        for i, law in enumerate(corpus):
            law_id = law["law_id"]
            if law_id in existing and existing[law_id].get("description"):
                out.append(existing[law_id])
                print(f"  [{i+1}/{len(corpus)}] {law_id} — skip (cached)")
                continue
            ep = ENDPOINTS[i % len(ENDPOINTS)]
            t0 = time.time()
            summary = summarize_law(client, ep, law_id, law["content"])
            entry = {
                "law_id": law_id,
                "article_count": len(law["content"]),
                "first_article_head": law["content"][0]["content_Article"][:200],
                "description": summary.get("description", "").strip(),
                "keywords": summary.get("keywords", []),
            }
            out.append(entry)
            OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2))
            print(f"  [{i+1}/{len(corpus)}] {law_id} ({time.time()-t0:.1f}s) -> {entry['description'][:80]}...")

    OUT_PATH.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nwrote {len(out)} law entries -> {OUT_PATH}")


if __name__ == "__main__":
    main()
