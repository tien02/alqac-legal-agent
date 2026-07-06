#!/usr/bin/env python3
"""CLI wrapper for the ALQAC 2026 case-evidence API.

Usage:
    python tools/search_case.py --query "yêu cầu bồi thường" --case-id public_test_0
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrieval.case_api_client import CaseApiClient  # noqa: E402


def main(query: str, case_id: str, runs_dir: Path | str = ROOT / "runs") -> dict:
    client = CaseApiClient(runs_dir=runs_dir)
    chunk = client.retrieve(query=query, case_id=case_id)
    return {"chunk_id": chunk.chunk_id, "text": chunk.text, "score": chunk.score}


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--query", required=True)
    p.add_argument("--case-id", required=True)
    args = p.parse_args()
    out = main(args.query, args.case_id)
    print(json.dumps(out, ensure_ascii=False, indent=2))
