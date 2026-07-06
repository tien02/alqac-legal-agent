#!/usr/bin/env python3
"""Tabular view of runs/registry.jsonl."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "runs" / "registry.jsonl"


def _fmt(v, digits: int = 3) -> str:
    if isinstance(v, float):
        return f"{v:.{digits}f}"
    return str(v)


def main() -> None:
    if not REGISTRY.exists():
        print("no runs — registry.jsonl not found", file=sys.stderr)
        return
    rows = [json.loads(l) for l in REGISTRY.read_text().splitlines() if l.strip()]
    if not rows:
        print("registry empty")
        return

    from rich.console import Console
    from rich.table import Table

    tbl = Table(show_header=True, header_style="bold cyan")
    tbl.add_column("run_id", overflow="fold")
    tbl.add_column("model")
    tbl.add_column("tag")
    tbl.add_column("step", justify="right")
    tbl.add_column("n", justify="right")
    tbl.add_column("acc", justify="right")
    tbl.add_column("law_f1", justify="right")
    tbl.add_column("api_pen", justify="right")
    tbl.add_column("combined", justify="right", style="bold")
    tbl.add_column("notes", overflow="fold")

    for r in rows:
        tbl.add_row(
            r.get("run_id", ""),
            r.get("model", ""),
            r.get("prompt_tag", ""),
            _fmt(r.get("step_limit")),
            _fmt(r.get("n_cases")),
            _fmt(r.get("accuracy_4class"), 2),
            _fmt(r.get("law_f1"), 3),
            _fmt(r.get("avg_api_penalty"), 2),
            _fmt(r.get("combined_score"), 3),
            r.get("notes", ""),
        )

    Console().print(tbl)


if __name__ == "__main__":
    main()
