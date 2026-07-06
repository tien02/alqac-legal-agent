#!/usr/bin/env python3
"""Side-by-side diff of two runs.

Usage: uv run python -m eval.compare_runs <run_id_a> <run_id_b>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"


def _load(run_id: str) -> tuple[dict, dict[str, dict]]:
    run_dir = RUNS / run_id
    if not run_dir.exists():
        sys.exit(f"run_id not found: {run_dir}")
    metrics = json.loads((run_dir / "metrics.json").read_text())
    preds = {}
    with (run_dir / "predictions.jsonl").open() as fh:
        for line in fh:
            row = json.loads(line)
            preds[row["case_id"]] = row
    return metrics, preds


def main(run_a: str, run_b: str) -> None:
    ma, pa = _load(run_a)
    mb, pb = _load(run_b)

    from rich.console import Console
    from rich.table import Table

    con = Console()

    agg = Table(title="Aggregate metrics", header_style="bold cyan")
    agg.add_column("metric")
    agg.add_column(run_a)
    agg.add_column(run_b)
    agg.add_column("Δ", justify="right")
    for k in ["n_cases", "accuracy_4class", "law_f1", "avg_api_penalty", "combined_score"]:
        va, vb = ma.get(k, 0), mb.get(k, 0)
        delta = ""
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            d = vb - va
            style = "green" if d > 0 else ("red" if d < 0 else "")
            delta = f"[{style}]{d:+.3f}[/]" if style else f"{d:+.3f}"
        agg.add_row(k, f"{va}", f"{vb}", delta)
    con.print(agg)

    flips = Table(title="Per-case diffs (verdict or law_f1)", header_style="bold cyan")
    flips.add_column("case_id")
    flips.add_column("gold")
    flips.add_column(f"pred_a")
    flips.add_column(f"pred_b")
    flips.add_column("f1_a", justify="right")
    flips.add_column("f1_b", justify="right")

    ids = sorted(set(pa) | set(pb))
    n_diff = 0
    for cid in ids:
        ra, rb = pa.get(cid, {}), pb.get(cid, {})
        va, vb = ra.get("pred_verdict", ""), rb.get("pred_verdict", "")
        f1a, f1b = ra.get("law_f1", 0.0), rb.get("law_f1", 0.0)
        if va == vb and abs(f1a - f1b) < 1e-9:
            continue
        n_diff += 1
        flips.add_row(
            cid,
            ra.get("gold_verdict", rb.get("gold_verdict", "")),
            va, vb,
            f"{f1a:.2f}", f"{f1b:.2f}",
        )
    con.print(flips)
    con.print(f"[dim]{n_diff} of {len(ids)} cases differ[/]")


if __name__ == "__main__":
    import typer

    typer.run(main)
