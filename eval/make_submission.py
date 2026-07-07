#!/usr/bin/env python3
"""Merge one or more predictions.jsonl into ALQAC submission format.

Usage:
    uv run python -m eval.make_submission runs/<run_a> runs/<run_b> -o submission.json

Emits JSON array with schema:
    [
      {
        "case_id": "case_4101",
        "prediction": "A_WIN",
        "case_evidence": ["case_4101_chunk_3"],
        "law_evidence": [{"law_id": "47/2010/QH12", "aid": 270}]
      },
      ...
    ]

Model emits `{"law_id":"...", "article_number": 584}` in law_refs. Our pipeline
stores that as `{"law_id":"...", "aid": 584}` in predictions.jsonl (see
eval/run_dev_set.py::run_one_case). Real corpus aid for that article may be
different (e.g. 53354). This script converts article_number -> corpus aid via
Data/corpus_law_pub.json. Refs that don't map to any corpus aid are dropped.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = ROOT / "Data" / "corpus_law_pub.json"


def _build_article_number_to_aid_map() -> dict[tuple[str, int], int]:
    corpus = json.loads(CORPUS_PATH.read_text())
    m: dict[tuple[str, int], int] = {}
    for law in corpus:
        law_id = law["law_id"]
        for idx, art in enumerate(law["content"], start=1):
            m[(law_id, idx)] = int(art["aid"])
    return m


def _valid_aids_by_law() -> dict[str, set[int]]:
    corpus = json.loads(CORPUS_PATH.read_text())
    out: dict[str, set[int]] = {}
    for law in corpus:
        out[law["law_id"]] = {int(a["aid"]) for a in law["content"]}
    return out


def _resolve_law_refs(
    refs: list,
    anum_to_aid: dict[tuple[str, int], int],
    valid_aids: dict[str, set[int]],
    case_id: str | None = None,
) -> list[dict]:
    """Resolve either:
      - list[int]: ref_ids looked up via runs/law_refs.db (scoped by case_id)
      - list[dict]: legacy {law_id, article_number|aid} shape.
    """
    resolved: list[dict] = []
    seen: set[tuple[str, int]] = set()
    if not refs:
        return resolved

    if case_id and all(isinstance(x, (int, float, str)) and str(x).lstrip("-").isdigit() for x in refs):
        try:
            from retrieval.law_refs_registry import LawRefsRegistry
        except Exception:
            return resolved
        reg = LawRefsRegistry()
        for x in refs:
            try:
                ref = reg.lookup(case_id, int(x))
            except (ValueError, TypeError):
                continue
            if ref is None or ref.aid is None:
                print(f"  drop ref (unknown ref_id): case={case_id} ref_id={x}", file=sys.stderr)
                continue
            key = (ref.law_id, int(ref.aid))
            if key in seen:
                continue
            seen.add(key)
            resolved.append({"law_id": ref.law_id, "aid": int(ref.aid)})
        return resolved

    # Legacy dict shape.
    for r in refs:
        if not isinstance(r, dict):
            continue
        law_id = str(r.get("law_id", "")).strip()
        val = r.get("article_number", r.get("aid"))
        if not law_id or val is None:
            continue
        try:
            num = int(val)
        except (TypeError, ValueError):
            continue
        aid = anum_to_aid.get((law_id, num))
        if aid is None and num in valid_aids.get(law_id, set()):
            aid = num
        if aid is None:
            print(f"  drop ref (unresolvable): law_id={law_id} num={num}", file=sys.stderr)
            continue
        key = (law_id, aid)
        if key in seen:
            continue
        seen.add(key)
        resolved.append({"law_id": law_id, "aid": aid})
    return resolved


def _load_run_predictions(run_dir: Path) -> list[dict]:
    p = run_dir / "predictions.jsonl"
    if not p.exists():
        sys.exit(f"predictions.jsonl not found in {run_dir}")
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def main(run_dirs: list[str], output: str = "submission.json") -> None:
    if not run_dirs:
        sys.exit("provide at least one run_dir")

    anum_to_aid = _build_article_number_to_aid_map()
    valid_aids = _valid_aids_by_law()

    merged: dict[str, dict] = {}
    for rd in run_dirs:
        run_dir = Path(rd)
        if not run_dir.is_absolute():
            run_dir = ROOT / run_dir
        preds = _load_run_predictions(run_dir)
        print(f"[{run_dir.name}] {len(preds)} predictions", file=sys.stderr)
        for r in preds:
            case_id = r.get("case_id")
            if not case_id:
                continue
            merged[case_id] = {
                "case_id": case_id,
                "prediction": r.get("pred_verdict", "") or "",
                "case_evidence": list(r.get("evidence_chunk_ids", []) or []),
                "law_evidence": _resolve_law_refs(
                    r.get("pred_law_refs", []), anum_to_aid, valid_aids, case_id=case_id
                ),
            }

    submission = sorted(merged.values(), key=lambda x: x["case_id"])
    out_path = Path(output)
    out_path.write_text(json.dumps(submission, ensure_ascii=False, indent=2))
    print(f"wrote {len(submission)} entries -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    import typer

    typer.run(main)
