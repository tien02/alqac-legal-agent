#!/usr/bin/env python3
"""Iterate ALQAC 2026 public-test set: run mini-swe-agent per case, score, write metrics.json."""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from minisweagent.agents import get_agent  # noqa: E402
from minisweagent.models import get_model  # noqa: E402
from minisweagent.config import get_config_from_spec  # noqa: E402
from minisweagent.utils.serialize import recursive_merge  # noqa: E402

from eval.metrics import (  # noqa: E402
    CaseResult,
    aggregate,
    build_law_name_map,
    parse_gold_law_refs,
    parse_pred_law_refs,
    parse_submission_verdict,
    score_case,
)
from retrieval.alqac_env import AlqacEnv  # noqa: E402
from retrieval.case_api_client import CaseApiClient  # noqa: E402


DATA_PATH = ROOT / "Data" / "ALQAC2026_public_test.json"
CORPUS_PATH = ROOT / "Data" / "corpus_law_pub.json"
CONFIG_PATH = (ROOT / "configs" / "alqac.yaml").resolve()
RUNS_DIR = ROOT / "runs"


def load_cases() -> list[dict]:
    return json.loads(DATA_PATH.read_text())


def build_base_config() -> dict:
    return get_config_from_spec(str(CONFIG_PATH))


def _filter_env_kwargs(env_cfg: dict) -> dict:
    return {k: v for k, v in env_cfg.items() if k != "environment_class"}


def _count_unique_chunks(case_id: str) -> int:
    try:
        client = CaseApiClient(runs_dir=str(RUNS_DIR))
        with sqlite3.connect(client.db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(DISTINCT chunk_id) FROM calls WHERE case_id = ?", (case_id,)
            ).fetchone()
        return int(row[0] or 0)
    except Exception:
        return 0


def run_one_case(case: dict, base_config: dict, gold_law_map: dict[str, str]) -> dict:
    case_id = case["case_id"]
    config = recursive_merge(
        base_config,
        {"agent": {"output_path": RUNS_DIR / f"{case_id}.traj.json"}},
    )

    model = get_model(config=config.get("model", {}))
    env = AlqacEnv(case=case, runs_dir=str(RUNS_DIR), **_filter_env_kwargs(config.get("environment", {})))
    agent = get_agent(model, env, config.get("agent", {}), default_type="default")

    t0 = time.time()
    extra = agent.run(task="")
    elapsed = time.time() - t0

    submission = extra.get("submission", "")
    pred_verdict, pred_law_refs_raw, evidence_chunk_ids = parse_submission_verdict(submission)
    pred_law_refs = parse_pred_law_refs(pred_law_refs_raw)
    gold_law_refs = parse_gold_law_refs(case.get("related_law_provisions", ""), fallback_map=gold_law_map)
    n_unique = _count_unique_chunks(case_id)

    result = CaseResult(
        case_id=case_id,
        gold_verdict=case.get("verdict_label", ""),
        pred_verdict=pred_verdict,
        gold_law_refs=gold_law_refs,
        pred_law_refs=pred_law_refs,
        n_unique_chunks=n_unique,
        gold_segments_estimate=5,
    )
    scored = score_case(result)
    scored["elapsed_seconds"] = round(elapsed, 1)
    scored["evidence_chunk_ids"] = evidence_chunk_ids
    scored["pred_law_refs"] = sorted(
        [{"law_id": k, "aid": v} for k, v in pred_law_refs], key=lambda x: (x["law_id"], x["aid"])
    )
    scored["gold_law_refs"] = sorted(
        [{"law_id": k, "aid": v} for k, v in gold_law_refs], key=lambda x: (x["law_id"], x["aid"])
    )
    return scored


def main(max_cases: int | None = None, case_filter: str | None = None) -> dict:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    cases = load_cases()
    if case_filter:
        cases = [c for c in cases if c["case_id"] == case_filter]
    if max_cases:
        cases = cases[:max_cases]

    base_config = build_base_config()
    gold_law_map = build_law_name_map(CORPUS_PATH)
    print(f"Loaded {len(cases)} cases. gold_law_map keys: {list(gold_law_map)[:5]}")

    results: list[dict] = []
    pred_path = RUNS_DIR / "predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as fh:
        for i, case in enumerate(cases, 1):
            print(f"[{i}/{len(cases)}] case_id={case['case_id']}", flush=True)
            try:
                r = run_one_case(case, base_config, gold_law_map)
            except Exception as e:
                import traceback

                traceback.print_exc()
                r = {
                    "case_id": case["case_id"],
                    "gold_verdict": case.get("verdict_label", ""),
                    "pred_verdict": "",
                    "verdict_correct": 0,
                    "law_precision": 0.0,
                    "law_recall": 0.0,
                    "law_f1": 0.0,
                    "n_unique_chunks": 0,
                    "api_penalty": 1.0,
                    "elapsed_seconds": 0,
                    "error": str(e),
                }
            results.append(r)
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()

    agg = aggregate(results)
    (RUNS_DIR / "metrics.json").write_text(json.dumps(agg, ensure_ascii=False, indent=2))
    print(json.dumps(agg, ensure_ascii=False, indent=2))
    return agg


if __name__ == "__main__":
    import typer

    typer.run(main)
