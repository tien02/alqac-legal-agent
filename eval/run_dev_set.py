#!/usr/bin/env python3
"""Iterate ALQAC 2026 public-test set: run mini-swe-agent per case, score, write metrics.

Multi-run layout: outputs land under runs/<run_id>/. Registry line appended to
runs/registry.jsonl for cross-run comparison.
"""
from __future__ import annotations

import json
import shutil
import sqlite3
import sys
import time
from datetime import datetime
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
REGISTRY_PATH = RUNS_DIR / "registry.jsonl"


def load_cases() -> list[dict]:
    return json.loads(DATA_PATH.read_text())


def build_base_config() -> dict:
    return get_config_from_spec(str(CONFIG_PATH))


def _filter_env_kwargs(env_cfg: dict) -> dict:
    return {k: v for k, v in env_cfg.items() if k != "environment_class"}


def _count_unique_chunks(case_id: str, runs_dir: Path) -> int:
    try:
        client = CaseApiClient(runs_dir=str(runs_dir))
        with sqlite3.connect(client.db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(DISTINCT chunk_id) FROM calls WHERE case_id = ?", (case_id,)
            ).fetchone()
        return int(row[0] or 0)
    except Exception:
        return 0


def _slugify(s: str) -> str:
    keep = "abcdefghijklmnopqrstuvwxyz0123456789-_."
    return "".join(c if c.lower() in keep else "-" for c in s).strip("-")


def _derive_run_id(base_config: dict, run_name: str | None, prompt_tag: str | None) -> str:
    ts = datetime.now().strftime("%Y%m%dT%H%M")
    if run_name:
        return f"{ts}_{_slugify(run_name)}"
    model = _slugify(base_config.get("model", {}).get("model_name", "unknown").split("/")[-1])
    tag = _slugify(prompt_tag) if prompt_tag else "run"
    return f"{ts}_{model}_{tag}"


def run_one_case(case: dict, base_config: dict, gold_law_map: dict[str, str], run_dir: Path) -> dict:
    case_id = case["case_id"]
    traj_path = run_dir / "traj" / f"{case_id}.traj.json"
    config = recursive_merge(
        base_config,
        {"agent": {"output_path": traj_path}},
    )

    model = get_model(config=config.get("model", {}))
    env = AlqacEnv(case=case, runs_dir=str(run_dir), **_filter_env_kwargs(config.get("environment", {})))
    agent = get_agent(model, env, config.get("agent", {}), default_type="default")

    t0 = time.time()
    extra = agent.run(task="")
    elapsed = time.time() - t0

    submission = extra.get("submission", "")
    pred_verdict, pred_law_refs_raw, evidence_chunk_ids = parse_submission_verdict(submission)
    pred_law_refs = parse_pred_law_refs(pred_law_refs_raw)
    gold_law_refs = parse_gold_law_refs(case.get("related_law_provisions", ""), fallback_map=gold_law_map)
    n_unique = _count_unique_chunks(case_id, run_dir)

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


def _append_registry(entry: dict) -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    with REGISTRY_PATH.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main(
    max_cases: int | None = None,
    case_filter: str | None = None,
    run_name: str | None = None,
    prompt_tag: str | None = None,
    notes: str = "",
    model_name: str | None = None,
    api_base: str | None = None,
    start_idx: int = 0,
    end_idx: int | None = None,
) -> dict:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    cases = load_cases()
    if case_filter:
        cases = [c for c in cases if c["case_id"] == case_filter]
    cases = cases[start_idx:end_idx]
    if max_cases:
        cases = cases[:max_cases]

    base_config = build_base_config()
    if model_name:
        base_config.setdefault("model", {})["model_name"] = model_name
    if api_base:
        base_config.setdefault("model", {}).setdefault("model_kwargs", {})["api_base"] = api_base
    gold_law_map = build_law_name_map(CORPUS_PATH)

    run_id = _derive_run_id(base_config, run_name, prompt_tag)
    run_dir = RUNS_DIR / run_id
    (run_dir / "traj").mkdir(parents=True, exist_ok=True)

    shutil.copy(CONFIG_PATH, run_dir / "config_snapshot.yaml")

    print(f"run_id={run_id}")
    print(f"Loaded {len(cases)} cases. gold_law_map keys: {list(gold_law_map)[:5]}")

    results: list[dict] = []
    pred_path = run_dir / "predictions.jsonl"
    with pred_path.open("w", encoding="utf-8") as fh:
        for i, case in enumerate(cases, 1):
            print(f"[{i}/{len(cases)}] case_id={case['case_id']}", flush=True)
            try:
                r = run_one_case(case, base_config, gold_law_map, run_dir)
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
    (run_dir / "metrics.json").write_text(json.dumps(agg, ensure_ascii=False, indent=2))

    entry = {
        "run_id": run_id,
        "ts": datetime.now().isoformat(timespec="seconds"),
        "model": base_config.get("model", {}).get("model_name", ""),
        "prompt_tag": prompt_tag or "",
        "notes": notes,
        "step_limit": base_config.get("agent", {}).get("step_limit"),
        **agg,
    }
    _append_registry(entry)

    print(json.dumps(agg, ensure_ascii=False, indent=2))
    print(f"run_dir={run_dir}")
    return agg


if __name__ == "__main__":
    import typer

    typer.run(main)
