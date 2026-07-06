# ALQAC 2026 — Legal Case Outcome Prediction

Budget-aware, multi-turn Vietnamese legal-QA harness for
[ALQAC 2026](https://sites.google.com/view/alqac2026/home) Task 1.

- **Harness**: [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) (unmodified upstream).
- **Model server**: vLLM, OpenAI-compatible endpoint. Default `Qwen/Qwen2.5-7B-Instruct`
  on tensor-parallel-size 2 (fits 2× 12 GB cards).
- **Retrieval**: BM25 + dense (`bkai-foundation-models/vietnamese-bi-encoder`) over
  `Data/corpus_law_pub.json`.

Design + rationale live in [`PLAN.md`](./PLAN.md). This file covers **only how to run it**.

---

## 1. Prerequisites

| Component | Notes |
|-----------|-------|
| NVIDIA GPUs | 2× 12 GB recommended (RTX 3060 tested) for TP=2 |
| Docker + `nvidia-container-toolkit` | vLLM serving |
| [`uv`](https://astral.sh/uv/) | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Git LFS | `Data/*.json` are LFS-tracked |
| ALQAC API key | Register at the task site above |

---

## 2. One-shot setup

```bash
git clone git@github.com:tien02/alqac-legal-agent.git
cd alqac-legal-agent

bash scripts/setup.sh
```

`scripts/setup.sh` is idempotent. It:

1. Verifies prerequisites (`uv`, `docker`, `git-lfs`, `nvidia-smi`).
2. Pulls LFS data + clones `mini-swe-agent/` if missing.
3. Runs `uv sync` to install Python deps into `.venv/`.
4. Writes `.env` (if missing) with placeholder `ALQAC_API_KEY`.
5. Builds BM25 + dense law indices into `runs/` (skip if present).
6. Starts vLLM via `docker compose` and waits for `/v1/models` to respond.

Edit `.env` afterwards to insert your real key:

```bash
ALQAC_API_KEY=alqac_YOUR_REAL_KEY
```

`.env` is `.gitignore`-tracked. Do not commit it.

---

## 3. Run evaluation

Smoke test (1 case):

```bash
uv run python -m eval.run_dev_set --max-cases 1
```

Single case by ID:

```bash
uv run python -m eval.run_dev_set --case-filter case_4101
```

Full public-test set:

```bash
uv run python -m eval.run_dev_set
```

Outputs land in `runs/`:

| File | Content |
|------|---------|
| `predictions.jsonl` | One line per case: verdict + law refs + score |
| `metrics.json` | Aggregated: accuracy, law-F1, API-penalty |
| `<case_id>.traj.json` | Full mini-swe-agent trajectory per case |
| `case_api.sqlite` | Per-case call counter + returned-chunk cache |

---

## 4. Manual setup (if you skip `scripts/setup.sh`)

```bash
# 4.1 LFS + submodule
git lfs pull
[ -d mini-swe-agent ] || git clone https://github.com/SWE-agent/mini-swe-agent.git

# 4.2 Python deps
uv sync

# 4.3 Env
cp .env .env.bak 2>/dev/null || true    # only if you had one
$EDITOR .env                            # set ALQAC_API_KEY

# 4.4 Build law indices (one-time, ~5–10 min)
uv run python -m retrieval.build_law_index

# 4.5 Start vLLM
docker compose -f docker-compose.vllm.yml up -d

# 4.6 Wait for readiness
curl -sf http://localhost:8001/v1/models -H "Authorization: Bearer local-dev-key"
```

Stop vLLM:

```bash
docker compose -f docker-compose.vllm.yml down
```

---

## 5. Configuration

Edit [`configs/alqac.yaml`](./configs/alqac.yaml):

| Key | Purpose |
|-----|---------|
| `agent.step_limit` | Max mini-swe-agent turns per case (default 8) |
| `agent.wall_time_limit_seconds` | Hard timeout per case |
| `model.model_name` | Must match `--served-model-name` in docker-compose |
| `model.model_kwargs.api_base` | vLLM endpoint (default `http://localhost:8001/v1`) |
| `model.model_kwargs.temperature` | Sampling temperature |
| `environment.timeout` | Per-shell-command timeout |

Budget policy (per-case API-call cap + redundancy check) lives in
[`stopping/budget_tracker.py`](./stopping/budget_tracker.py); tune constants directly.

---

## 6. Troubleshooting

**`torch.OutOfMemoryError` on vLLM startup**
→ Ensure `--tensor-parallel-size 2` and `count: 2` in `docker-compose.vllm.yml`,
or downgrade to `Qwen/Qwen2.5-3B-Instruct`.

**`GGUF model with architecture qwen35 is not supported yet`**
→ vLLM's bundled `transformers` rejects Qwen3.5 GGUF. Use safetensors
(`Qwen/Qwen2.5-7B-Instruct`) or serve via `llama.cpp` instead.

**`ModuleNotFoundError: minisweagent`**
→ `mini-swe-agent/` missing or `pyproject.toml` deps stale. Run
`git clone https://github.com/SWE-agent/mini-swe-agent.git` and `uv sync`.

**`RuntimeError: ALQAC_API_KEY not set`**
→ Edit `.env` and set the real key. `uv run` picks it up via `python-dotenv`.

**All predictions have empty verdict**
→ Model failed to emit the `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` block.
Inspect `runs/<case_id>.traj.json`, raise `agent.step_limit`, or tighten the
system prompt in `configs/alqac.yaml`.

**Law-F1 = 0 with obviously correct citations**
→ Law-name → `law_id` mapping gap. Add override in
`eval/metrics.py::CANONICAL_LAW_MAP`.

---

## 7. Known issues / next tuning

Non-blocking. Pipeline runs end-to-end; these hurt score, not execution.

- **Model emits multiple code blocks per turn → `RepeatedFormatError`.**
  mini-swe-agent v2 enforces exactly one fenced action per turn. Current system
  prompt in `configs/alqac.yaml` uses `mswea_bash_command` fence tag — verify
  against v2.4.4 action-parser expectations. Add explicit "EXACTLY ONE fenced
  block per turn" rule near top of `system_template`.
- **Verdict JSON block often missing.** Model doesn't reliably emit the
  `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` block before step limit. Options:
  raise `agent.step_limit`, shorten instance template, or add few-shot example.
- **Law-F1 low even when verdict correct.** Gold refs use law names; predictions
  use law IDs. Extend `CANONICAL_LAW_MAP` in `eval/metrics.py` for unmapped names
  observed in `runs/predictions.jsonl`.
- **Qwen3.5-9B GGUF blocked by vLLM.** Bundled `transformers` rejects `qwen35`
  architecture. Options: (1) serve via `llama.cpp` OpenAI-compatible endpoint,
  or (2) wait for transformers upstream support.

---

## 8. Repo layout

```
alqac-legal-agent/
├── configs/alqac.yaml           # Prompts + model + env for mini-swe-agent
├── docker-compose.vllm.yml      # vLLM server (Qwen2.5-7B, TP=2)
├── scripts/setup.sh             # One-shot setup (this README §2)
├── Data/                        # ALQAC test set + law corpus (git-lfs)
├── mini-swe-agent/              # Upstream harness (unmodified)
├── tools/                       # CLI wrappers the agent calls
│   ├── search_case.py
│   └── search_law.py
├── retrieval/                   # BM25 + dense + case API client
├── stopping/budget_tracker.py   # Per-case API-call budget
├── eval/                        # Scoring + dev-set runner
├── runs/                        # Outputs (indices, trajectories, metrics)
└── PLAN.md                      # Full design doc
```
