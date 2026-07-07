# ALQAC 2026 — Legal Case Outcome Prediction

Budget-aware, multi-turn Vietnamese legal-QA harness for
[ALQAC 2026](https://sites.google.com/view/alqac2026/home) Task 1.

- **Harness**: [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) — unmodified upstream.
- **Model server**: `llama.cpp` OpenAI-compat via docker (`docker-compose.llamacpp.yml`), one llama.cpp instance per GPU.
- **Best config so far**: `Qwen3.5-9B` Q4_K_M GGUF, `enable_thinking=false`, EN prompt, `step_limit=8`.
- **Retrieval**: BM25 (underthesea word-segmented) + BGE-M3 dense + BGE-reranker-v2-m3 cross-encoder over `Data/corpus_law_pub.json`.

Design lives in [`PLAN.md`](./PLAN.md). Architecture in [`docs/ARCHITECTURE.md`](./docs/ARCHITECTURE.md).

---

## 1. Prerequisites

| Component | Notes |
|-----------|-------|
| NVIDIA GPUs | 2× 12 GB (RTX 3060 tested). One llama.cpp instance per GPU. |
| Docker + `nvidia-container-toolkit` | Model serving |
| [`uv`](https://astral.sh/uv/) | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Git LFS | `Data/*.json` are LFS-tracked |
| ALQAC API key | Register at task site above |

---

## 2. One-shot setup

```bash
git clone git@github.com:tien02/alqac-legal-agent.git
cd alqac-legal-agent
bash scripts/setup.sh
```

`scripts/setup.sh` is idempotent:

1. Verifies prerequisites.
2. Pulls LFS + clones `mini-swe-agent/` if missing.
3. Installs Python deps via `uv sync`.
4. Writes `.env` (if missing) with placeholder key.
5. Creates `${HOME}/models_alqac/`, downloads GGUF weights.
6. Builds BM25 + dense law indices into `runs/`.
7. Starts llama.cpp containers on both GPUs.

Insert your real ALQAC key:

```
ALQAC_API_KEY=alqac_YOUR_REAL_KEY
```

`.env` is `.gitignore`-tracked. Do not commit.

---

## 3. Run evaluation

Every invocation writes to `runs/<run_id>/` and appends a line to `runs/registry.jsonl`.

Smoke test (1 case):

```bash
uv run python -m eval.run_dev_set --max-cases 1 --run-name smoke
```

Named run w/ custom model + endpoint:

```bash
uv run python -m eval.run_dev_set \
  --max-cases 10 \
  --run-name qwen35base-en-10-nothink \
  --prompt-tag en-decompose-nothink \
  --model-name openai/qwen3.5-9b-gguf \
  --api-base http://localhost:8002/v1 \
  --notes "Qwen35 Q4 nothink"
```

Parallel split across GPUs (uses `--start-idx` / `--end-idx`):

```bash
# GPU 0 :8001
uv run python -m eval.run_dev_set --start-idx 0 --end-idx 25 \
  --run-name full-part1 --api-base http://localhost:8001/v1 \
  --model-name openai/qwen3.5-9b-gguf --prompt-tag en-decompose-nothink &

# GPU 1 :8002
uv run python -m eval.run_dev_set --start-idx 25 --end-idx 50 \
  --run-name full-part2 --api-base http://localhost:8002/v1 \
  --model-name openai/qwen3.5-9b-gguf --prompt-tag en-decompose-nothink &

wait
```

---

## 4. Multi-run tracking

```
runs/
├── registry.jsonl                         # one JSON line per run
└── <run_id>/                              # e.g. 20260707T0141_qwen35-full-part1
    ├── config_snapshot.yaml               # frozen configs/alqac.yaml
    ├── metrics.json                       # aggregated metrics
    ├── predictions.jsonl                  # per-case rows
    ├── case_api.sqlite                    # search_case call log
    └── traj/case_<id>.traj.json           # full trajectories
```

Inspect:

```bash
uv run python -m eval.list_runs                       # tabular registry
uv run python -m eval.compare_runs <run_id_a> <run_id_b>
```

---

## 5. Build submission

Merges predictions from N run dirs, converts `article_number` refs to corpus `aid`,
emits the required submission JSON:

```bash
uv run python -m eval.make_submission \
  runs/<full-part1_run_id> runs/<full-part2_run_id> \
  -o submission.json
```

Output schema:

```json
[
  {
    "case_id": "case_4101",
    "prediction": "A_WIN",
    "case_evidence": ["case_4101_chunk_3"],
    "law_evidence": [{"law_id": "47/2010/QH12", "aid": 270}]
  }
]
```

`aid` is the corpus-internal article id (from `Data/corpus_law_pub.json`), not the
1-indexed article number the model cites.

---

## 6. Serving config

`docker-compose.llamacpp.yml` defines three services (only `qwen35base` starts by default,
others are behind `--profile`):

| Service | Port | GPU | Model | Profile |
|---|---|---|---|---|
| `qwen35base` | 8002 | 1 | `Qwen3.5-9B-base.Q4_K_M.gguf` | default |
| `qwen35base2` | 8001 | 0 | `Qwen3.5-9B-base.Q4_K_M.gguf` | default |
| `jackrong` | 8001 | 0 | `Qwen3.5-9B.Q4_K_M.gguf` (Jackrong distill v2) | `jackrong` |
| `deepseek` | 8003 | 0 | `DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf` | `deepseek` |

Start default pair (best config, both GPUs):

```bash
docker compose -f docker-compose.llamacpp.yml up -d qwen35base qwen35base2
```

Swap in another model on GPU 0:

```bash
docker stop alqac-gpu0 && docker rm alqac-gpu0
docker compose -f docker-compose.llamacpp.yml --profile jackrong up -d jackrong
```

Weights live in `${HOME}/models_alqac/`. Downloaded via `hf` CLI.

Health check:

```bash
curl -sf http://localhost:8001/v1/models -H "Authorization: Bearer local-dev-key"
curl -sf http://localhost:8002/v1/models -H "Authorization: Bearer local-dev-key"
```

Stop everything:

```bash
docker compose -f docker-compose.llamacpp.yml --profile jackrong --profile deepseek down
```

---

## 7. Configuration

`configs/alqac.yaml`:

| Key | Purpose |
|---|---|
| `agent.step_limit` | Max turns per case (default 8) |
| `agent.wall_time_limit_seconds` | Hard timeout per case |
| `model.model_name` | Overridable via `--model-name` |
| `model.model_kwargs.api_base` | Overridable via `--api-base` |
| `model.model_kwargs.extra_body.chat_template_kwargs.enable_thinking` | Qwen3.5 thinking toggle (keep `false`) |
| `environment.env.LAW_INDEX_DEVICE` | Force `cpu` when GPUs are LLM-owned |

Budget policy (per-case API-call cap + redundancy check) lives in
[`stopping/budget_tracker.py`](./stopping/budget_tracker.py).

---

## 8. Benchmark leaderboard (10-case slice, EN prompt)

| Run | Model | Serving | Thinking | acc | law_f1 | combined |
|---|---|---|---|---|---|---|
| baseline (20c) | Qwen3.5-9B safetensors | vLLM fp16 TP=2 | off | 0.60 | 0.049 | **0.570** |
| qwen35base-nothink | Qwen3.5-9B Q4_K_M | llama.cpp | off | **0.60** | 0.029 | **0.566** |
| jackrong-nothink | Jackrong distill v2 Q4 | llama.cpp | off | 0.50 | 0.000 | 0.500 |
| jackrong | Jackrong distill v2 Q4 | llama.cpp | on | 0.30 | 0.029 | 0.386 |
| qwen35base | Qwen3.5-9B Q4_K_M | llama.cpp | on | 0.30 | 0.029 | 0.386 |
| deepseek | DeepSeek-R1-Distill-Qwen-7B Q4 | llama.cpp | on | 0.00 | 0.000 | 0.200 |
| deepseek-nothink | DeepSeek-R1-Distill-Qwen-7B Q4 | llama.cpp | off (ignored) | 0.00 | 0.000 | 0.200 |

Takeaways:

- Thinking mode hurts every model — reasoning tokens burn `step_limit` before submission.
- Q4 GGUF + no-think ≈ fp16 safetensors baseline.
- Reasoning distill (Jackrong, DeepSeek) worse than base Qwen3.5 on this legal task.
- DeepSeek chat template ignores `enable_thinking=false`.

---

## 9. Troubleshooting

**`RepeatedFormatError`** — model emitted >1 fenced block per turn. Tighten prompt or raise `max_consecutive_format_errors`.

**All predictions empty** — model never emitted `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`. Inspect `traj/`, raise `step_limit`.

**`ContextWindowExceededError`** at 16384 — long case_fact + system prompt. Trim, or increase `--max-model-len` / `-c` on llama.cpp side.

**`finish_reason=length` w/ empty content** — thinking mode filled max_tokens. Add `extra_body.chat_template_kwargs.enable_thinking: false`.

**Law_f1=0 with GPUs busy** — `search_law.py` OOM. Ensure `LAW_INDEX_DEVICE=cpu` + `CUDA_VISIBLE_DEVICES=""` in `configs/alqac.yaml::environment.env`.

**`ModuleNotFoundError: minisweagent`** — `git clone https://github.com/SWE-agent/mini-swe-agent.git` then `uv sync`.

---

## 10. Repo layout

```
alqac-legal-agent/
├── configs/alqac.yaml              # prompts + model + env
├── docker-compose.llamacpp.yml     # llama.cpp servers (per-GPU)
├── scripts/setup.sh                # idempotent bootstrap
├── Data/                           # ALQAC test set + law corpus (git-lfs)
├── mini-swe-agent/                 # upstream harness (unmodified)
├── tools/
│   ├── search_case.py              # CLI wrappers agent shells out to
│   └── search_law.py
├── retrieval/
│   ├── alqac_env.py                # LocalEnvironment subclass
│   ├── case_api_client.py          # ALQAC HTTP client + SQLite log
│   ├── embeddings.py               # BGE-M3 + BGE-reranker
│   ├── law_index.py                # BM25 (underthesea)
│   └── build_law_index.py
├── stopping/budget_tracker.py      # per-case call budget
├── eval/
│   ├── run_dev_set.py              # main runner
│   ├── metrics.py                  # scoring
│   ├── list_runs.py                # tabular registry view
│   ├── compare_runs.py             # A/B diff
│   └── make_submission.py          # emit submission JSON
├── runs/                           # per-run outputs + registry
├── docs/ARCHITECTURE.md            # workflow architecture
└── PLAN.md                         # design rationale
```
