# ALQAC 2026 — Workflow Architecture

End-to-end pipeline for Vietnamese legal case-outcome prediction.
Complements [`PLAN.md`](../PLAN.md) (design rationale) and
[`README.md`](../README.md) (run instructions).

---

## 1. High-level pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ INPUT DATA (Data/, git-lfs)                                       │
│  ALQAC2026_public_test.json  — 100 test cases                      │
│  corpus_law_pub.json         — 18 laws, 3352 articles              │
└───────────────────────┬─────────────────────────────────────────┘
                        │
             one-time   │ retrieval/build_law_index.py
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ LAW INDEX (runs/)                                                 │
│  law_bm25.pkl        — BM25Okapi (underthesea word-segmented)      │
│  law_dense.npy       — BGE-M3 embeddings (1024-d)                  │
│  law_meta.json       — article metadata                            │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        │  per eval invocation
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ eval/run_dev_set.py                                               │
│   loads configs/alqac.yaml (prompts + model + env)                 │
│   loops over cases → mini-swe-agent DefaultAgent per case          │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ MINI-SWE-AGENT LOOP (unmodified upstream)                          │
│   system prompt + case task                                        │
│   ↻ model.query() ─▶ single ```mswea_bash_command``` block         │
│                       │                                             │
│                       ▼                                             │
│              subprocess.run(tools/search_case.py |                 │
│                             tools/search_law.py)                    │
│                       │                                             │
│                       ▼                                             │
│              observation → next user turn                          │
│   exits on: Submitted / LimitsExceeded / TimeExceeded /            │
│             RepeatedFormatError / uncaught exception                │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ eval/metrics.py — score_case()                                     │
│   parse verdict + citations from submission JSON                   │
│   compare vs Data/…verdict_label + related_law_provisions          │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────────┐
│ PER-RUN OUTPUT (runs/<run_id>/)                                    │
│  predictions.jsonl, metrics.json, config_snapshot.yaml,            │
│  case_api.sqlite, traj/case_<id>.traj.json                         │
│                                                                    │
│  + append line to runs/registry.jsonl                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Roles

Three logical roles inside a **single agent process** (not three frameworks):

| Role | Where implemented |
|---|---|
| **Planner** | Model turn 1, driven by `configs/alqac.yaml::system_template` — decomposes case into 3–5 legal sub-issues |
| **Evidence agent** | Model calls `search_case.py` in a loop; budget tracker inside `stopping/budget_tracker.py` caps calls + deduplicates chunks |
| **Law-retrieval agent** | Model calls `search_law.py`; BM25 (sparse) + BGE-M3 (dense) + BGE-reranker-v2-m3 fused via RRF |
| **Verdict agent** | Final assistant turn emits `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` + JSON |

Design rationale: adopt mini-swe-agent as-is (no custom loop, no JSON tool-calling schema), so the pipeline works with any instruction-tuned model regardless of native function-calling support.

---

## 3. Model serving

Two OpenAI-compatible endpoints via **llama.cpp docker** (`docker-compose.llamacpp.yml`):

| Container | Port | GPU | Model | Format |
|---|---|---|---|---|
| `alqac-jackrong` | 8001 | 0 | `Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2` | Q4_K_M GGUF |
| `alqac-qwen35base` | 8002 | 1 | `unsloth/Qwen3.5-9B-GGUF` | Q4_K_M GGUF |
| `alqac-deepseek` (profile) | 8003 | 0 | `unsloth/DeepSeek-R1-Distill-Qwen-7B-GGUF` | Q4_K_M GGUF |

Each 9B Q4_K_M fits comfortably on 12 GB (RTX 3060). Runtime image:
`ghcr.io/ggml-org/llama.cpp:server-cuda`. Pinned per-GPU via
`NVIDIA_VISIBLE_DEVICES`.

Model files live in `${HOME}/models_alqac/` (outside repo).
Downloaded via `hf` CLI in `scripts/setup.sh`.

---

## 4. Configuration

`configs/alqac.yaml` — single YAML consumed by `mini-swe-agent.config.get_config_from_spec`.

| Section | Purpose |
|---|---|
| `agent` | `step_limit`, `wall_time_limit_seconds`, `system_template`, `instance_template` |
| `model` | `model_class`, `model_name`, `api_base`, `temperature`, `max_tokens`, `extra_body.chat_template_kwargs` |
| `environment` | `LocalEnvironment` subclass, per-shell timeout, env-var injection (`LAW_INDEX_DEVICE=cpu`, `CUDA_VISIBLE_DEVICES=""`) |

CLI overrides (`eval/run_dev_set.py`):

```
--model-name openai/deepseek-r1-qwen-7b
--api-base   http://localhost:8003/v1
```

Overrides mutate `base_config["model"]` **before** `get_model` is called. Full snapshot of the effective config is copied to `runs/<run_id>/config_snapshot.yaml`.

---

## 5. Loop stop conditions

`mini-swe-agent/src/minisweagent/agents/default.py::run()`:

```python
while True:
    step()          # query model → parse actions → exec → observe
    if messages[-1].role == "exit":
        break
```

| Exit status | Trigger | Source |
|---|---|---|
| **Submitted** | subprocess stdout first line == `COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT` AND returncode==0 | `environments/local.py::_check_finished` |
| **LimitsExceeded** | `n_calls >= step_limit` OR `cost >= cost_limit` | `agents/default.py::query` |
| **TimeExceeded** | `time.time() - start >= wall_time_limit_seconds` | `agents/default.py::query` |
| **RepeatedFormatError** | `n_consecutive_format_errors >= max_consecutive_format_errors` (default 3). Clean step resets counter. | `agents/default.py::run` |
| **UncaughtException** | any other exception from env/model | re-raises through outer try |

Only **Submitted** yields a scoreable submission. Everything else zero-fills `pred_verdict` and hurts `combined_score`.

Submit pattern the prompt requires:

```bash
echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && cat <<'EOF'
{"verdict":"...","evidence_chunk_ids":[...],
 "law_refs":[{"law_id":"91/2015/QH13","article_number":584}],
 "reason":"..."}
EOF
```

---

## 6. Retrieval stack

### 6.1 Case API (`retrieval/case_api_client.py`)

- Remote HTTP endpoint (`https://alqac-api.ngrok.pro/retrieve`), auth via `ALQAC_API_KEY`.
- SQLite log per case_id (`runs/<run_id>/case_api.sqlite`) — tracks all returned chunks + counter.
- Budget policy in `stopping/budget_tracker.py`:
  - Hard cap: N calls per case (soft: ≤10 ideal).
  - Redundancy check: cosine similarity to already-returned chunks; if > threshold, return `"no new relevant segment found"` instead of the chunk (nudges model to stop).

### 6.2 Law retrieval (`retrieval/law_index.py`, `retrieval/embeddings.py`)

Three-stage hybrid:

1. **Sparse (BM25Okapi)** — `underthesea.word_tokenize()` merges Vietnamese compound words (`hợp đồng` → `hợp_đồng`). Necessary because whitespace tokenization treats each syllable as a token, destroying phrase precision on legal jargon.
2. **Dense (BGE-M3)** — 1024-d multilingual embeddings, cosine over normalized vectors. CPU inference during eval to avoid contending with the LLM for GPU memory (`LAW_INDEX_DEVICE=cpu` env var).
3. **Rerank (BGE-reranker-v2-m3)** — cross-encoder scores top-30 fused candidates; returns top-k.

Fusion via Reciprocal Rank Fusion (`k=60`) between BM25 top-30 and dense top-30.

---

## 7. Scoring (`eval/metrics.py`)

Per case:

```
verdict_correct   ∈ {0,1}  — exact match on 4-class label
law_precision     = |pred ∩ gold| / |pred|
law_recall        = |pred ∩ gold| / |gold|
law_f1            = 2·P·R / (P+R)
api_penalty       = max(0, (n_unique_chunks − gold_segments_estimate) / gold_segments_estimate)
```

Aggregate:

```
accuracy_4class  = mean(verdict_correct)
law_f1           = mean(law_f1)
avg_api_penalty  = mean(api_penalty)
combined_score   = 0.5·accuracy + 0.3·(1 − avg_api_penalty) + 0.2·law_f1
```

Gold `law_id` normalization: `related_law_provisions` free-text → canonical id via `CANONICAL_LAW_MAP` + fuzzy match on corpus's article-1 first sentence.

---

## 8. Multi-run registry

Every eval invocation:

1. Derives `run_id = YYYYMMDDTHHMM_<model>_<tag>` (or `--run-name`).
2. Creates `runs/<run_id>/{traj/,predictions.jsonl,metrics.json,case_api.sqlite,config_snapshot.yaml}`.
3. Appends one line to `runs/registry.jsonl`:

   ```json
   {"run_id":"...","ts":"...","model":"...","prompt_tag":"...",
    "notes":"...","step_limit":8,"n_cases":10,
    "accuracy_4class":0.30,"law_f1":0.029,
    "avg_api_penalty":0.0,"combined_score":0.386}
   ```

Inspection:

```
uv run python -m eval.list_runs                       # rich table
uv run python -m eval.compare_runs <run_a> <run_b>    # diff table
```

---

## 9. Component boundaries

```
┌────────────────────────┐
│ eval/ (owner: this repo) │
│   run_dev_set.py         │
│   metrics.py             │
│   list_runs.py           │
│   compare_runs.py        │
└──────────┬───────────────┘
           │
┌──────────▼────────────────────┐
│ retrieval/ (owner: this repo)   │
│   alqac_env.py                  │  AlqacEnv extends LocalEnv
│   case_api_client.py            │
│   law_index.py                  │
│   embeddings.py                 │
│   build_law_index.py            │
└──────────┬────────────────────┘
           │
┌──────────▼────────────────────┐
│ tools/ (owner: this repo)       │
│   search_case.py                │  CLI wrappers
│   search_law.py                 │  agent-facing
└──────────┬────────────────────┘
           │
┌──────────▼────────────────────┐
│ mini-swe-agent/ (upstream)      │
│   agents/default.py             │  DefaultAgent, DO NOT MODIFY
│   models/litellm_textbased.py   │
│   environments/local.py         │
└──────────┬────────────────────┘
           │
┌──────────▼────────────────────┐
│ Model server (docker)           │
│   llama.cpp OpenAI-compat       │  per-GPU llama.cpp
└─────────────────────────────────┘
```

Rule: **mini-swe-agent stays unmodified**. All ALQAC-specific logic lives in
`tools/`, `retrieval/`, `eval/`. `AlqacEnv` extends `LocalEnvironment` only to inject
Jinja template vars for the case; loop / parser / message-history behavior is upstream.

---

## 10. Failure modes seen so far

| Symptom | Root cause | Mitigation |
|---|---|---|
| `RepeatedFormatError` after 3 turns, empty submission | Model emits >1 fenced block per turn | Prompt: enforce "EXACTLY ONE fenced block" at top |
| `LimitsExceeded` before submit | 4-step VN workflow needs ≥6 tool calls, step_limit=8 too tight | Raise step_limit to 12–15 |
| `ContextWindowExceededError` at 16384 | Long case_fact + long system prompt + big observation | Trim system prompt or truncate observation via `observation_template` |
| Empty `content`, `finish_reason=length` | Qwen3.5 default thinking mode fills max_tokens with `<think>` | Set `chat_template_kwargs.enable_thinking=false` in `extra_body` |
| All law_f1 = 0 while retrieval seemingly working | GPU OOM in `search_law.py` (LLM occupies both GPUs) | `LAW_INDEX_DEVICE=cpu` + `CUDA_VISIBLE_DEVICES=""` in env config |
| Q4 GGUF regression vs fp16 safetensors | Quantization loss compounds with reasoning | Q5/Q6 quant or restore safetensors for evaluation |

---

## 11. Change checklist

Before changing:

- **System prompt** → snapshot goes to `runs/<run_id>/config_snapshot.yaml` automatically. Tag with `--prompt-tag` for cross-run search.
- **Model** → `--model-name` + `--api-base`; also verify container up + healthy.
- **Retrieval tokenizer** → rebuild BM25 (`retrieval.build_law_index`) or query tokens will mismatch corpus tokens.
- **Metrics formula** → migrate historical `registry.jsonl` entries or note metric change in `notes`.

---

## 12. Non-goals

- No fine-tuning. Every model is used off-the-shelf.
- No custom agent loop. mini-swe-agent handles turns, parsing, exit.
- No structured tool-calling API. Agent shells out via fenced code blocks — works with any instruction-tuned model.
- No `harness/` package. All ALQAC glue is CLI wrappers + one config file.
