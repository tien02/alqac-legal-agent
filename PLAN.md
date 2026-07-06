# ALQAC 2026 — Minimal Agent Harness Implementation Plan

**Constraints driving every decision below:** no fine-tuning, single local open-weight
model in the ~8B–14B range that is unreliable at structured/JSON output, reuse an
existing, unmodified open-source harness rather than hand-rolled or invented code,
inline tool-result delivery, a hard API-call budget per case, and fast/concurrent
execution for demo purposes.

---

## 0. Task recap

- Input: a short `case_query` per case (plaintiff/defendant dispute description).
- Tool 1 (`search_case_segments`): multi-turn API over ONE hidden case document — you
  can call it repeatedly per case to pull relevant segments.
- Tool 2 (`search_law`): retrieval over a static, open statute corpus.
- Output per case: `A_WIN` / `B_WIN` + cited case-evidence segment IDs + cited law
  provision IDs.
- Score = outcome accuracy + **penalized** case-evidence recall (fewer, higher-precision
  API calls score better) + law-evidence F1.

This makes it a **budget-aware, single-document agentic retrieval problem**, not a
corpus-wide RAG problem for the case side, plus a classic hybrid-IR problem for the
law side.

---

## 1. Architecture

Three roles inside **one harness process** (not three separate frameworks):

```
                     ┌─────────────────────────────┐
  case_query ───────▶│   PLANNER                     │
                     │  decomposes query into        │
                     │  2-4 sub-questions:            │
                     │  - plaintiff's specific claim  │
                     │  - defendant's specific defense│
                     │  - disputed fact / asset       │
                     └───────────────┬────────────────┘
                                     │ sub-questions
                                     ▼
                     ┌─────────────────────────────┐
                     │   CASE-EVIDENCE AGENT         │
                     │  loop per sub-question:        │
                     │   call search_case_segments    │
                     │   -> inline result             │
                     │   -> verify: "sufficient?"     │
                     │   -> stop or reformulate        │
                     │  hard cap: N calls/sub-question│
                     └───────────────┬────────────────┘
                                     │ deduped evidence set
                                     ▼
                     ┌─────────────────────────────┐
                     │   LAW-RETRIEVAL AGENT          │
                     │  query = case evidence + query │
                     │  BM25 + dense hybrid + rerank   │
                     └───────────────┬────────────────┘
                                     │ top-k law provisions
                                     ▼
                     ┌─────────────────────────────┐
                     │   VERDICT AGENT                │
                     │  neutral evidence ordering      │
                     │  cites evidence + law IDs        │
                     │  outputs A_WIN / B_WIN            │
                     │  (optional: 2 runs, swapped      │
                     │   party order, majority vote)    │
                     └─────────────────────────────┘
```

Design rationale (from prior research review):
- Modular Planner → Evidence → Law → Verdict split mirrors MA-RAG's training-free
  multi-agent design, which is validated to work with 8B-class models without any
  fine-tuning: https://github.com/thangylvp/MA-RAG
- Keep everything **inline** in the conversation — do NOT write retrieved segments to
  a scratch file and issue a second "read file" tool call. A harness-design study found
  that this extra indirection step can collapse accuracy (93%→55% in one measured
  config) specifically because smaller/weaker models fail the read-then-integrate step:
  https://arxiv.org/pdf/2605.15184
- Stopping rule should be an explicit **verify-then-continue** check each turn
  ("is current evidence sufficient?"), not a fixed blind call count — this is the
  training-free LLatrieval pattern (verify/update loop, no supervised training needed).
- Use category-conditioned system prompts (different prompt per stage: planning vs.
  evidence-gathering vs. law-retrieval vs. verdict) — this was the single biggest
  contributor to harness quality in the study above.

---

## 2. Repo layout to create

```
alqac2026/
├── mini-swe-agent/          # git submodule / pip install, UNMODIFIED upstream
├── tools/                   # thin CLI wrappers — the ONLY new "agent-facing" code
│   ├── search_case.py       # CLI: python search_case.py --case-id X --query "..."
│   └── search_law.py        # CLI: python search_law.py --query "..."
├── retrieval/
│   ├── case_api_client.py   # wraps ALQAC's search_case_segments endpoint
│   ├── law_index.py         # BM25 + dense hybrid + rerank over statute corpus
│   └── embeddings.py        # loads pretrained (no fine-tune) Vietnamese embedder
├── stopping/
│   └── budget_tracker.py    # per-case call counter + redundancy check (see §5)
│                             # invoked FROM inside search_case.py, not the harness
├── eval/
│   ├── run_dev_set.py       # local scoring harness matching official metric
│   └── metrics.py           # accuracy + penalized recall + F1 implementation
├── data/                    # law corpus, dev set (gitignored raw data)
├── configs/
│   └── mini_swe_agent.yaml  # mini-swe-agent's own config: model, prompt, step cap
└── README.md
```

Key change from earlier drafts: there is no `harness/` package to write. The agent
loop, message history, and action-parsing all come from mini-swe-agent unmodified.
The only code you own is the two CLI tools and the retrieval/eval logic behind them.

---

## 3. Model serving (no fine-tuning — pick one)

- **Ollama** (simplest to stand up): https://github.com/ollama/ollama
- **vLLM** (if you want an OpenAI-compatible server with better throughput, and
  better handling of many concurrent requests for the demo):
  https://github.com/vllm-project/vllm
- Native tool/function-calling support is **not required** — mini-swe-agent doesn't
  use it, so any instruction-tuned ~8B–14B model works: `qwen2.5:14b-instruct`,
  `llama3.1:8b-instruct`, `gemma2:9b-instruct` are all fine candidates. This is the
  main practical payoff of reusing mini-swe-agent: it sidesteps the structured-output
  reliability problem entirely rather than working around it.

Action: stand up Ollama first (fastest to iterate). Move to vLLM if you need higher
concurrent throughput once you're running many cases in parallel for the demo.

---

## 4. Harness — reuse mini-swe-agent unmodified

**https://github.com/SWE-agent/mini-swe-agent**

Decision: adopt this harness as-is, with zero custom loop/parser code. Do not hand-roll
a ReAct loop, and do not adopt smolagents/CrewAI/PydanticAI — all three either force a
JSON/schema-validated action format (PydanticAI, CrewAI, smolagents' ToolCallingAgent)
or ask the model to emit syntactically valid code (smolagents' CodeAgent), both of which
are exactly the failure mode this project needs to avoid with a <10B model, and both add
framework overhead that works against the "fast, simple, concurrent demo" goal.

Why mini-swe-agent fits without modification:
- It has no tool-calling interface at all — the model just writes a single fenced
  ` ```bash ` code block per turn, and the harness executes it and feeds back
  stdout/stderr. No JSON, no schema, no native function-calling requirement, so it
  works with any instruction-tuned model regardless of tool-calling support.
- ~100 lines of core loop, linear message history, one stateless `subprocess.run`
  per action — nothing to configure beyond its own YAML config.
- Concurrency is free: each case is one independent agent instance; run N of them
  under an `asyncio`/process pool against your local Ollama/vLLM server. No
  orchestration layer to fight, because there isn't one.

Integration: your two tools (`search_case_segments`, `search_law`) become plain CLI
scripts (§2, `tools/`). The model calls them exactly like any other shell command:

```bash
python tools/search_case.py --case-id 1042 --query "hợp đồng thuê máy xúc"
python tools/search_law.py --query "trách nhiệm bồi thường thiệt hại hợp đồng"
```

No custom action schema to design, no parser to write — mini-swe-agent's existing
fenced-code-block extraction handles it, and the CLI's stdout (JSON or plain text,
your choice, since it's just a subprocess call) becomes the next observation.

---

## 5. Stopping / budget policy — lives inside `tools/search_case.py`, not the harness

Since the harness (mini-swe-agent) is unmodified, the budget logic can't live in a
harness hook — it has to live in the CLI tool itself, using process-local state keyed
by `--case-id` (e.g. a small JSON/SQLite counter file under `stopping/`), plus
mini-swe-agent's own built-in step cap as the outer safety net:

1. `search_case.py` tracks, per case ID, how many times it's been called and the
   embeddings of everything returned so far.
   - If a new candidate segment's max similarity to already-returned segments exceeds
     a threshold (e.g. 0.92), skip returning it again and instead return a short
     `"no new relevant segment found"` message — this nudges the model to stop
     re-querying the same angle without needing any harness-level intervention.
2. Enforce a hard per-case call ceiling directly in the script (e.g. after 8 calls,
   return `"budget exhausted, proceed to law search and verdict with current
   evidence"` instead of a result) — this is your real backstop for the penalized
   case-recall metric, independent of the model's own judgment.
3. Separately, cap mini-swe-agent's own `step_limit` in `configs/mini_swe_agent.yaml`
   as a blunt outer bound on total turns per case (covers both tools, prevents
   runaway loops from wasted law-search calls too).

Tune both the similarity threshold and the call ceiling empirically against the
dev-set penalty curve once `eval/metrics.py` (§8) is working.

---

## 6. Law retrieval (`retrieval/law_index.py`)

- Sparse: `rank_bm25` (zero setup, pure Python).
- Dense: pretrained Vietnamese sentence embedder, no fine-tuning needed:
  `bkai-foundation-models/vietnamese-bi-encoder` (Hugging Face) — or `BAAI/bge-m3`
  for a multilingual fallback if Vietnamese-specific embeddings underperform.
- Rerank top ~30 candidates with **FlashRank** (lightweight, CPU-friendly
  cross-encoder reranker, no training): https://github.com/PrithivirajDamodaran/FlashRank
- Query for this stage = `case_query` + retrieved case-evidence text (not the raw
  short query alone) — statute language rarely matches colloquial dispute phrasing.
- Warm-start / sanity-check the pipeline against public ALQAC 2021–2025 Task 1 data
  (same retrieval task, different year) before touching this year's hidden data:
  https://sites.google.com/view/ALQAC-2024 and https://sites.google.com/view/ALQAC-2025

---

## 7. Verdict agent — bias control

- Present plaintiff/defendant evidence in a fixed neutral order (e.g. chronological),
  not grouped by party — LLM predictions are known to be biased toward whichever
  party's evidence is presented last.
- If budget allows: run the verdict step twice with evidence order swapped, take the
  majority label, keep the union of cited evidence/law IDs from the run whose reasoning
  is more specific (or from both, deduped).

---

## 8. Local evaluation harness (`eval/run_dev_set.py`)

Implement the exact scoring formula (accuracy + penalized case-recall + law-F1) locally
before touching the real API, using a held-out slice of any labeled/practice data the
organizers release. This lets you tune the budget cap in §5 against the real
efficiency/recall tradeoff instead of guessing.

---

## 9. Milestones (suggested order for Claude Code to execute)

1. `pip install mini-swe-agent` (or clone it as a submodule), get it running
   end-to-end on a "hello world" task with a local Ollama model and its default
   `bash` tool — confirm the loop, config, and step cap work before touching ALQAC.
2. Implement `case_api_client.py` against the real ALQAC endpoint spec; write a
   throwaway script that fires one manual query and prints the raw response shape.
3. Wrap it as `tools/search_case.py` (plain CLI, args in, text/JSON out to stdout).
   Point mini-swe-agent's config at a system prompt that tells the model this tool
   exists and how to call it; confirm the model can retrieve evidence for one case
   end-to-end using mini-swe-agent's own loop, no custom stopping logic yet.
4. Add the per-case call counter + redundancy check inside `search_case.py` (§5);
   compare API-call counts before/after on a handful of cases.
5. Build `law_index.py` (BM25 + dense + rerank), validate against public ALQAC
   2024/2025 Task 1 data; wrap as `tools/search_law.py`.
6. Update the system prompt so the model knows to gather case evidence first, then
   search law, then output a final `A_WIN`/`B_WIN` verdict with cited IDs as its
   last message (mini-swe-agent just needs a clear instruction for what "done" looks
   like — no special verdict-agent code required).
7. Add neutral evidence ordering to how `search_case.py` formats its output text
   (chronological, not grouped by party) to control the known last-evidence bias.
8. Build `eval/metrics.py` implementing the official scoring formula; run the full
   mini-swe-agent pipeline over the dev set (in parallel, one process per case);
   tune stopping thresholds and budget caps against it.
9. Ablate: call-ceiling value; similarity threshold for redundancy detection;
   BM25-only vs. hybrid law retrieval — keep whichever combination wins on the
   dev-set score, not on accuracy alone (the penalty term matters).

---

## Reference links (recommended path only — all reused as-is, nothing hand-rolled)

- **mini-SWE-agent** (the harness — unmodified): https://github.com/SWE-agent/mini-swe-agent
- Ollama (model serving): https://github.com/ollama/ollama
- vLLM (model serving, higher concurrency): https://github.com/vllm-project/vllm
- `rank_bm25` (sparse law retrieval, pure Python, zero setup):
  https://github.com/dorianbrown/rank_bm25
- FlashRank (lightweight reranker, no training): https://github.com/PrithivirajDamodaran/FlashRank
- Vietnamese bi-encoder (pretrained, no fine-tuning needed):
  https://huggingface.co/bkai-foundation-models/vietnamese-bi-encoder
- BGE-M3 (multilingual fallback embedder): https://huggingface.co/BAAI/bge-m3
- ALQAC 2026 (task site): https://sites.google.com/view/alqac2026/home
- ALQAC 2024/2025 (public Task 1 data for warm-starting law retrieval):
  https://sites.google.com/view/ALQAC-2024 · https://sites.google.com/view/ALQAC-2025

### Background / rationale only (not part of the build — no code from these is used)
- Harness-design study that motivated the inline-delivery and "avoid extra
  indirection" decisions in §1 and §4: https://arxiv.org/pdf/2605.15184