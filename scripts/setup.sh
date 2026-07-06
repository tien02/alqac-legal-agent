#!/usr/bin/env bash
# One-shot setup for ALQAC 2026 harness.
# Idempotent — safe to re-run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

log() { printf "\033[1;34m[setup]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[warn]\033[0m %s\n" "$*"; }
die() { printf "\033[1;31m[error]\033[0m %s\n" "$*" >&2; exit 1; }

# ------------------------------------------------------------------
# 1. Prerequisites
# ------------------------------------------------------------------
log "Checking prerequisites..."
command -v git >/dev/null || die "git not found"
command -v docker >/dev/null || die "docker not found"
command -v uv >/dev/null || die "uv not found. Install: curl -LsSf https://astral.sh/uv/install.sh | sh"
command -v git-lfs >/dev/null || warn "git-lfs missing — Data/*.json will be pointers"
command -v nvidia-smi >/dev/null || warn "nvidia-smi missing — vLLM will fail without a GPU"

# ------------------------------------------------------------------
# 2. Git LFS + submodule
# ------------------------------------------------------------------
if command -v git-lfs >/dev/null; then
  log "Pulling git-lfs data..."
  git lfs install --local >/dev/null
  git lfs pull
fi

if [ -f .gitmodules ]; then
  log "Updating submodules..."
  git submodule update --init --recursive
fi

if [ ! -d mini-swe-agent ]; then
  log "Cloning mini-swe-agent (unmodified upstream)..."
  git clone https://github.com/SWE-agent/mini-swe-agent.git mini-swe-agent
fi

# ------------------------------------------------------------------
# 3. Python environment
# ------------------------------------------------------------------
log "Installing Python dependencies via uv..."
uv sync

# ------------------------------------------------------------------
# 4. .env
# ------------------------------------------------------------------
if [ ! -f .env ]; then
  log "Creating .env from template. Set ALQAC_API_KEY before running eval."
  cat > .env <<'EOF'
ALQAC_API_KEY=alqac_REPLACE_ME
LLAMACPP_BASE_URL=http://localhost:8001/v1
LLAMACPP_API_KEY=local-dev-key
LLAMACPP_MODEL_NAME=jackrong-distill-9b
HF_TOKEN=
ALQAC_DATA_DIR=./Data
ALQAC_RUNS_DIR=./runs
EOF
  warn ".env created with placeholder ALQAC_API_KEY — edit before running eval."
else
  log ".env already exists — leaving untouched."
fi

# ------------------------------------------------------------------
# 5. Runs / models dirs
# ------------------------------------------------------------------
mkdir -p runs "${HOME}/models_alqac"

# ------------------------------------------------------------------
# 6. Download GGUF models (skip if present)
# ------------------------------------------------------------------
JACKRONG=${HOME}/models_alqac/Qwen3.5-9B.Q4_K_M.gguf
QWEN35BASE=${HOME}/models_alqac/Qwen3.5-9B-base.Q4_K_M.gguf

if [ ! -f "$JACKRONG" ]; then
  log "Downloading Jackrong Qwen3.5-9B Q4_K_M (~5.6 GB)..."
  uv run hf download Jackrong/Qwen3.5-9B-Claude-4.6-Opus-Reasoning-Distilled-v2-GGUF \
    Qwen3.5-9B.Q4_K_M.gguf --local-dir "${HOME}/models_alqac/"
else
  log "Jackrong GGUF present — skipping."
fi

if [ ! -f "$QWEN35BASE" ]; then
  log "Downloading Qwen3.5-9B base Q4_K_M (~5.3 GB)..."
  uv run hf download unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf \
    --local-dir "${HOME}/models_alqac/"
  mv "${HOME}/models_alqac/Qwen3.5-9B-Q4_K_M.gguf" "$QWEN35BASE"
else
  log "Qwen3.5-9B base GGUF present — skipping."
fi

# ------------------------------------------------------------------
# 7. Build law retrieval indices (skip if already built)
# ------------------------------------------------------------------
if [ ! -f runs/law_bm25.pkl ] || [ ! -f runs/law_dense.npy ]; then
  log "Building law indices (BM25 + dense) — first run ~5–10 min..."
  uv run python -m retrieval.build_law_index
else
  log "Law indices already present — skipping rebuild."
fi

# ------------------------------------------------------------------
# 8. llama.cpp servers (Jackrong on GPU 0 :8001, Qwen35-base on GPU 1 :8002)
# ------------------------------------------------------------------
if docker ps --format '{{.Names}}' | grep -q '^alqac-jackrong$'; then
  log "llama.cpp containers already running."
else
  log "Starting llama.cpp (Jackrong :8001, Qwen35-base :8002)..."
  docker compose -f docker-compose.llamacpp.yml up -d

  log "Waiting for both endpoints (max 5 min)..."
  for i in $(seq 1 30); do
    sleep 10
    A=$(curl -sf http://localhost:8001/health >/dev/null 2>&1 && echo ok || echo wait)
    B=$(curl -sf http://localhost:8002/health >/dev/null 2>&1 && echo ok || echo wait)
    if [ "$A $B" = "ok ok" ]; then
      log "Both llama.cpp endpoints ready after ${i}0s."
      break
    fi
    printf "  jackrong:%s qwen35base:%s (%s s)\n" "$A" "$B" "$((i * 10))"
    if [ "$i" = "30" ]; then
      warn "Some endpoints did not become ready. Check: docker logs alqac-jackrong / alqac-qwen35base"
    fi
  done
fi

# ------------------------------------------------------------------
# 9. Next steps
# ------------------------------------------------------------------
log "Smoke test (1 case):  uv run python -m eval.run_dev_set --max-cases 1 --run-name smoke"
log "Full evaluation:      uv run python -m eval.run_dev_set --run-name full"
log "List runs:            uv run python -m eval.list_runs"
log "Compare two runs:     uv run python -m eval.compare_runs <run_id_a> <run_id_b>"
log "Setup complete."
