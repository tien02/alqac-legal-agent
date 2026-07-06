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
VLLM_BASE_URL=http://localhost:8001/v1
VLLM_API_KEY=local-dev-key
VLLM_MODEL_NAME=qwen2.5-7b
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
mkdir -p runs models

# ------------------------------------------------------------------
# 6. Build law retrieval indices (skip if already built)
# ------------------------------------------------------------------
if [ ! -f runs/law_bm25.pkl ] || [ ! -f runs/law_dense.npy ]; then
  log "Building law indices (BM25 + dense) — first run ~5–10 min..."
  uv run python -m retrieval.build_law_index
else
  log "Law indices already present — skipping rebuild."
fi

# ------------------------------------------------------------------
# 7. vLLM
# ------------------------------------------------------------------
if docker ps --format '{{.Names}}' | grep -q '^alqac-vllm$'; then
  log "vLLM container already running."
else
  log "Starting vLLM (Qwen2.5-7B-Instruct, tensor-parallel-size=2)..."
  docker compose -f docker-compose.vllm.yml up -d

  log "Waiting for vLLM health (max 5 min)..."
  for i in $(seq 1 30); do
    sleep 10
    if curl -sf http://localhost:8001/v1/models \
        -H "Authorization: Bearer local-dev-key" >/dev/null 2>&1; then
      log "vLLM ready after ${i}0s."
      break
    fi
    printf "  waiting ${i}0s...\n"
    if [ "$i" = "30" ]; then
      warn "vLLM did not become ready in 5 min. Check: docker logs alqac-vllm"
    fi
  done
fi

# ------------------------------------------------------------------
# 8. Next steps
# ------------------------------------------------------------------
log "Smoke test (1 case):  uv run python -m eval.run_dev_set --max-cases 1"
log "Full evaluation:      uv run python -m eval.run_dev_set"
log "Setup complete."
