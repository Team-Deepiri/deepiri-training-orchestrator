#!/usr/bin/env bash
# Install deepiri-training-orchestrator via curl:
#   curl -fsSL https://raw.githubusercontent.com/Team-Deepiri/deepiri-training-orchestrator/main/scripts/install.sh | bash
set -euo pipefail

REPO="Team-Deepiri/deepiri-training-orchestrator"
REPO_URL="https://github.com/${REPO}.git"
BRANCH="${DEEPIRI_TRAINING_ORCHESTRATOR_BRANCH:-main}"
KEEP_DIR="${DEEPIRI_TRAINING_ORCHESTRATOR_KEEP_DIR:-0}"

usage() {
  cat <<'EOF'
Usage: install.sh [options]

Clone (when needed) and install deepiri-training-orchestrator (Poetry preferred).

Note: pulls PyTorch and MLflow — first install may take several minutes.

Options:
  -h, --help     Show this help
  --dry-run      Print actions without installing

Environment:
  DEEPIRI_TRAINING_ORCHESTRATOR_SRC       Existing checkout
  DEEPIRI_TRAINING_ORCHESTRATOR_BRANCH    Git branch (default: main)
  DEEPIRI_TRAINING_ORCHESTRATOR_KEEP_DIR  Keep clone when set to 1

Requires: git, python3 (>=3.10, <3.14), poetry (optional)
Verify:   python3 -c "import deepiri_training_orchestrator; print('ok')"
EOF
}

log() { printf '==> %s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }

DRY_RUN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help) usage; exit 0 ;;
    --dry-run) DRY_RUN=1; shift ;;
    *) echo "Unknown option: $1" >&2; exit 1 ;;
  esac
done

for cmd in git python3; do
  command -v "$cmd" >/dev/null 2>&1 || { echo "error: $cmd is required." >&2; exit 1; }
done

ROOT=""
CLEANUP=""

if [[ -n "${DEEPIRI_TRAINING_ORCHESTRATOR_SRC:-}" && -f "${DEEPIRI_TRAINING_ORCHESTRATOR_SRC}/pyproject.toml" ]]; then
  ROOT="${DEEPIRI_TRAINING_ORCHESTRATOR_SRC}"
elif [[ -n "${BASH_SOURCE[0]:-}" ]] && [[ "${BASH_SOURCE[0]}" != bash ]] && [[ -f "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/pyproject.toml" ]]; then
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
else
  ROOT="$(mktemp -d)"
  [[ "$KEEP_DIR" != "1" ]] && CLEANUP="$ROOT"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    log "Would clone ${REPO_URL} to ${ROOT}"
    log "Would poetry install (PyTorch + MLflow)"
    exit 0
  fi
  git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$ROOT"
fi

[[ "$DRY_RUN" -eq 1 ]] && { log "Would install from ${ROOT}"; exit 0; }

trap '[[ -n "$CLEANUP" ]] && rm -rf "$CLEANUP"' EXIT
cd "$ROOT"

warn "Installing PyTorch and MLflow — this may take a while"

if command -v poetry >/dev/null 2>&1; then
  log "Installing with Poetry"
  poetry install --no-interaction --no-ansi
  PYTHON="poetry run python"
else
  warn "poetry not found; using pip editable install (may be slower)"
  VENV="${ROOT}/.venv"
  python3 -m venv "$VENV"
  "$VENV/bin/pip" install -U pip wheel poetry-core -q
  "$VENV/bin/pip" install -e . -q
  PYTHON="${VENV}/bin/python"
fi

"$PYTHON" -c "import deepiri_training_orchestrator; print('deepiri-training-orchestrator import ok')"
echo ""
echo "Verify: python3 -c \"import deepiri_training_orchestrator; print('ok')\""
