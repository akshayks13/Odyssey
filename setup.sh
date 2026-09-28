#!/usr/bin/env bash
# One-time setup: a Python virtual environment for the backend and the npm packages for the UI.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
if ! "$PY" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)'; then
  echo "Python 3.11 or newer is required (found: $("$PY" --version 2>&1)). Set PYTHON=/path/to/python3.11+ and rerun." >&2
  exit 1
fi
command -v npm >/dev/null || { echo "Node.js 18+ (npm) is required for the UI." >&2; exit 1; }

echo "== backend: virtual environment and packages"
"$PY" -m venv backend/.venv
backend/.venv/bin/pip install --quiet --upgrade pip
backend/.venv/bin/pip install --quiet -r backend/requirements-dev.txt

echo "== frontend: npm packages"
(cd frontend && npm install --silent)

echo
echo "Done. Start it with ./run.sh   (tests: ./run.sh tests   experiments: ./run.sh experiments)"
