#!/usr/bin/env bash
# ./run.sh              backend on :8000 and the UI on :3000
# ./run.sh tests        the test suite
# ./run.sh experiments  the strategy comparison (pass --quick for 3 seeds)
set -euo pipefail
cd "$(dirname "$0")"

[ -x backend/.venv/bin/python ] || { echo "Run ./setup.sh first." >&2; exit 1; }

case "${1:-demo}" in
  tests)
    cd backend && exec .venv/bin/python -m pytest -q ;;
  experiments)
    shift; cd backend && exec .venv/bin/python -m experiments.run_experiments "$@" ;;
  demo)
    (cd backend && exec .venv/bin/uvicorn api.main:app --reload --port 8000) &
    BACKEND=$!
    trap 'kill $BACKEND 2>/dev/null || true' EXIT INT TERM
    echo "Backend: http://localhost:8000/docs    UI: http://localhost:3000"
    cd frontend && npm run dev ;;
  *)
    echo "usage: ./run.sh [demo|tests|experiments [--quick]]" >&2; exit 2 ;;
esac
