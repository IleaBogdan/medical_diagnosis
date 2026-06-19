#!/usr/bin/env bash
#
# Launcher for the multi-agent heart disease diagnosis system.
#
# Usage:
#   ./run.sh                         # interactive chat (describe symptoms)
#   ./run.sh chat --text "..."       # one-shot natural-language diagnosis
#   ./run.sh diagnose --row 1        # diagnose a dataset patient
#   ./run.sh evaluate --engine heuristic   # evaluate on the UCI dataset
#   ./run.sh test                    # run the pytest suite
#
# Any arguments are forwarded to the CLI (heart_diagnosis.main).

set -euo pipefail

# Resolve the project root (directory of this script).
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Pick a Python interpreter: prefer the project venv.
if [ -x ".venv/bin/python" ]; then
    PY=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PY="python3"
else
    PY="python"
fi

export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

# `./run.sh test` -> run the test suite.
if [ "${1:-}" = "test" ]; then
    exec "$PY" -m pytest "${@:2}"
fi

# No arguments -> drop into interactive chat.
if [ "$#" -eq 0 ]; then
    exec "$PY" -m heart_diagnosis.main chat
fi

# Otherwise forward all arguments to the CLI.
exec "$PY" -m heart_diagnosis.main "$@"
