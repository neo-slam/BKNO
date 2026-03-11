#!/usr/bin/env bash
set -euo pipefail

PYTHON_EXE="${1:-python3}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

"${PYTHON_EXE}" "${REPO_ROOT}/scripts/test_bkno_cpp.py"

