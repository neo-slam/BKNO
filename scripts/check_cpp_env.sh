#!/usr/bin/env bash
set -euo pipefail

PYTHON_EXE="${1:-python3}"

echo "=== BKNO C++ Build Environment Check (Linux) ==="
echo "PWD: $(pwd)"

if command -v "${PYTHON_EXE}" >/dev/null 2>&1; then
  echo "[OK] python: $(command -v "${PYTHON_EXE}")"
  "${PYTHON_EXE}" -c "import torch; print('[OK] torch', torch.__version__)"
else
  echo "[MISSING] python executable: ${PYTHON_EXE}"
  exit 1
fi

if command -v g++ >/dev/null 2>&1; then
  echo "[OK] g++: $(command -v g++)"
else
  echo "[MISSING] g++"
fi

if command -v ninja >/dev/null 2>&1; then
  echo "[OK] ninja: $(command -v ninja)"
else
  echo "[MISSING] ninja"
fi

if command -v cmake >/dev/null 2>&1; then
  echo "[OK] cmake: $(command -v cmake)"
else
  echo "[MISSING] cmake"
fi

echo
echo "Ubuntu 22 packages (if missing):"
echo "  sudo apt update"
echo "  sudo apt install -y build-essential ninja-build cmake python3-dev"

