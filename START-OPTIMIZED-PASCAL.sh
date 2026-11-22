#!/usr/bin/env bash
# Start the optimized profile server on Linux (mirror of START-OPTIMIZED-PASCAL.bat).
# Binds 0.0.0.0 so the API and web UI are reachable on the LAN; pass --host 127.0.0.1 to keep it local.
#
# usage: ./START-OPTIMIZED-PASCAL.sh /path/to/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf [extra launcher options]
#        e.g. --port 8081 --context 25000 --threads 6 --host 127.0.0.1
set -euo pipefail
cd "$(dirname "$0")"

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 /path/to/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf [options]"
  echo "Example: $0 ~/.llama.cpp/models/Qwen3.6-35B-A3B-UD-Q4_K_M/MTP/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf --port 8081"
  exit 2
fi

MODEL=$1; shift

if [ ! -d .venv ]; then
  echo "Creating the Python environment in .venv ..."
  python3 -m venv .venv
  .venv/bin/python -m pip install -q -r requirements.txt
fi

if [ ! -x build-sm61/strata-qwen35moe ]; then
  echo "Engine not built. Run ./BUILD-PASCAL-SM61.sh first."
  exit 1
fi

exec .venv/bin/python tools/run_qwen36_pascal_optimized.py "$MODEL" \
  --engine build-sm61/strata-qwen35moe \
  --no-open --host 0.0.0.0 "$@"
