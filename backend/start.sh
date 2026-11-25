#!/usr/bin/env bash
# start.sh — used by Railway / Render as an alternative start command
set -e
exec python -m uvicorn main:app --host 0.0.0.0 --port "${PORT:-8081}"
