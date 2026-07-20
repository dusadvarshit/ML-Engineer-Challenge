#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -eq 0 ]]; then
  echo "No command supplied to the container." >&2
  exit 64
fi

exec "$@"
