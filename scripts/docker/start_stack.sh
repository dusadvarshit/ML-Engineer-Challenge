#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="$ROOT_DIR/.env"

cd "$ROOT_DIR"

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi

if [[ -z "${POSTGRES_USER:-}" || -z "${POSTGRES_PASSWORD:-}" ]]; then
  echo "POSTGRES_USER and POSTGRES_PASSWORD must be set before starting the stack." >&2
  exit 1
fi

echo "Building application image..."
sudo docker compose build ml-api

echo "Starting postgres..."
sudo docker compose up -d postgres

echo "Waiting for postgres healthcheck..."
for attempt in $(seq 1 60); do
  postgres_status=$(sudo docker compose ps --format json postgres | python3 -c "
import json
import sys

raw = sys.stdin.read().strip()
if not raw:
    print('')
    raise SystemExit(0)

try:
    data = json.loads(raw)
except json.JSONDecodeError:
    data = [json.loads(line) for line in raw.splitlines() if line.strip()]

if isinstance(data, dict):
    print(data.get('Health', ''))
elif isinstance(data, list) and data:
    first = data[0]
    print(first.get('Health', '') if isinstance(first, dict) else '')
else:
    print('')
")
  if [[ "$postgres_status" == "healthy" ]]; then
    break
  fi

  if [[ "$attempt" -eq 60 ]]; then
    echo "Postgres did not become healthy in time." >&2
    exit 1
  fi

  sleep 2
done

echo "Running alembic migrations..."
sudo docker compose --profile migration run --rm migrate

echo "Starting the full stack..."
sudo docker compose up -d

echo "Stack is up."
