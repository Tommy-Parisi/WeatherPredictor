#!/usr/bin/env bash
# Start the WeatherPredictor sidecar.
# On the server, run this in the background or under a process supervisor:
#   nohup ./start_weather_sidecar.sh >> var/logs/weather_sidecar.log 2>&1 &
#
# Requirements: python3 with packages from requirements.txt installed.
# No NOAA_API_TOKEN needed — sidecar now uses GEFS ensemble data from NOMADS (no auth).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f .env ]; then
    set -a; source .env; set +a
fi

HOST="${WEATHER_SIDECAR_HOST:-127.0.0.1}"
PORT="${WEATHER_SIDECAR_PORT:-8765}"
mkdir -p var/logs

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting WeatherPredictor sidecar on ${HOST}:${PORT}"

exec python3 -m uvicorn sidecar:app \
    --host "$HOST" \
    --port "$PORT" \
    --log-level info
