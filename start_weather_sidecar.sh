#!/usr/bin/env bash
# Start the WeatherPredictor sidecar.
# On the server, run this in the background or under a process supervisor:
#   nohup ./start_weather_sidecar.sh >> var/logs/weather_sidecar.log 2>&1 &
#
# Or add to the bot's server startup sequence alongside run_live_shadow.sh.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Source .env if present (picks up NOAA_API_TOKEN and any overrides)
if [ -f .env ]; then
    set -a; source .env; set +a
fi

HOST="${WEATHER_SIDECAR_HOST:-127.0.0.1}"
PORT="${WEATHER_SIDECAR_PORT:-8765}"
LOG_DIR="var/logs"
mkdir -p "$LOG_DIR"

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] Starting WeatherPredictor sidecar on ${HOST}:${PORT}"

exec venv/bin/uvicorn sidecar:app \
    --host "$HOST" \
    --port "$PORT" \
    --log-level info
