"""
WeatherPredictor sidecar — HTTP service for the Kalshi trading bot.

Prediction path (new)
---------------------
Uses GEFS 31-member ensemble forecasts from NOMADS. A background thread fetches
and caches ensemble data for today and tomorrow, refreshing every GEFS_REFRESH_SECS.
The /predict endpoint reads from that cache and returns in well under the Rust
bot's 3-second timeout.

Response contract (motorcade standard)
---------------------------------------
    {
        "probability":    0.62,       # P(daily_high > floor_strike_f)
        "data_age_secs":  1800,       # seconds since GEFS data was fetched
        "data_source_ok": true,       # false → Rust bot falls back to bucket model
        "model_version":  "gefs_v1"
    }

data_source_ok is false when:
  - GEFS cache is empty (startup warmup not complete)
  - Cached data is older than MAX_DATA_AGE_SECS (default 2 h)
  - Fewer than MIN_MEMBERS_REQUIRED ensemble members succeeded

Endpoints
---------
    GET /health              → {"status": "ok", "cache_dates": [...], "model_version": "gefs_v1"}
    GET /predict?ticker=...  → motorcade response contract above

Supported tickers: KXHIGH{PHI,PHIL,PHILLY,PHL}*  (Philadelphia high-temp only).

Environment variables
---------------------
    WEATHER_SIDECAR_HOST         Optional. Default: 127.0.0.1
    WEATHER_SIDECAR_PORT         Optional. Default: 8765
    GEFS_REFRESH_SECS            Optional. Default: 7200 (2 hours)
    GEFS_MAX_DATA_AGE_SECS       Optional. Default: 7200 (2 hours)
"""

import json
import logging
import os
import re
import threading
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException

from src.data_collection.gefs_fetcher import GEFSResult, fetch_ensemble_daily_highs, MEMBERS
from src.modeling.ensemble_predictor import predict as ensemble_predict

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="WeatherPredictor Sidecar", version="2.0")

# ── Config ─────────────────────────────────────────────────────────────────────

GEFS_REFRESH_SECS    = int(os.getenv("GEFS_REFRESH_SECS",      "7200"))
MAX_DATA_AGE_SECS    = int(os.getenv("GEFS_MAX_DATA_AGE_SECS", "7200"))
PREDICTION_LOG_DIR   = Path(os.getenv("GEFS_PREDICTION_LOG_DIR", "var/logs/gefs_predictions"))

MODEL_VERSION = "gefs_v1"

# Philadelphia city codes accepted in tickers
PHILADELPHIA_CODES = {"PHI", "PHIL", "PHILLY", "PHL"}

# ── Prediction log ────────────────────────────────────────────────────────────
#
# One JSONL file per day under PREDICTION_LOG_DIR.
# Each line is a complete prediction record including all member highs so we
# can later join against NOAA actuals to build the bias correction table.
#
# Schema (all fields always present):
#   ts              ISO-8601 UTC timestamp of this prediction
#   ticker          Kalshi ticker
#   target_date     YYYY-MM-DD date the market resolves
#   threshold_f     floor strike in °F from the ticker
#   probability     P(high > threshold_f) returned to the bot
#   n_members       number of ensemble members that succeeded
#   member_highs_f  list of per-member predicted daily highs (°F)
#   run_time        ISO-8601 UTC of the GEFS model run used
#   data_age_secs   age of GEFS data at prediction time
#   model_version   e.g. "gefs_v1"

_log_lock = threading.Lock()


def _write_prediction_log(record: dict) -> None:
    """Append one prediction record to today's JSONL log. Silently swallows errors."""
    try:
        day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        path = PREDICTION_LOG_DIR / f"predictions_{day}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(record, default=str) + "\n"
        with _log_lock:
            with open(path, "a") as f:
                f.write(line)
    except Exception as exc:
        logger.warning(f"prediction log write failed: {exc}")


# ── Cache ──────────────────────────────────────────────────────────────────────
#
# Keyed by target date. Each value is a GEFSResult (which carries its own
# fetch_time). A single lock guards all reads and writes.

_cache: dict[date, GEFSResult] = {}
_cache_lock = threading.Lock()


def _refresh_date(target_date: date) -> None:
    """Fetch GEFS data for one target date and store in cache."""
    logger.info(f"GEFS refresh: fetching {target_date}")
    result = fetch_ensemble_daily_highs(target_date)
    if result is not None:
        with _cache_lock:
            _cache[target_date] = result
        logger.info(f"GEFS cache updated: {target_date}  members={result.n_members}")
    else:
        logger.warning(f"GEFS refresh failed for {target_date}")


def _background_refresh() -> None:
    """Background thread: refresh today + tomorrow every GEFS_REFRESH_SECS.
    Sleeps first so startup warmup and first background fetch don't overlap."""
    time.sleep(GEFS_REFRESH_SECS)
    while True:
        today = datetime.now(timezone.utc).date()
        for target_date in [today, today + timedelta(days=1)]:
            try:
                _refresh_date(target_date)
            except Exception as exc:
                logger.error(f"GEFS refresh error for {target_date}: {exc}", exc_info=True)
        time.sleep(GEFS_REFRESH_SECS)


# ── Ticker parsing ─────────────────────────────────────────────────────────────

def _parse_ticker(ticker: str) -> tuple[Optional[str], Optional[date], Optional[float]]:
    """
    Parse a Kalshi weather ticker into (city_code, target_date, floor_strike_f).

    Examples:
        KXHIGHPHI-26APR15-T55   → ("PHI",  date(2026, 4, 15),  55.0)
        KXHIGHPHIL-25JUL31-T92  → ("PHIL", date(2025, 7, 31),  92.0)
    """
    upper = ticker.upper()
    if not upper.startswith("KXHIGH"):
        return None, None, None

    rest     = upper[len("KXHIGH"):]
    city_end = next((i for i, c in enumerate(rest) if not c.isalpha()), len(rest))
    city     = rest[:city_end]

    date_match = re.search(r"-(\d{2}[A-Z]{3}\d{2})-", upper)
    target_date = None
    if date_match:
        try:
            target_date = datetime.strptime(date_match.group(1), "%y%b%d").date()
        except ValueError:
            pass

    thresh_match = re.search(r"-T(\d+)$", upper)
    floor_strike_f = float(thresh_match.group(1)) if thresh_match else None

    return city, target_date, floor_strike_f


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.on_event("startup")
def on_startup():
    # Pre-warm the cache before serving traffic, then keep it fresh in background.
    today = datetime.now(timezone.utc).date()
    for target_date in [today, today + timedelta(days=1)]:
        try:
            _refresh_date(target_date)
        except Exception as exc:
            logger.error(f"Startup GEFS fetch failed for {target_date}: {exc}")

    t = threading.Thread(target=_background_refresh, daemon=True)
    t.start()
    logger.info("Background GEFS refresh thread started")


@app.get("/health")
def health():
    with _cache_lock:
        cache_dates = sorted(str(d) for d in _cache.keys())
    return {
        "status":        "ok",
        "cache_dates":   cache_dates,
        "model_version": MODEL_VERSION,
    }


@app.get("/predict")
def predict(ticker: str):
    city, target_date, floor_strike_f = _parse_ticker(ticker)

    if city is None or city not in PHILADELPHIA_CODES:
        raise HTTPException(404, f"Unsupported city '{city}' — only Philadelphia supported")
    if target_date is None:
        raise HTTPException(400, f"Cannot parse target date from ticker '{ticker}'")
    if floor_strike_f is None:
        raise HTTPException(400, f"Cannot parse floor_strike_f from ticker '{ticker}'")

    with _cache_lock:
        result = _cache.get(target_date)

    # No data yet (startup warmup still running or date not cached)
    if result is None:
        logger.warning(f"predict: cache miss for {target_date} ({ticker})")
        return {
            "probability":    0.5,
            "data_age_secs":  -1,
            "data_source_ok": False,
            "model_version":  MODEL_VERSION,
        }

    data_age_secs = int((datetime.now(timezone.utc) - result.fetch_time).total_seconds())

    if data_age_secs > MAX_DATA_AGE_SECS:
        logger.warning(
            f"predict: stale cache for {target_date} "
            f"(age={data_age_secs}s > max={MAX_DATA_AGE_SECS}s)"
        )
        return {
            "probability":    0.5,
            "data_age_secs":  data_age_secs,
            "data_source_ok": False,
            "model_version":  MODEL_VERSION,
        }

    prob = ensemble_predict(
        member_highs_f=result.member_highs_f,
        floor_strike_f=floor_strike_f,
        target_date=target_date,
        members=MEMBERS[:result.n_members],
    )

    logger.info(
        f"predict  ticker={ticker}  threshold={floor_strike_f}°F  "
        f"prob={prob:.4f}  members={result.n_members}  age={data_age_secs}s"
    )

    _write_prediction_log({
        "ts":             datetime.now(timezone.utc).isoformat(),
        "ticker":         ticker,
        "target_date":    str(target_date),
        "threshold_f":    floor_strike_f,
        "probability":    prob,
        "n_members":      result.n_members,
        "member_highs_f": result.member_highs_f,
        "run_time":       result.run_time.isoformat(),
        "data_age_secs":  data_age_secs,
        "model_version":  MODEL_VERSION,
    })

    return {
        "probability":    prob,
        "data_age_secs":  data_age_secs,
        "data_source_ok": True,
        "model_version":  MODEL_VERSION,
    }


# ── Entrypoint ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("WEATHER_SIDECAR_PORT", "8765"))
    host = os.getenv("WEATHER_SIDECAR_HOST", "127.0.0.1")
    uvicorn.run("sidecar:app", host=host, port=port, reload=False)
