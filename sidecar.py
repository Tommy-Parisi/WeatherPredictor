"""
WeatherPredictor sidecar — HTTP service for the Kalshi trading bot.

Loads the Philadelphia high-temp XGBoost model at startup, caches recent NOAA
data (TTL configurable), and serves calibrated P(high_temp > floor_strike_f)
for Kalshi weather market tickers.

Endpoints:
    GET /health              → {"status": "ok", "model_loaded": true}
    GET /predict?ticker=...  → {"prob": 0.73, "floor_strike_f": 85.0, "city": "PHI"}

Supported tickers: KXHIGH{PHI,PHIL,PHILLY,PHL}*  (Philadelphia high-temp only).

Environment variables:
    NOAA_API_TOKEN          Required. NOAA CDO API token.
    NOAA_STATION_ID         Optional. Default: GHCND:USW00013739 (PHL Airport).
    WEATHER_MODEL_PATH      Optional. Explicit model path; auto-selects newest if unset.
    WEATHER_SIDECAR_HOST    Optional. Default: 127.0.0.1
    WEATHER_SIDECAR_PORT    Optional. Default: 8765
    WEATHER_NOAA_CACHE_TTL_SECS  Optional. Default: 3600 (1 hour).
"""

import glob
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta
from typing import Optional

import joblib
import pandas as pd
import requests
from fastapi import FastAPI, HTTPException

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="WeatherPredictor Sidecar", version="1.0")

# ── Config ────────────────────────────────────────────────────────────────────

NOAA_API_TOKEN       = os.getenv("NOAA_API_TOKEN", "")
NOAA_STATION_ID      = os.getenv("NOAA_STATION_ID", "GHCND:USW00013739")
MODEL_PATH           = os.getenv("WEATHER_MODEL_PATH", "")
NOAA_CACHE_TTL_SECS  = int(os.getenv("WEATHER_NOAA_CACHE_TTL_SECS", "3600"))

# All city codes that map to Philadelphia in Kalshi tickers
PHILADELPHIA_CODES = {"PHI", "PHIL", "PHILLY", "PHL"}

# ── Runtime state ─────────────────────────────────────────────────────────────

_model        = None
_feature_cols = None
_noaa_cache: dict = {"df": None, "fetched_at": 0.0}

# ── Model loading ─────────────────────────────────────────────────────────────

def _find_latest_model() -> str:
    candidates = sorted(glob.glob("models/philly_weather_xgb_target_high_temp_yes_*.joblib"))
    if not candidates:
        raise FileNotFoundError(
            "No model files found in models/. Run train_weather_model.py first."
        )
    return candidates[-1]


def _load_model():
    global _model, _feature_cols
    path = MODEL_PATH or _find_latest_model()
    logger.info(f"Loading model from {path}")
    _model = joblib.load(path)

    # Load the feature column list saved alongside the model during training
    meta_path = re.sub(
        r"philly_weather_xgb_target_high_temp_yes_",
        "philly_weather_metrics_target_high_temp_yes_",
        path,
    ).replace(".joblib", ".json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        _feature_cols = meta.get("feature_columns")
        logger.info(f"Feature list loaded ({len(_feature_cols)} cols) from {meta_path}")
    else:
        _feature_cols = None
        logger.warning(f"No metrics JSON at {meta_path}; relying on model.feature_names_in_")
    logger.info("Model ready.")


# ── NOAA data ─────────────────────────────────────────────────────────────────

def _fetch_noaa_df() -> pd.DataFrame:
    """Fetch ~35 days of NOAA CDO data for Philadelphia in memory (no local CSV write)."""
    if not NOAA_API_TOKEN:
        raise RuntimeError("NOAA_API_TOKEN is not set")

    end_date   = datetime.utcnow().date()
    start_date = end_date - timedelta(days=35)

    resp = requests.get(
        "https://www.ncei.noaa.gov/cdo-web/api/v2/data",
        headers={"token": NOAA_API_TOKEN},
        params={
            "datasetid":      "GHCND",
            "stationid":      NOAA_STATION_ID,
            "startdate":      str(start_date),
            "enddate":        str(end_date),
            "datatypeid":     "TMAX,TMIN,PRCP",
            "limit":          1000,
            "units":          "metric",
            "includemetadata": "false",
        },
        timeout=15,
    )
    resp.raise_for_status()
    results = resp.json().get("results", [])
    if not results:
        raise RuntimeError("NOAA returned no results for the requested date range")

    df = pd.DataFrame(results)
    pivot = (
        df.pivot_table(index="date", columns="datatype", values="value", aggfunc="first")
        .reset_index()
        .rename(columns={"TMAX": "max_temp", "TMIN": "min_temp", "PRCP": "precipitation"})
    )
    pivot["date"] = pd.to_datetime(pivot["date"])
    pivot = pivot.sort_values("date").reset_index(drop=True)

    # CDO metric API already returns °C and mm — no unit conversion needed
    logger.info(
        f"NOAA: {len(pivot)} rows  "
        f"{pivot['date'].min().date()} – {pivot['date'].max().date()}  "
        f"max_temp {pivot['max_temp'].min():.1f}–{pivot['max_temp'].max():.1f}°C"
    )
    return pivot


def _get_noaa_df() -> pd.DataFrame:
    """Return cached NOAA data, refreshing if the cache is stale."""
    now = time.monotonic()
    if (
        _noaa_cache["df"] is not None
        and (now - _noaa_cache["fetched_at"]) < NOAA_CACHE_TTL_SECS
    ):
        return _noaa_cache["df"]
    logger.info("NOAA cache miss — fetching fresh data")
    df = _fetch_noaa_df()
    _noaa_cache["df"]         = df
    _noaa_cache["fetched_at"] = now
    return df


# ── Inference ─────────────────────────────────────────────────────────────────

def _build_features(noaa_df: pd.DataFrame, floor_strike_f: float) -> pd.DataFrame:
    from src.feature_engineering.feature_generator import WeatherFeatureGenerator
    gen    = WeatherFeatureGenerator()
    feat   = gen.generate_philly_features(noaa_df.copy(), floor_strike_f=None, save=False)
    feat["floor_strike_f"] = floor_strike_f
    return feat


def _run_inference(feat_df: pd.DataFrame) -> float:
    row = feat_df.tail(1).copy()

    if _feature_cols:
        for c in _feature_cols:
            if c not in row.columns:
                row[c] = 0.0
        row = row[_feature_cols]

    row = row.fillna(0).apply(pd.to_numeric, errors="coerce").fillna(0)
    return float(_model.predict_proba(row)[0, 1])


# ── Ticker parsing ────────────────────────────────────────────────────────────

def _parse_ticker(ticker: str) -> tuple[Optional[str], Optional[float]]:
    """
    Extract (city_code, floor_strike_f) from a Kalshi weather ticker.

    Examples:
        KXHIGHPHI-26APR15-T55   → ("PHI",  55.0)
        KXHIGHPHIL-25JUL31-T92  → ("PHIL", 92.0)
    """
    upper = ticker.upper()
    if not upper.startswith("KXHIGH"):
        return None, None

    rest     = upper[len("KXHIGH"):]
    city_end = next((i for i, c in enumerate(rest) if not c.isalpha()), len(rest))
    city     = rest[:city_end]

    m = re.search(r"-T(\d+)$", upper)
    return city, float(m.group(1)) if m else None


# ── Routes ────────────────────────────────────────────────────────────────────

@app.on_event("startup")
def on_startup():
    _load_model()


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


@app.get("/predict")
def predict(ticker: str):
    if _model is None:
        raise HTTPException(503, "Model not loaded")

    city, floor_strike_f = _parse_ticker(ticker)

    if city is None or city not in PHILADELPHIA_CODES:
        raise HTTPException(404, f"Unsupported city '{city}' — only Philadelphia (PHI/PHIL) supported")

    if floor_strike_f is None:
        raise HTTPException(400, f"Cannot parse floor_strike_f from ticker '{ticker}'")

    try:
        noaa_df = _get_noaa_df()
    except Exception as exc:
        logger.error(f"NOAA fetch failed: {exc}")
        raise HTTPException(502, f"NOAA data unavailable: {exc}")

    try:
        feat_df = _build_features(noaa_df, floor_strike_f)
        prob    = _run_inference(feat_df)
    except Exception as exc:
        logger.error(f"Inference error for {ticker}: {exc}", exc_info=True)
        raise HTTPException(500, f"Inference error: {exc}")

    logger.info(f"predict  ticker={ticker}  city={city}  threshold={floor_strike_f}°F  prob={prob:.4f}")
    return {"prob": prob, "floor_strike_f": floor_strike_f, "city": city, "ticker": ticker}


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("WEATHER_SIDECAR_PORT", "8765"))
    host = os.getenv("WEATHER_SIDECAR_HOST", "127.0.0.1")
    uvicorn.run("sidecar:app", host=host, port=port, reload=False)
