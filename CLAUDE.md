# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Train the XGBoost model (requires NOAA data in data/philly_features_*.csv)
python main.py train

# Run backtesting (requires a trained model in models/)
python main.py backtest --env demo

# Run live trading
python main.py trade --env demo    # demo environment
python main.py trade --env prod    # production environment
```

## Environment Setup

Copy `.env.example` to `.env` and populate:
- `DEMO_KEYID` / `DEMO_KEYFILE` — Kalshi demo API key ID and path to PEM private key file
- `PROD_KEYID` / `PROD_KEYFILE` — Kalshi production credentials
- `NOAA_API_TOKEN` — NOAA API token for historical weather data

Authentication uses RSA-PSS signing. The private key file must be a PEM-encoded RSA key.

## Role

This repo is a **prediction sidecar** to a separate Kalshi trading bot. Its only responsibilities are:
1. Train and maintain per-city XGBoost weather models
2. Run predictions against live/recent weather data
3. Pipe probability outputs to the trading bot via HTTP (the sidecar exposes an HTTP server for this)

It does **not** manage order execution, portfolio state, or market scanning for trading decisions — that lives in the parent bot.

## Architecture

**Data flow:**
1. `src/data_collection/kalshi_scanner.py` — Scans Kalshi API for open weather markets by city, saves snapshots to `data/`
2. `src/data_collection/noaa_downloader.py` — Downloads historical weather data from NOAA per city station
3. `src/feature_engineering/feature_generator.py` — Transforms raw weather data into ML features
4. `src/modeling/trainer.py` — Trains one XGBoost binary classifier **per city**, saves to `models/<city>_weather_xgb_<target>_<timestamp>.joblib`
5. `src/modeling/predictor.py` — Loads a city-specific model and returns probabilities
6. `src/backtesting/backtester.py` — Walk-forward backtesting with Kelly-based position sizing; outputs to `backtests/`
7. `src/trading/trader.py` — Currently a stub; prediction output is piped to the parent trading bot over HTTP

**Kalshi API client** (`clients.py`):
- `KalshiHttpClient` — authenticated REST client with built-in 100ms rate limiting
- `KalshiWebSocketClient` — async WebSocket client for real-time ticker subscriptions
- Demo base URL: `https://demo-api.kalshi.co`; Prod: `https://api.elections.kalshi.com`

**City configuration** (`config.py`):
One model is trained and maintained per city. Each city maps to a NOAA station ID. Currently only Philadelphia is active; the design supports expanding to any city with a Kalshi weather market:

```python
TARGET_CITIES = {
    "Philadelphia": {"station_id": "GHCND:USW00013739"},  # PHL Airport
    # "New York":   {"station_id": "GHCND:USW00094728"},  # JFK Airport
    # "Chicago":    {"station_id": "GHCND:USW00094846"},  # O'Hare Airport
    # "Los Angeles":{"station_id": "GHCND:USW00023174"},  # LAX Airport
    # "Miami":      {"station_id": "GHCND:USW00012839"},  # MIA Airport
    # "Dallas":     {"station_id": "GHCND:USW00003927"},  # DFW Airport
    # "Atlanta":    {"station_id": "GHCND:USW00013874"},  # ATL Airport
    # "Boston":     {"station_id": "GHCND:USW00014739"},  # BOS Airport
}
```

To add a city: uncomment it in `config.py`, run `train` to build its model, and the scanner/predictor will automatically handle it.

**Trading parameters** (`config.py`):
- `MINIMUM_EDGE_CENTS = 7` — minimum edge over market price to flag a trade opportunity
- `MAX_RISK_PER_TRADE = 0.05` — Kelly fraction cap (5% of portfolio per trade)

**Model naming convention:** `models/<city>_weather_xgb_<target_column>_<YYYYMMDD_HHMMSS>.joblib`

The backtester auto-discovers the most recent model per city matching that pattern.
