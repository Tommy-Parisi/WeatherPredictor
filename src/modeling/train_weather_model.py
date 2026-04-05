"""
Train the XGBoost weather model for Philadelphia.

Training design:
- One model handles all thresholds by including floor_strike_f as a feature.
- For each historical day, we create one training row per threshold in THRESHOLD_GRID_F,
  labelled by whether the actual high temp exceeded that threshold.
- Features derived from today's max_temp (lags are fine; same-day values are leaky).
"""

import glob
import json
import logging
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, log_loss, brier_score_loss
from sklearn.model_selection import cross_val_score

from src.data_collection.noaa_downloader import NOAADownloader
from src.feature_engineering.feature_generator import WeatherFeatureGenerator
from src.modeling.trainer import WeatherModelTrainer

logger = logging.getLogger(__name__)

# Threshold grid in °F — covers the full range of Kalshi high-temp markets.
# Includes all thresholds seen in historical Kalshi data plus a broader grid
# so the model generalises to markets we haven't seen yet.
THRESHOLD_GRID_F = [30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 87, 89, 90, 91, 92, 95, 98, 100]

# Features derived from today's actual max_temp — must be excluded to avoid leaking
# the answer into the model inputs.
LEAKY_FEATURES = {
    "max_temp",          # today's actual high in °C
    "max_temp_f",        # today's actual high in °F
    "max_temp_above_avg",
    "above_80F",
    "above_90F",
    "above_95F",
    "above_98F",
    # rolling/EMA windows include today's value
    "max_temp_ma_3",
    "max_temp_ma_7",
    "max_temp_ma_14",
    "max_temp_ma_30",
    "max_temp_ema_7",
    "max_temp_ema_14",
    # diff from today
    "max_temp_diff_1",
    "max_temp_diff_7",
    # volatility/trend windows include today
    "max_temp_volatility_7",
    "max_temp_volatility_30",
    "max_temp_trend_7",
    "max_temp_trend_14",
    # derived from today's max and min
    "temp_range",
    "temp_mean",
    "temp_range_ma_7",
    # target columns
    "target_high_temp_yes",
    "target_low_temp_yes",
}


def load_noaa_data() -> pd.DataFrame:
    """Load and deduplicate all cached NOAA CSV files.

    Existing CSVs were saved with a /10 bug in _process_noaa_data: the NOAA CDO
    metric API returns temperatures in Celsius but the downloader divided by 10
    again, so stored values are 10× too small. We detect and correct this on load
    so we don't have to re-download years of historical data.
    """
    files = sorted(glob.glob("data/noaa_philly_*.csv"))
    if not files:
        raise FileNotFoundError("No NOAA data files found in data/. Run data collection first.")
    dfs = [pd.read_csv(f) for f in files]
    df = pd.concat(dfs).drop_duplicates("date").sort_values("date").reset_index(drop=True)
    logger.info(f"Loaded {len(df)} unique daily records from {len(files)} NOAA files")
    logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Detect the /10 bug: Philadelphia max temps should reach at least 25°C (77°F)
    # in summer. If max(max_temp) < 10, the values are 10× too small.
    if "max_temp" in df.columns and df["max_temp"].max() < 10:
        logger.warning(
            f"max_temp max={df['max_temp'].max():.2f} looks like the /10 bug. "
            "Multiplying temperature columns by 10 to correct."
        )
        for col in ["max_temp", "min_temp"]:
            if col in df.columns:
                df[col] = df[col] * 10.0
        if "precipitation" in df.columns:
            df["precipitation"] = df["precipitation"] * 10.0
        # Recompute derived columns that were also saved with the bug
        if "max_temp" in df.columns and "min_temp" in df.columns:
            df["temp_range"] = df["max_temp"] - df["min_temp"]
        for col_f, threshold_c in [("above_80F", 26.67), ("above_90F", 32.22),
                                    ("above_95F", 35.00), ("above_98F", 36.67)]:
            if col_f in df.columns:
                df[col_f] = (df["max_temp"] > threshold_c).astype(int)
        logger.info(f"After correction: max_temp range {df['max_temp'].min():.1f}–{df['max_temp'].max():.1f}°C")

    return df


def build_training_df(weather_df: pd.DataFrame) -> pd.DataFrame:
    """
    Expand one weather row per day into N rows — one per threshold in THRESHOLD_GRID_F.

    Returns a DataFrame ready for model training with:
    - All lag/rolling/seasonal features (no same-day max_temp leakage)
    - floor_strike_f as a feature
    - target_high_temp_yes as the label
    """
    generator = WeatherFeatureGenerator()

    # Build base features without target (floor_strike_f=None skips target generation)
    base_df = generator.generate_philly_features(weather_df.copy(), floor_strike_f=None)

    # Compute max_temp_f separately so we can label each threshold row,
    # then drop it from the feature matrix.
    max_temp_f_series = base_df["max_temp"] * 9 / 5 + 32

    rows = []
    for threshold in THRESHOLD_GRID_F:
        chunk = base_df.copy()
        chunk["floor_strike_f"] = float(threshold)
        chunk["target_high_temp_yes"] = (max_temp_f_series > threshold).astype(int)
        rows.append(chunk)

    combined = pd.concat(rows, ignore_index=True)
    yes_rate = combined["target_high_temp_yes"].mean()
    logger.info(
        f"Expanded to {len(combined)} training rows across {len(THRESHOLD_GRID_F)} thresholds "
        f"(overall yes_rate={yes_rate:.3f})"
    )
    return combined


def select_features(df: pd.DataFrame) -> list[str]:
    """Return feature columns: everything except leaky columns, date, and target."""
    exclude = LEAKY_FEATURES | {"date"}
    return [c for c in df.columns if c not in exclude]


def evaluate(model, X, y, tag: str) -> dict:
    proba = model.predict_proba(X)[:, 1]
    auc = roc_auc_score(y, proba)
    ll = log_loss(y, proba)
    bs = brier_score_loss(y, proba)
    logger.info(f"  {tag}: AUC={auc:.4f}  LogLoss={ll:.4f}  Brier={bs:.4f}")
    return {"auc": auc, "log_loss": ll, "brier": bs}


def train_weather_model():
    """Train and save the Philadelphia high-temp weather model."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger.info("Starting weather model training for Philadelphia...")

    # 1. Load historical weather data
    weather_df = load_noaa_data()

    # Quick sanity check on temperatures
    max_f = weather_df["max_temp"] * 9 / 5 + 32
    logger.info(f"max_temp_f range in NOAA data: {max_f.min():.1f}°F to {max_f.max():.1f}°F")
    if max_f.max() < 60:
        raise ValueError(
            "max_temp_f looks wrong (max < 60°F). "
            "NOAA data may still contain the /10 bug — check noaa_downloader.py."
        )

    # 2. Build multi-threshold training dataset
    training_df = build_training_df(weather_df)

    # 3. Select features
    feature_cols = select_features(training_df)
    logger.info(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    X = training_df[feature_cols].copy()
    y = training_df["target_high_temp_yes"].copy()

    # Fill NaNs (lag features produce NaN for the first few rows)
    X = X.fillna(0).apply(pd.to_numeric, errors="coerce").fillna(0)
    y = y.astype(int)

    # 4. Time-based train/test split (last 20% of days as test)
    unique_dates = sorted(training_df["date"].unique())
    split_idx = int(len(unique_dates) * 0.8)
    split_date = unique_dates[split_idx]
    logger.info(f"Train/test split date: {split_date} ({split_idx}/{len(unique_dates)} days in train)")

    train_mask = training_df["date"] < split_date
    test_mask  = ~train_mask

    X_train, y_train = X[train_mask], y[train_mask]
    X_test,  y_test  = X[test_mask],  y[test_mask]
    logger.info(f"Train rows: {len(X_train)}  Test rows: {len(X_test)}")

    # 5. Train
    trainer = WeatherModelTrainer()
    model, metrics = trainer.train_model(
        pd.concat([X_train, y_train.rename("target_high_temp_yes")], axis=1),
        target_column="target_high_temp_yes",
    )

    # 6. Evaluate on held-out test set
    test_metrics = evaluate(model, X_test, y_test, "Test")

    # Per-threshold breakdown on test set
    logger.info("Per-threshold AUC on test set:")
    for threshold in THRESHOLD_GRID_F:
        mask = test_mask & (training_df["floor_strike_f"] == float(threshold))
        if mask.sum() < 10:
            continue
        xf = X[mask]
        yf = y[mask]
        xf = xf.fillna(0)
        proba = model.predict_proba(xf)[:, 1]
        try:
            auc = roc_auc_score(yf, proba)
            yes_r = yf.mean()
            logger.info(f"  {threshold}°F  n={mask.sum():4d}  yes_rate={yes_r:.3f}  AUC={auc:.4f}")
        except Exception:
            pass

    # 7. Save model with metadata
    os.makedirs("models", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/philly_weather_xgb_target_high_temp_yes_{timestamp}.joblib"
    meta_path  = f"models/philly_weather_metrics_target_high_temp_yes_{timestamp}.json"

    joblib.dump(model, model_path)

    full_metrics = {
        **metrics,
        "test_auc":      test_metrics["auc"],
        "test_log_loss": test_metrics["log_loss"],
        "test_brier":    test_metrics["brier"],
        "threshold_grid": THRESHOLD_GRID_F,
        "feature_columns": feature_cols,
        "train_rows": int(len(X_train)),
        "test_rows":  int(len(X_test)),
        "split_date": str(split_date),
        "trained_at": timestamp,
    }
    with open(meta_path, "w") as f:
        json.dump(full_metrics, f, indent=2)

    logger.info(f"Model saved  → {model_path}")
    logger.info(f"Metrics saved → {meta_path}")
    logger.info("Training complete.")
    return model, full_metrics


if __name__ == "__main__":
    train_weather_model()
