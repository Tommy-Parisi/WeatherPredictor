"""
Ensemble probability predictor for Philadelphia high-temp markets.

Takes per-member daily-high forecasts from the GEFS fetcher and computes
P(daily_high > floor_strike_f).

Bias correction
---------------
Each GEFS member is known to have systematic warm/cold biases for specific
cities and seasons. A bias correction table maps (month, member) → offset_f.
Until we have enough GEFS-vs-actuals history to estimate offsets, the table
is empty and corrections default to 0.0.

To populate: after ~60 days of logged GEFSResult objects, compute the mean
error per (month, member) against NOAA GHCND TMAX actuals and write the
offsets into BIAS_CORRECTIONS below.
"""

import logging
from datetime import date
from typing import Optional

logger = logging.getLogger(__name__)

# ── Bias correction table ──────────────────────────────────────────────────────
#
# Format: { month_int: { member_str: offset_f } }
# Positive offset = model runs warm (subtract from forecast to correct).
# Example once data is available:
#   BIAS_CORRECTIONS = {
#       4: {"c00": 1.2, "p01": 0.8, ...},   # April: GFS runs ~1°F warm
#       7: {"c00": -0.5, ...},               # July: GFS runs slightly cold
#   }
#
BIAS_CORRECTIONS: dict[int, dict[str, float]] = {}

# Data age threshold: reject if GEFS data is older than this many seconds.
MAX_DATA_AGE_SECS = 7_200   # 2 hours — matches sidecar TTL


def _apply_bias(member: str, month: int, raw_high_f: float) -> float:
    """Return bias-corrected temperature. Defaults to raw value until table is populated."""
    month_table = BIAS_CORRECTIONS.get(month, {})
    correction  = month_table.get(member, 0.0)
    return raw_high_f - correction


def predict(
    member_highs_f: list[float],
    floor_strike_f: float,
    target_date: Optional[date] = None,
    members: Optional[list[str]] = None,
) -> float:
    """
    Compute P(daily_high > floor_strike_f) from ensemble member forecasts.

    Parameters
    ----------
    member_highs_f : per-member predicted daily highs (°F), same order as `members`
    floor_strike_f : temperature threshold from the Kalshi ticker
    target_date    : used to look up the correct month for bias correction
    members        : member identifiers ("c00", "p01", ...) matching member_highs_f.
                     If None, bias correction is skipped (all corrections = 0.0).

    Returns
    -------
    Probability in (0, 1), clamped away from hard boundaries.
    """
    if not member_highs_f:
        raise ValueError("member_highs_f is empty")

    month = target_date.month if target_date else 0

    corrected = []
    for i, raw_f in enumerate(member_highs_f):
        member = members[i] if members and i < len(members) else ""
        corrected.append(_apply_bias(member, month, raw_f))

    votes_yes = sum(1 for t in corrected if t > floor_strike_f)
    raw_prob  = votes_yes / len(corrected)

    # Clamp away from 0.0 / 1.0 — a 31-member ensemble can only express
    # multiples of 1/31 ≈ 0.032, so 0 or 31 votes is genuinely informative
    # but we still avoid feeding hard boundaries to the policy layer.
    prob = max(0.005, min(0.995, raw_prob))

    logger.debug(
        f"ensemble predict: threshold={floor_strike_f}°F  "
        f"votes={votes_yes}/{len(corrected)}  raw={raw_prob:.3f}  clamped={prob:.3f}"
    )
    return prob
