"""
GEFS ensemble fetcher for Philadelphia temperature forecasts.

Downloads TMP at 2m from all 31 GEFS members (control + 30 perturbed) via the
NOMADS GRIB filter service. Each request returns a tiny GRIB2 file (~260 bytes)
for the Philadelphia sub-region only. Requests are parallelised.

Public API
----------
fetch_ensemble_daily_highs(target_date) -> Optional[GEFSResult]
    Returns per-member predicted daily-high temperatures (°F) for target_date,
    or None if the GEFS run is unavailable or too few members succeed.
"""

import logging
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import eccodes
import numpy as np
import requests

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

GRIB_FILTER_URL = "https://nomads.ncep.noaa.gov/cgi-bin/filter_gefs_atmos_0p50a.pl"
GEFS_PROD_BASE  = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/gens/prod"

# PHL Airport (GHCND:USW00013739)
PHILLY_LAT      =  39.87
PHILLY_LON      = -75.24
PHILLY_LON_360  = PHILLY_LON % 360   # 284.76 — GRIB2 uses 0-360

# 31 members: control run + 30 perturbed
MEMBERS = ["c00"] + [f"p{i:02d}" for i in range(1, 31)]

# 3-hourly UTC hours sampled to find the calendar-day high.
# 03Z–27Z covers midnight-to-midnight Eastern (EDT UTC-4 / EST UTC-5) with margin.
SAMPLE_UTC_HOURS = list(range(3, 28, 3))   # [3, 6, 9, 12, 15, 18, 21, 24, 27]

# A GEFS run becomes fully available ~5 h after its nominal start time.
RUN_LAG_HOURS = 5

# Minimum members that must succeed for the result to be used.
MIN_MEMBERS_REQUIRED = 20

# Philly sub-region bounding box sent to the GRIB filter.
BBOX = {"toplat": "41.5", "leftlon": "-77.0", "rightlon": "-73.5", "bottomlat": "38.5"}

# Thread pool size — keeps NOMADS load reasonable while still being fast.
# 15 workers gives ~8-10s fetch time; higher values cause NOMADS to drop connections.
MAX_WORKERS = 15


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class GEFSResult:
    member_highs_f: list[float]          # per-member predicted daily high (°F)
    run_time: datetime                   # GEFS run used (UTC-aware)
    target_date: date
    fetch_time: datetime                 # when this result was produced (UTC-aware)
    n_members: int
    forecast_hours_used: list[int] = field(default_factory=list)


# ── Run discovery ─────────────────────────────────────────────────────────────

def _find_latest_available_run() -> Optional[tuple[date, int]]:
    """
    Return (run_date, cycle_hour) for the most recent GEFS run that is both
    published and has had enough time to fully populate.

    Checks in order: today 12z, today 06z, today 00z, yesterday 12z.
    """
    now = datetime.now(timezone.utc)
    candidates: list[tuple[date, int]] = []
    for delta_days in [0, 1]:
        d = now.date() - timedelta(days=delta_days)
        for cycle in [12, 6, 0]:
            run_dt = datetime(d.year, d.month, d.day, cycle, tzinfo=timezone.utc)
            if now >= run_dt + timedelta(hours=RUN_LAG_HOURS):
                candidates.append((d, cycle))

    for run_date, cycle in candidates:
        probe = (
            f"{GEFS_PROD_BASE}/gefs.{run_date.strftime('%Y%m%d')}"
            f"/{cycle:02d}/atmos/pgrb2ap5/"
            f"gec00.t{cycle:02d}z.pgrb2a.0p50.f006"
        )
        try:
            r = requests.head(probe, timeout=5)
            if r.status_code == 200:
                logger.info(f"GEFS: using run {run_date} {cycle:02d}z")
                return run_date, cycle
        except Exception:
            continue

    logger.error("GEFS: no run available on NOMADS")
    return None


# ── Forecast-hour selection ───────────────────────────────────────────────────

def _forecast_hours_for_date(run_date: date, run_cycle: int, target_date: date) -> list[int]:
    """
    Map SAMPLE_UTC_HOURS on target_date to GEFS forecast-hour offsets from the run.
    Skips hours that are ≤0 (before the run) or >384 (beyond GEFS horizon).
    """
    run_dt = datetime(run_date.year, run_date.month, run_date.day, run_cycle, tzinfo=timezone.utc)
    hours = []
    for utc_h in SAMPLE_UTC_HOURS:
        day_offset = utc_h // 24
        actual_h   = utc_h % 24
        target_dt  = datetime(
            target_date.year, target_date.month, target_date.day, actual_h,
            tzinfo=timezone.utc,
        ) + timedelta(days=day_offset)
        delta_h = round((target_dt - run_dt).total_seconds() / 3600 / 3) * 3
        if 3 <= delta_h <= 384 and delta_h not in hours:
            hours.append(delta_h)
    return sorted(hours)


# ── GRIB2 extraction ──────────────────────────────────────────────────────────

def _extract_phl_temp_k(grib2_bytes: bytes) -> Optional[float]:
    """Parse a GRIB2 message and return temperature (K) at the nearest PHL grid point."""
    with tempfile.NamedTemporaryFile(suffix=".grib2", delete=False) as tmp:
        tmp.write(grib2_bytes)
        tmp_path = tmp.name
    try:
        with open(tmp_path, "rb") as f:
            msg = eccodes.codes_grib_new_from_file(f)
        if msg is None:
            return None
        try:
            lats = eccodes.codes_get_array(msg, "latitudes")
            lons = eccodes.codes_get_array(msg, "longitudes")
            vals = eccodes.codes_get_array(msg, "values")
        finally:
            eccodes.codes_release(msg)
        dist = (lats - PHILLY_LAT) ** 2 + (lons - PHILLY_LON_360) ** 2
        return float(vals[int(np.argmin(dist))])
    except Exception as exc:
        logger.debug(f"GRIB2 parse error: {exc}")
        return None
    finally:
        os.unlink(tmp_path)


def _fetch_one(run_date: date, run_cycle: int, member: str, fhour: int) -> Optional[float]:
    """
    Fetch TMP@2m for one (member, forecast_hour) via the NOMADS GRIB filter.
    Returns temperature in Kelvin, or None on any failure.
    """
    date_str = run_date.strftime("%Y%m%d")
    filename  = f"ge{member}.t{run_cycle:02d}z.pgrb2a.0p50.f{fhour:03d}"
    dir_path  = f"/gefs.{date_str}/{run_cycle:02d}/atmos/pgrb2ap5"
    params = {
        "file":                 filename,
        "var_TMP":              "on",
        "lev_2_m_above_ground": "on",
        "subregion":            "",
        "dir":                  dir_path,
        **BBOX,
    }
    try:
        resp = requests.get(GRIB_FILTER_URL, params=params, timeout=15)
        resp.raise_for_status()
        if len(resp.content) < 100:   # filter returned an HTML error page
            return None
        return _extract_phl_temp_k(resp.content)
    except Exception as exc:
        logger.debug(f"fetch failed ge{member} f{fhour:03d}: {exc}")
        return None


# ── Public entry point ────────────────────────────────────────────────────────

def fetch_ensemble_daily_highs(target_date: date) -> Optional[GEFSResult]:
    """
    Fetch all 31 GEFS member forecasts for target_date.
    Returns per-member predicted daily-high temperatures (°F), or None on failure.
    """
    run_info = _find_latest_available_run()
    if run_info is None:
        return None
    run_date, run_cycle = run_info
    run_dt = datetime(run_date.year, run_date.month, run_date.day, run_cycle, tzinfo=timezone.utc)

    fhours = _forecast_hours_for_date(run_date, run_cycle, target_date)
    if not fhours:
        logger.error(f"GEFS: no valid forecast hours for {target_date} from {run_dt}")
        return None

    logger.info(
        f"GEFS fetch: run={run_dt.isoformat()} target={target_date} "
        f"hours={fhours} members={len(MEMBERS)}"
    )

    # Collect per-member temperature lists across all forecast hours
    member_temps: dict[str, list[float]] = {m: [] for m in MEMBERS}
    tasks = [(m, h) for m in MEMBERS for h in fhours]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(_fetch_one, run_date, run_cycle, m, h): (m, h)
            for m, h in tasks
        }
        for fut in as_completed(futures):
            member, _ = futures[fut]
            temp_k = fut.result()
            if temp_k is not None:
                member_temps[member].append(temp_k)

    # Per-member daily high = max over all sampled hours
    member_highs_k: list[float] = []
    for m in MEMBERS:
        if member_temps[m]:
            member_highs_k.append(max(member_temps[m]))
        else:
            logger.warning(f"GEFS: member {m} returned no data")

    if len(member_highs_k) < MIN_MEMBERS_REQUIRED:
        logger.error(
            f"GEFS: only {len(member_highs_k)}/{len(MEMBERS)} members succeeded — discarding"
        )
        return None

    member_highs_f = [(k - 273.15) * 9 / 5 + 32 for k in member_highs_k]
    logger.info(
        f"GEFS: {len(member_highs_f)} members  "
        f"range {min(member_highs_f):.1f}–{max(member_highs_f):.1f}°F"
    )

    return GEFSResult(
        member_highs_f=member_highs_f,
        run_time=run_dt,
        target_date=target_date,
        fetch_time=datetime.now(timezone.utc),
        n_members=len(member_highs_f),
        forecast_hours_used=fhours,
    )
