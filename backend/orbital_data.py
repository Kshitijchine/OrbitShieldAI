from datetime import datetime, timedelta, timezone
import argparse
import sys
from urllib.request import Request, urlopen

import numpy as np
from skyfield.api import EarthSatellite, load, wgs84

DEFAULT_SATELLITE_A = 25544  # ISS
DEFAULT_SATELLITE_B = 20580  # Hubble Space Telescope
DEFAULT_HOURS = 24
COARSE_STEP_SECONDS = 60
FINE_STEP_SECONDS = 5
TLE_STALE_DAYS = 7


def get_satellite_tle(norad_id):
    """Download and validate a public TLE for a satellite."""
    norad_id = int(norad_id)
    if norad_id <= 0:
        raise ValueError("NORAD ID must be a positive integer.")

    url = (
        "https://celestrak.org/NORAD/elements/gp.php"
        f"?CATNR={norad_id}&FORMAT=TLE"
    )
    request = Request(url, headers={"User-Agent": "OrbitShieldAI/1.0"})

    with urlopen(request, timeout=30) as response:
        lines = [
            line.strip()
            for line in response.read().decode("utf-8").splitlines()
            if line.strip()
        ]

    if len(lines) < 3:
        raise ValueError(f"TLE data is incomplete for NORAD ID {norad_id}.")

    name, line1, line2 = lines[:3]
    if not line1.startswith("1 ") or not line2.startswith("2 "):
        raise ValueError(f"Invalid TLE format for NORAD ID {norad_id}.")

    if line1[2:7].strip() != str(norad_id):
        raise ValueError(f"TLE NORAD ID mismatch for {norad_id}.")
    if line2[2:7].strip() != str(norad_id):
        raise ValueError(f"TLE NORAD ID mismatch for {norad_id}.")

    return name, line1, line2


def tle_epoch_utc(line1):
    """Read the UTC epoch from a standard TLE line 1."""
    try:
        year = int(line1[18:20])
        year += 2000 if year < 57 else 1900
        day_of_year = float(line1[20:32])
        return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(
            days=day_of_year - 1
        )
    except (ValueError, IndexError) as exc:
        raise ValueError("Could not parse the TLE epoch.") from exc


def satellite_state(satellite, ts, moment):
    """Return geocentric position and velocity at a UTC instant."""
    position = satellite.at(ts.from_datetime(moment))
    subpoint = wgs84.subpoint(position)
    return {
        "latitude_deg": float(subpoint.latitude.degrees),
        "longitude_deg": float(subpoint.longitude.degrees),
        "altitude_km": float(subpoint.elevation.km),
        "position_km": np.asarray(position.position.km, dtype=float),
        "velocity_km_s": np.asarray(position.velocity.km_per_s, dtype=float),
    }


def _separation_samples(sat_a, sat_b, ts, dates):
    times = ts.from_datetimes(dates)
    pos_a = sat_a.at(times).position.km
    pos_b = sat_b.at(times).position.km
    return np.linalg.norm(pos_a - pos_b, axis=0)


def _refine_minimum(sat_a, sat_b, ts, start, end):
    duration = (end - start).total_seconds()
    count = int(duration // FINE_STEP_SECONDS)
    dates = [
        start + timedelta(seconds=i * FINE_STEP_SECONDS)
        for i in range(count + 1)
    ]
    if dates[-1] < end:
        dates.append(end)

    distances = _separation_samples(sat_a, sat_b, ts, dates)
    index = int(np.argmin(distances))
    return dates[index], float(distances[index])