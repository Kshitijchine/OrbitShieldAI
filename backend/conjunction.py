
from datetime import datetime, timedelta, timezone
import sys

import numpy as np
import requests
from skyfield.api import EarthSatellite, load

FINE_STEP_SECONDS = 5
COARSE_STEP_SECONDS = 60
DEFAULT_HOURS = 24
SCREENING_THRESHOLD_KM = 5.0
TLE_STALE_DAYS = 7


def get_satellite_tle(norad_id):
    """Download and validate public TLE data for one satellite."""
    url = (
        "https://celestrak.org/NORAD/elements/gp.php"
        f"?CATNR={norad_id}&FORMAT=TLE"
    )
    response = requests.get(
        url,
        timeout=30,
        headers={"User-Agent": "OrbitShieldAI/1.0"},
    )
    response.raise_for_status()

    lines = [line.strip() for line in response.text.splitlines()
             if line.strip()]
    if len(lines) < 3 or not lines[1].startswith("1 ") or not lines[2].startswith("2 "):
        raise ValueError(f"Could not find valid TLE data for NORAD ID {norad_id}")

    if lines[1][2:7].strip() != str(int(norad_id)):
        raise ValueError(f"TLE NORAD ID mismatch for {norad_id}")
    if lines[2][2:7].strip() != str(int(norad_id)):
        raise ValueError(f"TLE NORAD ID mismatch for {norad_id}")

    return lines[0], lines[1], lines[2]


def tle_epoch_utc(line1):
    """Parse the epoch from a standard two-line element set."""
    try:
        year = int(line1[18:20])
        year += 2000 if year < 57 else 1900
        day_of_year = float(line1[20:32])
        return datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(
            days=day_of_year - 1
        )
    except (ValueError, IndexError) as exc:
        raise ValueError("Unable to parse TLE epoch") from exc


def separation_km(satellite_a, satellite_b, ts, moment):
    time = ts.from_datetime(moment)
    pos_a = satellite_a.at(time).position.km
    pos_b = satellite_b.at(time).position.km
    return float(np.linalg.norm(pos_a - pos_b))


def _refine_candidate(sat_a, sat_b, ts, start, end):
    """Refine a candidate minimum with a 5-second grid."""
    count = int((end - start).total_seconds() / FINE_STEP_SECONDS) + 1
    dates = [
        start + timedelta(seconds=i * FINE_STEP_SECONDS)
        for i in range(count)
    ]
    if dates[-1] < end:
        dates.append(end)

    times = ts.from_datetimes(dates)
    pos_a = sat_a.at(times).position.km
    pos_b = sat_b.at(times).position.km
    distances = np.linalg.norm(pos_a - pos_b, axis=0)
    index = int(np.argmin(distances))
    return dates[index], float(distances[index])


def find_closest_approach(
    satellite_a_id,
    satellite_b_id,
    hours=DEFAULT_HOURS,
):
    """Screen a time window and refine each sampled local minimum."""
    satellite_a_id = int(satellite_a_id)
    satellite_b_id = int(satellite_b_id)

    if satellite_a_id == satellite_b_id:
        raise ValueError("Please enter two different satellite NORAD IDs")
    if hours <= 0:
        raise ValueError("hours must be greater than zero")

    ts = load.timescale()
    name_a, line1_a, line2_a = get_satellite_tle(satellite_a_id)
    name_b, line1_b, line2_b = get_satellite_tle(satellite_b_id)

    epoch_a = tle_epoch_utc(line1_a)
    epoch_b = tle_epoch_utc(line1_b)
    now = datetime.now(timezone.utc)
    age_a_days = (now - epoch_a).total_seconds() / 86400
    age_b_days = (now - epoch_b).total_seconds() / 86400

    if age_a_days < 0 or age_b_days < 0:
        raise ValueError("A TLE epoch is in the future; check source data")

    sat_a = EarthSatellite(line1_a, line2_a, name_a, ts)
    sat_b = EarthSatellite(line1_b, line2_b, name_b, ts)

    start = now
    end = start + timedelta(hours=hours)
    count = int(hours * 3600 / COARSE_STEP_SECONDS)
    coarse_dates = [
        start + timedelta(seconds=i * COARSE_STEP_SECONDS)
        for i in range(count + 1)
    ]
    if coarse_dates[-1] < end:
        coarse_dates.append(end)

    coarse_times = ts.from_datetimes(coarse_dates)
    pos_a = sat_a.at(coarse_times).position.km
    pos_b = sat_b.at(coarse_times).position.km
    distances = np.linalg.norm(pos_a - pos_b, axis=0)

    # Identify all sampled local minima, plus both window endpoints.
    candidates = {0, len(distances) - 1}
    for i in range(1, len(distances) - 1):
        if distances[i] <= distances[i - 1] and distances[i] <= distances[i + 1]:
            candidates.add(i)

    refined = []
    for index in sorted(candidates):
        left = max(start, coarse_dates[index] - timedelta(
            seconds=COARSE_STEP_SECONDS
        ))
        right = min(end, coarse_dates[index] + timedelta(
            seconds=COARSE_STEP_SECONDS
        ))
        refined.append(
            _refine_candidate(sat_a, sat_b, ts, left, right)
        )

    closest_time, closest_distance = min(refined, key=lambda item: item[1])
    stale = age_a_days > TLE_STALE_DAYS or age_b_days > TLE_STALE_DAYS

    return {
        "satellite_a_id": satellite_a_id,
        "satellite_b_id": satellite_b_id,
        "satellite_a": name_a,
        "satellite_b": name_b,
        "satellite_a_tle_epoch_utc": epoch_a.isoformat(),
        "satellite_b_tle_epoch_utc": epoch_b.isoformat(),
        "satellite_a_tle_age_days": round(age_a_days, 3),
        "satellite_b_tle_age_days": round(age_b_days, 3),
        "tle_stale_warning": stale,
        "closest_approach_utc": closest_time.isoformat(),
        "minimum_sampled_separation_km": closest_distance,
        "coarse_sampling_interval_seconds": COARSE_STEP_SECONDS,
        "fine_sampling_interval_seconds": FINE_STEP_SECONDS,
        "candidate_minima_refined": len(candidates),
        "window_hours": hours,
        "assessment_type": "screening estimate from public TLEs",
        "warning": (
            "Not a certified conjunction assessment or collision probability. "
            "TLE uncertainty and finite sampling may miss events. "
            "Stale TLEs reduce confidence."
        ),
    }


def recommend_simulated_maneuver(
    separation_km,
    threshold_km=SCREENING_THRESHOLD_KM,
    simulated_delta_v_m_s=0.1,
):
    """Return an illustrative flag, not an actual maneuver plan."""
    if separation_km < 0:
        raise ValueError("separation_km cannot be negative")
    if threshold_km <= 0:
        raise ValueError("threshold_km must be greater than zero")
    if simulated_delta_v_m_s <= 0:
        raise ValueError("simulated delta-v must be greater than zero")

    if separation_km < threshold_km:
        return {
            "screening_flag": "REVIEW_REQUIRED",
            "simulated_delta_v_m_s": simulated_delta_v_m_s,
            "direction": "NOT_COMPUTED",
            "timing": "NOT_COMPUTED",
            "note": (
                "Illustrative threshold flag only. Qualified review of "
                "validated conjunction data is required before any action."
            ),
        }

    return {
        "screening_flag": "NO_THRESHOLD_TRIGGER",
        "simulated_delta_v_m_s": 0.0,
        "direction": "NONE",
        "timing": "NONE",
        "note": (
            "The sampled separation did not cross the configured threshold. "
            "This does not prove the event is safe."
        ),
    }


def main():
    try:
        if len(sys.argv) >= 3:
            satellite_a_id = int(sys.argv[1])
            satellite_b_id = int(sys.argv[2])
        else:
            print("OrbitShieldAI - Satellite Conjunction Screening")
            satellite_a_id = int(
                input("First satellite NORAD ID (default 25544): ") or "25544"
            )
            satellite_b_id = int(
                input("Second satellite NORAD ID (default 20580): ") or "20580"
            )

        result = find_closest_approach(
            satellite_a_id, satellite_b_id, hours=DEFAULT_HOURS
        )

        print("\n--- Close Approach Screening Report ---")
        for key, value in result.items():
            print(f"{key}: {value}")

        maneuver = recommend_simulated_maneuver(
            result["minimum_sampled_separation_km"]
        )
        print("\n--- Illustrative Maneuver Flag ---")
        for key, value in maneuver.items():
            print(f"{key}: {value}")

    except requests.RequestException as exc:
        print("Could not download public TLE data:", exc)
    except (ValueError, RuntimeError) as exc:
        print("Screening calculation failed:", exc)
    except Exception as exc:
        print(f"Unexpected screening error: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
