import subprocess
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from backend.conjunction import find_closest_approach, recommend_simulated_maneuver

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="OrbitShieldAI",
    page_icon="🛰️",
    layout="wide",
)

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "backend" / "orbitshield_classifier.joblib"
SCRIPT_PATH = PROJECT_DIR / "backend" / "conjunction.py"
PYTHON_PATH = PROJECT_DIR / ".venv313" / "Scripts" / "python.exe"
SCREENING_TIMEOUT_SECONDS = 300

# Verified results from the saved model's training-time held-out partition.
# These are historical classification metrics, not collision probabilities.
EVALUATION_EVENTS = 2385
BALANCED_ACCURACY = 0.6877
ROC_AUC = 0.7338
CLASS_0_PRECISION = 0.79
CLASS_0_RECALL = 0.71
CLASS_1_PRECISION = 0.56
CLASS_1_RECALL = 0.67

st.title("🛰️ OrbitShieldAI")
st.subheader("AI-Assisted Satellite Collision Screening")
st.write(
    "A research demonstration for satellite conjunction screening "
    "using machine learning and publicly available orbital data."
)

st.info(
    "Research prototype only. Historical machine-learning classification "
    "and live orbital screening are separate components. This application "
    "does not calculate a calibrated collision probability or provide "
    "operational maneuver instructions."
)


# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

@st.cache_resource
def load_model(model_path: str, modified_time: float):
    """Load and validate the saved model bundle."""
    bundle = joblib.load(model_path)

    if not isinstance(bundle, dict):
        raise ValueError("The model file does not contain a valid model bundle.")

    if "model" not in bundle or "feature_names" not in bundle:
        raise ValueError(
            "The model bundle must contain 'model' and 'feature_names'."
        )

    if not bundle["feature_names"]:
        raise ValueError("The model bundle contains no feature names.")

    return bundle


st.divider()
st.header("🤖 Machine Learning Model")

bundle = None
model = None
feature_names = []

if not MODEL_PATH.is_file():
    st.error(f"Model file not found: {MODEL_PATH}")
else:
    try:
        bundle = load_model(
            str(MODEL_PATH),
            MODEL_PATH.stat().st_mtime,
        )
        model = bundle["model"]
        feature_names = bundle["feature_names"]

        st.success("Trained model loaded successfully!")

        col1, col2, col3 = st.columns(3)
        col1.metric(
            "Model type",
            str(bundle.get("model_type", type(model).__name__)),
        )
        col2.metric("Input features", len(feature_names))
        col3.metric("Model status", "Loaded")

        st.write(
            "**Training target:**",
            bundle.get("target", "Not specified"),
        )

        st.warning(
            "The model predicts a historical dataset category, not a "
            "calibrated collision probability. Do not use its output "
            "for real satellite maneuver decisions."
        )

        with st.expander("About the model and training data"):
            st.markdown(
                """
                - Trained on historical conjunction records.
                - Class 0 means the final recorded risk value was less
                  than or equal to -30.
                - Class 1 means the final recorded risk value was greater
                  than -30.
                - These classes are dataset labels, not direct estimates
                  of collision probability.
                - The model expects 98 features in the exact order used
                  during training.
                """
            )

        st.subheader("Historical Model Evaluation")

        metric1, metric2, metric3 = st.columns(3)
        metric1.metric(
            "Balanced accuracy",
            f"{BALANCED_ACCURACY:.2%}",
        )
        metric2.metric(
            "ROC-AUC",
            f"{ROC_AUC:.4f}",
        )
        metric3.metric(
            "Evaluation events",
            f"{EVALUATION_EVENTS:,}",
        )

        st.caption(
            "Metrics reproduced on the training script's held-out "
            "partition. This is a consistency check, not independent "
            "validation on a separate dataset."
        )

        with st.expander("View historical classification details"):
            st.markdown(
                """
                | Metric | Class 0 | Class 1 |
                |---|---:|---:|
                | Precision | 0.79 | 0.56 |
                | Recall | 0.71 | 0.67 |

                **Interpretation**

                - Balanced accuracy averages recall across both classes.
                - ROC-AUC measures how well model scores distinguish
                  the two historical label classes.
                - Class 1 is not a confirmed collision event.
                - These metrics do not establish collision probability,
                  operational safety, or maneuver effectiveness.
                """
            )

        st.warning(
            "Independent test evaluation on 2,092 eligible events "
            "produced ROC-AUC 0.6438 and Brier score 0.2113. "
            "A constant-rate baseline achieved a better Brier score "
            "of 0.1696. The model's scores should not be interpreted "
            "as calibrated probabilities or used for operational decisions."
        )

        st.subheader("Model Integration Status")

        st.info(
            "The model is loaded, but it is not connected to live orbital "
            "screening. The live TLE script does not produce all the "
            "historical features required by this classifier."
        )

        st.write("**Current capabilities**")
        st.markdown(
            """
            - Load the saved machine-learning model.
            - Retrieve public TLE records through the orbital screening script.
            - Estimate sampled separation between two satellites.
            - Display the screening report and its limitations.
            """
        )

        st.write("**Not yet implemented**")
        st.markdown(
            """
            - Map real conjunction observations to all 98 model features.
            - Validate feature units, meanings, and missing-value handling.
            - Generate predictions from verified, compatible conjunction records.
            - Independently validate and calibrate predictions against outcomes.
            """
        )

        st.caption(
            "No prediction is generated from artificial or zero-filled "
            "orbital inputs."
        )

    except Exception as error:
        st.error("The model could not be loaded.")
        st.code(f"{type(error).__name__}: {error}")


# --------------------------------------------------
# SCREENING REPORT HELPERS
# --------------------------------------------------

def parse_report(report: str) -> dict:
    """Convert key:value lines in a report into a dictionary."""
    parsed = {}

    for line in report.splitlines():
        line = line.strip()

        if not line or line.startswith("---") or ":" not in line:
            continue

        key, value = line.split(":", 1)
        parsed[key.strip()] = value.strip()

    return parsed


def format_utc(value: str) -> str:
    """Format a timestamp as UTC, preserving it if parsing fails."""
    try:
        timestamp = pd.to_datetime(value, utc=True)
        return timestamp.strftime("%b %d, %Y · %H:%M:%S UTC")
    except (TypeError, ValueError, OverflowError):
        return value


def display_screening_report(report: str) -> None:
    """Display a summary and the complete technical report."""
    data = parse_report(report)

    satellite_a = data.get(
        "satellite_a",
        data.get("satellite_a_id", "Unknown"),
    )
    satellite_b = data.get(
        "satellite_b",
        data.get("satellite_b_id", "Unknown"),
    )
    separation = data.get(
        "minimum_sampled_separation_km",
        "Not reported",
    )
    closest_approach = data.get(
        "closest_approach_utc",
        "Not reported",
    )
    window_hours = data.get("window_hours", "Not reported")
    threshold_flag = data.get("screening_flag", "Not reported")

    try:
        separation_text = f"{float(separation):,.2f} km"
    except (TypeError, ValueError):
        separation_text = separation

    try:
        window_text = f"{float(window_hours):g} hours"
    except (TypeError, ValueError):
        window_text = window_hours

    st.divider()
    st.subheader("🛰️ Close Approach Screening Results")

    st.caption(f"Satellite {satellite_a} · Satellite {satellite_b}")

    col1, col2, col3 = st.columns(3)
    col1.metric("Minimum sampled separation", separation_text)
    col2.metric("Screening window", window_text)
    col3.metric("Threshold assessment", threshold_flag.replace("_", " ").title())

    summary = pd.DataFrame(
        [
            {
                "Parameter": "Estimated closest approach",
                "Result": format_utc(closest_approach),
            },
            {
                "Parameter": "Satellite A TLE age",
                "Result": (
                    f"{data.get('satellite_a_tle_age_days', 'Not reported')} days"
                ),
            },
            {
                "Parameter": "Satellite B TLE age",
                "Result": (
                    f"{data.get('satellite_b_tle_age_days', 'Not reported')} days"
                ),
            },
            {
                "Parameter": "Threshold assessment",
                "Result": threshold_flag.replace("_", " ").title(),
            },
        ]
    )

    st.table(summary)

    stale_warning = data.get("tle_stale_warning", "").lower()
    if stale_warning == "true":
        st.warning(
            "One or more TLE records are flagged as stale. "
            "Treat the screening estimate with additional caution."
        )

    if threshold_flag == "NO_THRESHOLD_TRIGGER":
        st.info(
            "No configured screening threshold trigger was detected. "
            "This does not establish that the satellites are safe."
        )
    elif threshold_flag == "REVIEW_REQUIRED":
        st.warning(
            "The illustrative threshold was triggered. "
            "This is not a validated maneuver recommendation."
        )
    elif threshold_flag == "Not reported":
        st.warning("The screening script did not report a threshold result.")

    st.caption(
        "Research estimate from public orbital data and finite sampling. "
        "It is not a certified conjunction assessment or collision probability."
    )

    with st.expander("View full technical report"):
        st.code(report, language="text")


# --------------------------------------------------
# ORBITAL SCREENING
# --------------------------------------------------
def run_orbital_screening(
    satellite_a_id: int,
    satellite_b_id: int,
) -> None:
    """Run orbital screening directly and display its report."""

    satellite_a_id = int(satellite_a_id)
    satellite_b_id = int(satellite_b_id)

    if satellite_a_id == satellite_b_id:
        st.warning("Please enter two different satellite NORAD IDs.")
        return

    try:
        with st.spinner(
            "Retrieving public TLE data and estimating orbital separation..."
        ):
            result = find_closest_approach(
                satellite_a_id,
                satellite_b_id,
            )

            maneuver = recommend_simulated_maneuver(
                result["minimum_sampled_separation_km"]
            )

        report_lines = [
            f"{key}: {value}" for key, value in result.items()
        ]
        report_lines.extend(
            f"{key}: {value}" for key, value in maneuver.items()
        )
        report = "\n".join(report_lines)

        st.success("Screening run completed.")
        display_screening_report(report)

    except Exception as error:
        st.error("Could not run orbital screening.")
        st.code(f"{type(error).__name__}: {error}")

# --------------------------------------------------
# CONFIGURABLE SATELLITE SCREENING
# --------------------------------------------------

st.divider()
st.header("🛰️ Configurable Satellite Screening")
st.write(
    "Enter two NORAD catalog IDs to estimate their closest sampled "
    "separation over the configured screening window."
)

with st.form("satellite_screening_form"):
    col1, col2 = st.columns(2)

    with col1:
        satellite_a_id = st.number_input(
            "First satellite NORAD ID",
            min_value=1,
            value=25544,
            step=1,
            key="screen_satellite_a",
        )

    with col2:
        satellite_b_id = st.number_input(
            "Second satellite NORAD ID",
            min_value=1,
            value=20580,
            step=1,
            key="screen_satellite_b",
        )

    submit_screening = st.form_submit_button(
        "Screen Selected Satellites",
        type="primary",
    )

if submit_screening:
    run_orbital_screening(satellite_a_id, satellite_b_id)


# --------------------------------------------------
# LIMITATIONS
# --------------------------------------------------

st.divider()
st.header("⚠️ Screening Limitations")

with st.expander("Read important safety and accuracy notes"):
    st.markdown(
        """
        - **Public orbital data:** TLE records are estimates, not live
          precision telemetry.
        - **Finite sampling:** Coarse-to-fine sampling can miss close
          approaches between samples.
        - **TLE age:** Older orbital elements can reduce confidence in
          predicted positions.
        - **Illustrative threshold:** The configured 5 km threshold is not
          an operational safety limit.
        - **No validated maneuver planning:** The illustrative maneuver
          flag does not establish a safe maneuver direction, timing,
          or delta-v.
        - **Machine-learning limitations:** The classifier predicts a
          historical dataset label; it does not calculate a calibrated
          collision probability.
        - **No safety guarantee:** A large estimated separation or an
          untriggered threshold does not prove a satellite pair is safe.
        - **Research use only:** Do not use this prototype to make real
          satellite maneuver decisions.
        """
    )

st.caption("OrbitShieldAI · Research and educational demonstration")