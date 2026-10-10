
import pandas as pd
from pathlib import Path

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "kelvins_competition_data"
    / "train_data.csv"
)


def prepare_event_dataset():
    print("Loading conjunction dataset...")

    df = pd.read_csv(DATA_PATH)

    required = {"event_id", "time_to_tca", "risk"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    X_rows = []
    y_values = []
    event_ids = []

    excluded = {
        "event_id",
        "risk",
        "max_risk_estimate",
        "max_risk_scaling",
    }

    # Process each conjunction event separately.
    for event_id, event in df.groupby("event_id", sort=False):

        # Largest time_to_tca means the earliest observation.
        event = event.sort_values(
            "time_to_tca",
            ascending=False,
        )

        # Need at least two records for this event.
        if len(event) < 2:
            continue

        # Find observations at least 2 days before TCA.
        early_candidates = event[event["time_to_tca"] >= 2]

        if early_candidates.empty:
            continue

        # Use the earliest available observation at or before
        # the 2-day prediction boundary (time_to_tca >= 2).
        early = early_candidates.iloc[0]

        # Target: risk from the final recorded observation.
        latest = event.iloc[-1]

        # Remove identifiers and risk-related columns from inputs.
        features = early.drop(
            labels=[
                column
                for column in excluded
                if column in event.columns
            ]
        )

        X_rows.append(features)
        y_values.append(latest["risk"])
        event_ids.append(event_id)

    if not X_rows:
        raise ValueError(
            "No eligible events found with multiple records "
            "and an observation at least 2 days before TCA."
        )

    X = pd.DataFrame(X_rows, index=event_ids)
    X.index.name = "event_id"

    y = pd.Series(
        y_values,
        index=event_ids,
        name="target_final_recorded_risk",
    )
    y.index.name = "event_id"

    # Keep numeric features only.
    X = X.select_dtypes(include="number").copy()

    # Replace infinite values and remove entirely empty columns.
    X = X.replace([float("inf"), float("-inf")], float("nan"))
    X = X.dropna(axis=1, how="all")

    print("\nEARLY-OBSERVATION DATASET READY")
    print("Eligible events:", len(X))
    print("Feature columns:", X.shape[1])
    print("Target:", y.name)
    print("Missing target values:", y.isna().sum())

    print("\nExample target values:")
    print(y.head().to_string())

    return X, y


if __name__ == "__main__":
    prepare_event_dataset()
