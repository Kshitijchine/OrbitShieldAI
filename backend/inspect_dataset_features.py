
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
CSV_PATH = (
    PROJECT_DIR
    / "data"
    / "kelvins_competition_data"
    / "train_data.csv"
)


def main():
    if not CSV_PATH.exists():
        print("ERROR: Dataset not found:", CSV_PATH)
        return

    df = pd.read_csv(CSV_PATH)

    print("=" * 72)
    print("ORBITSHIELDAI - HISTORICAL DATASET FEATURE AUDIT")
    print("=" * 72)
    print("Dataset rows:", len(df))
    print("Dataset columns:", len(df.columns))

    print("\nALL CSV COLUMN NAMES")
    print("-" * 72)
    for index, column in enumerate(df.columns, start=1):
        print(f"{index:03d}. {column}")

    print("\nFEATURE COVERAGE FOR THE 98 MODEL FEATURES")
    print("-" * 72)

    model_path = PROJECT_DIR / "backend" / "orbitshield_classifier.joblib"

    import joblib
    bundle = joblib.load(model_path)
    feature_names = bundle["feature_names"]

    rows = []
    for name in feature_names:
        if name not in df.columns:
            rows.append({
                "Feature": name,
                "In CSV": "NO",
                "Non-missing": 0,
                "Missing %": "100.0%",
            })
        else:
            non_missing = int(df[name].notna().sum())
            missing_pct = 100 * (1 - non_missing / len(df))
            rows.append({
                "Feature": name,
                "In CSV": "YES",
                "Non-missing": non_missing,
                "Missing %": f"{missing_pct:.1f}%",
            })

    report = pd.DataFrame(rows)
    print(report.to_string(index=False))

    print("\nAudit only. No dataset or model files were changed.")


if __name__ == "__main__":
    main()
