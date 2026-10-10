from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.dataset_loader import prepare_event_dataset


def main():
    print("=" * 65)
    print("ORBITSHIELD AI - MODEL FEATURE MAPPING AUDIT")
    print("=" * 65)

    try:
        X, y = prepare_event_dataset()
    except Exception as exc:
        print(f"\nERROR loading training dataset: {exc}")
        return

    print(f"\nDataset feature count: {X.shape[1]}")
    print(f"Dataset row count: {X.shape[0]}")
    print(f"Target count: {len(y)}")

    model_path = ROOT / "backend" / "orbitshield_classifier.joblib"

    if not model_path.exists():
        print(f"\nERROR: Trained model file not found: {model_path}")
        return

    try:
        import joblib
        saved = joblib.load(model_path)
        model_features = list(saved["feature_names"])
    except Exception as exc:
        print(f"\nERROR loading saved model: {exc}")
        return

    dataset_features = list(X.columns)

    missing = [f for f in model_features if f not in dataset_features]
    extra = [f for f in dataset_features if f not in model_features]

    print(f"\nSaved model feature count: {len(model_features)}")
    print(f"Features missing from dataset: {len(missing)}")
    print(f"Extra dataset features: {len(extra)}")

    if missing:
        print("\nMISSING FEATURES:")
        for name in missing:
            print(f"  - {name}")

    if extra:
        print("\nEXTRA DATASET FEATURES:")
        for name in extra:
            print(f"  - {name}")

    print(f"\nFeature names and order match: {model_features == dataset_features}")

    print("\nFEATURE-BY-FEATURE MISSING DATA:")
    for name in model_features:
        if name in X.columns:
            count = int(X[name].isna().sum())
            pct = count / max(len(X), 1) * 100
            print(f"{name:32} missing={count:7} ({pct:5.1f}%)")

    print("\nAUDIT COMPLETE")
    print("This checks the historical dataset/model schema only.")
    print("It does NOT prove that live TLE data can supply these features.")
    print("=" * 65)


if __name__ == "__main__":
    main()