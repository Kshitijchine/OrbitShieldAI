
from pathlib import Path
import joblib

BACKEND_DIR = Path(__file__).resolve().parent
MODEL_PATH = BACKEND_DIR / "orbitshield_classifier.joblib"


def main():
    if not MODEL_PATH.exists():
        print("ERROR: Model file not found:", MODEL_PATH)
        return

    bundle = joblib.load(MODEL_PATH)
    feature_names = bundle.get("feature_names", [])
    model = bundle.get("model")

    print("=" * 70)
    print("ORBITSHIELDAI - MODEL FEATURE INSPECTION")
    print("=" * 70)
    print("Model type:", bundle.get("model_type", type(model).__name__))
    print("Feature count:", len(feature_names))
    print("Training target:", bundle.get("target", "Not specified"))

    print("\nFEATURES IN THE EXACT TRAINING ORDER")
    print("-" * 70)

    for index, name in enumerate(feature_names, start=1):
        print(f"{index:02d}. {name}")

    print("\nMODEL PIPELINE STEPS")
    print("-" * 70)

    if hasattr(model, "named_steps"):
        for name, step in model.named_steps.items():
            print(f"{name}: {type(step).__name__}")
    else:
        print(type(model).__name__)

    print("\nInspection only. No model or application files were changed.")


if __name__ == "__main__":
    main()
