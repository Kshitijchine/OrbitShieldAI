from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit

PROJECT_DIR = Path(__file__).resolve().parent.parent
BACKEND_DIR = PROJECT_DIR / "backend"
MODEL_PATH = BACKEND_DIR / "orbitshield_classifier.joblib"

sys.path.insert(0, str(BACKEND_DIR))
from dataset_loader import prepare_event_dataset


def main():
    print("=" * 68)
    print("ORBITSHIELDAI - SAVED MODEL EVALUATION")
    print("=" * 68)

    if not MODEL_PATH.exists():
        print("ERROR: Saved model not found:", MODEL_PATH)
        return

    print("\nPreparing data using the original dataset loader...")
    X, original_risk = prepare_event_dataset()

    print("\nLoading saved model...")
    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_names = bundle["feature_names"]

    # Match the feature cleanup used by the training script.
    X = X.select_dtypes(include=[np.number]).copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")

    # Match the exact binary target definition used in training.
    y = (original_risk > -30).astype(int)

    # Keep the saved model's exact feature order.
    missing_features = [name for name in feature_names if name not in X.columns]
    if missing_features:
        print("ERROR: Required model features are missing:", missing_features)
        return

    X = X.reindex(columns=feature_names)

    # Match the deterministic split used by train_model.py.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42,
    )
    train_indices, test_indices = next(
        splitter.split(X, y, groups=X.index)
    )

    X_test = X.iloc[test_indices]
    y_test = y.iloc[test_indices]

    # This verifies the saved model on the same test partition used
    # by the training script, rather than an independent test set.
    predictions = model.predict(X_test)

    print("\nEvaluation results")
    print("-" * 68)
    print("Eligible events:", len(X))
    print("Training partition:", len(train_indices))
    print("Evaluation partition:", len(X_test))
    print("Feature count:", len(feature_names))
    print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
    print(
        "Balanced accuracy:",
        round(balanced_accuracy_score(y_test, predictions), 4),
    )

    print("\nClassification report")
    print(classification_report(y_test, predictions, zero_division=0))

    print("Confusion matrix (rows=actual, columns=predicted)")
    print(confusion_matrix(y_test, predictions))

    if hasattr(model, "predict_proba") and y_test.nunique() == 2:
        probabilities = model.predict_proba(X_test)[:, 1]
        print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))
    else:
        print("ROC-AUC unavailable: probabilities or both classes are missing.")

    print("\nIMPORTANT")
    print(
        "These metrics measure prediction of the historical dataset label, "
        "not collision probability or operational maneuver safety."
    )
    print(
        "This is a consistency check on the training script's held-out "
        "partition, not independent validation on unseen data."
    )


if __name__ == "__main__":
    main()