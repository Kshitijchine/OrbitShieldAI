
import sys
from pathlib import Path

import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

from dataset_loader import prepare_event_dataset


def main():
    print("Loading event-level dataset...")
    X, y = prepare_event_dataset()

    X = X.select_dtypes(include=[np.number]).copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")

    # Class 1 means final recorded risk is greater than -30.
    target = (y > -30).astype(int)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42,
    )

    train_idx, test_idx = next(
        splitter.split(X, target, groups=X.index)
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = target.iloc[train_idx]
    y_test = target.iloc[test_idx]

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
        )),
    ])

    print("\nTraining standardized Logistic Regression...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    scores = model.predict_proba(X_test)[:, 1]

    print("\n--- EVALUATION ---")
    print("Training events:", len(X_train))
    print("Testing events:", len(X_test))
    print(
        "Majority-class baseline accuracy:",
        round(max(y_test.mean(), 1 - y_test.mean()), 4),
    )
    print(
        "Balanced accuracy:",
        round(balanced_accuracy_score(y_test, predictions), 4),
    )
    print("ROC-AUC:", round(roc_auc_score(y_test, scores), 4))
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))

    output_path = BACKEND_DIR / "orbitshield_classifier.joblib"

    joblib.dump(
        {
            "model": model,
            "feature_names": X.columns.tolist(),
            "target": "final_recorded_risk_greater_than_minus_30",
            "model_type": "standardized balanced LogisticRegression",
        },
        output_path,
    )

    print("\nModel saved to:", output_path)


if __name__ == "__main__":
    main()
