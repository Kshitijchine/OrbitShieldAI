
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

from dataset_loader import prepare_event_dataset


def main():
    print("Loading event-level dataset...")
    X, y = prepare_event_dataset()

    X = X.select_dtypes(include=[np.number]).copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")

    # Same target definition as the existing training script.
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
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
        )),
    ])

    print("\nTraining a temporary evaluation model...")
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]

    print("\n--- THRESHOLD EVALUATION ---")
    print("This is a temporary model; no model file is saved.")
    print("Test events:", len(y_test))
    print("Class 0 = final recorded risk <= -30")
    print("Class 1 = final recorded risk > -30")
    print("\nThreshold | Balanced accuracy | Class 0 recall | Class 1 recall")

    for threshold in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        predictions = (probabilities >= threshold).astype(int)
        report = classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            output_dict=True,
            zero_division=0,
        )

        balanced = balanced_accuracy_score(y_test, predictions)
        recall_0 = report["0"]["recall"]
        recall_1 = report["1"]["recall"]

        print(
            f"{threshold:9.2f} | "
            f"{balanced:18.4f} | "
            f"{recall_0:14.4f} | "
            f"{recall_1:14.4f}"
        )

    default_predictions = (probabilities >= 0.50).astype(int)
    print("\n--- CONFUSION MATRIX AT THRESHOLD 0.50 ---")
    print("Rows = actual classes; columns = predicted classes")
    print(confusion_matrix(
        y_test,
        default_predictions,
        labels=[0, 1],
    ))

    print("\n--- CLASSIFICATION REPORT AT THRESHOLD 0.50 ---")
    print(classification_report(
        y_test,
        default_predictions,
        labels=[0, 1],
        zero_division=0,
    ))


if __name__ == "__main__":
    main()
