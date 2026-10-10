
import sys
from pathlib import Path

import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

from dataset_loader import prepare_event_dataset


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    scores = model.predict_proba(X_test)[:, 1]
    predictions = (scores >= 0.5).astype(int)

    print(f"\n{name}")
    print("-" * len(name))
    print("Balanced accuracy:", round(
        balanced_accuracy_score(y_test, predictions), 4
    ))
    print("ROC-AUC:", round(
        roc_auc_score(y_test, scores), 4
    ))


def main():
    print("Loading event-level dataset...")
    X, y = prepare_event_dataset()

    X = X.select_dtypes(include=[np.number]).copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")
    target = (y > -30).astype(int)

    # Use the same split as the previous experiments.
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42,
    )
    train_idx, test_idx = next(
        splitter.split(X, target, groups=X.index)
    )

    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()
    y_train = target.iloc[train_idx]
    y_test = target.iloc[test_idx]

    print("\nFeatures:", X.shape[1])
    print("Training events:", len(X_train))
    print("Testing events:", len(X_test))
    print("\n--- MODEL COMPARISON ---")

    baseline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
        )),
    ])

    standardized = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
        )),
    ])

    # Comparison excluding suspicious covariance determinant columns.
    covariance_columns = [
        col for col in X.columns
        if "position_covariance_det" in col
    ]
    X_train_no_cov = X_train.drop(
        columns=covariance_columns, errors="ignore"
    )
    X_test_no_cov = X_test.drop(
        columns=covariance_columns, errors="ignore"
    )

    standardized_no_cov = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
        )),
    ])

    evaluate_model(
        "1. Existing approach (no scaling)",
        baseline, X_train, X_test, y_train, y_test
    )
    evaluate_model(
        "2. Standardized features",
        standardized, X_train, X_test, y_train, y_test
    )
    evaluate_model(
        "3. Standardized, without covariance determinant columns",
        standardized_no_cov,
        X_train_no_cov, X_test_no_cov, y_train, y_test
    )

    print("\nNo model file was saved or overwritten.")


if __name__ == "__main__":
    main()
