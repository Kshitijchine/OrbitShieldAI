
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_DIR))

from dataset_loader import prepare_event_dataset


def main():
    print("Loading dataset...")
    X, y = prepare_event_dataset()

    X = X.select_dtypes(include=[np.number]).copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")
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

    print("\nTraining temporary model...")
    model.fit(X_train, y_train)
    scores = model.predict_proba(X_test)[:, 1]

    print("\n--- SCORE INSPECTION ---")
    print("ROC-AUC:", round(roc_auc_score(y_test, scores), 4))
    print("Score summary:")
    print(pd.Series(scores).describe().round(4).to_string())

    print("\nPredicted score ranges by actual class:")
    for class_value in [0, 1]:
        class_scores = scores[y_test.to_numpy() == class_value]
        print(f"\nClass {class_value}:")
        print(pd.Series(class_scores).describe().round(4).to_string())

    print("\nPercentiles of predicted Class 1 scores:")
    print(
        pd.Series(scores)
        .quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])
        .round(4)
        .to_string()
    )

    print("\nNo model file was saved or overwritten.")


if __name__ == "__main__":
    main()
