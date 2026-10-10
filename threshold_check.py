
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
)

test_path = r"data\kelvins_competition_data\test_data.csv"
label_path = r"data\kelvins_competition_data\test_data_private.csv"

data = pd.read_csv(test_path)
labels = pd.read_csv(label_path).set_index("event_id")
saved = joblib.load(r"backend\orbitshield_classifier.joblib")

rows = []
for event_id, group in data.groupby("event_id", sort=False):
    group = group.sort_values("time_to_tca", ascending=False)

    if len(group) < 2:
        continue

    candidates = group[group["time_to_tca"] >= 2]
    if candidates.empty:
        continue

    row = candidates.iloc[0].drop(
        labels=[
            col for col in (
                "event_id", "risk",
                "max_risk_estimate", "max_risk_scaling"
            )
            if col in group.columns
        ]
    )
    row.name = event_id
    rows.append(row)

X = pd.DataFrame(rows).reindex(columns=saved["feature_names"])
X = X.replace([float("inf"), float("-inf")], float("nan"))
y = (labels.loc[X.index, "true_risk"] > -30).astype(int)

probabilities = saved["model"].predict_proba(X)[:, 1]

print("Threshold | Accuracy | Balanced accuracy | Class 1 precision | Class 1 recall | False positives")

for threshold in [0.30, 0.40, 0.50, 0.60, 0.70]:
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(
        y, predictions, labels=[0, 1]
    ).ravel()

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    print(
        f"{threshold:.2f} | "
        f"{accuracy_score(y, predictions):.3f} | "
        f"{balanced_accuracy_score(y, predictions):.3f} | "
        f"{precision:.3f} | {recall:.3f} | {fp}"
    )

print("ROC-AUC:", round(roc_auc_score(y, probabilities), 4))
print("Evaluated events:", len(y))
