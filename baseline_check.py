import pandas as pd
import joblib
from sklearn.metrics import brier_score_loss

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

baseline_probability = y.mean()
baseline_predictions = [baseline_probability] * len(y)

model_score = brier_score_loss(y, probabilities)
baseline_score = brier_score_loss(y, baseline_predictions)

print("Evaluated events:", len(y))
print("Actual Class 1 rate:", round(baseline_probability, 4))
print("Model Brier score:", round(model_score, 4))
print("Baseline Brier score:", round(baseline_score, 4))

if model_score < baseline_score:
    print("Result: Model scores improve on the constant-rate baseline.")
else:
    print("Result: Model scores do not improve on the constant-rate baseline.")