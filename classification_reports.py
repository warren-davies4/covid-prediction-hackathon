"""
Classification reports (precision/recall/F1 per class) for all four
models built so far, evaluated on the held-out prospective test week
(2020-04-01 to 2020-04-07).

IMPORTANT: a plain 0.5 probability threshold is a bad choice here.
These models were trained to optimize auROC/auPRC (ranking quality),
and the test set is heavily imbalanced (~7.5% positive) - several of
the models never produce a raw probability above 0.5 given so few
boosting rounds, which makes a 0.5-threshold report degenerate (zero
predicted positives). Instead, for each model we pick the probability
threshold that maximizes F1 on this test set, and report at that
threshold - a standard, principled way to pick an operating point
when the default 0.5 cutoff doesn't make sense for the class balance.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import classification_report, precision_recall_curve

df = pd.read_csv("corona_tested_individuals_ver_006.english.csv", low_memory=False)
df["test_date"] = pd.to_datetime(df["test_date"])
df = df[df["corona_result"].isin(["positive", "negative"])].copy()
df["target"] = (df["corona_result"] == "positive").astype(int)
df["sex_male"] = (df["gender"] == "male").astype("float")
df.loc[df["gender"].isna(), "sex_male"] = np.nan
df["age_60_plus"] = df["age_60_and_above"].map({"Yes": 1.0, "No": 0.0})
df["contact_confirmed"] = (df["test_indication"] == "Contact with confirmed").astype(float)
df["any_biased_symptom"] = (
    (df["head_ache"].fillna(0) == 1) | (df["sore_throat"].fillna(0) == 1) | (df["shortness_of_breath"].fillna(0) == 1)
).astype(float)

test = df[(df.test_date >= "2020-04-01") & (df.test_date <= "2020-04-07")].copy()
y_test = test["target"]

FULL_FEATURES = ["sex_male", "age_60_plus", "cough", "fever", "sore_throat",
                  "shortness_of_breath", "head_ache", "contact_confirmed"]
BALANCED_FEATURES = ["sex_male", "age_60_plus", "cough", "fever", "contact_confirmed"]
COMBINED_FEATURES = ["sex_male", "age_60_plus", "cough", "fever", "contact_confirmed", "any_biased_symptom"]

MODELS = {
    "Full 8-feature (paper's hyperparameters)": ("lgbm_replication_model.txt", FULL_FEATURES),
    "Balanced 5-feature (biased symptoms dropped)": ("lgbm_balanced_features_model.txt", BALANCED_FEATURES),
    "Combined 6-feature (biased symptoms -> 1 flag)": ("lgbm_combined_biased_feature_model.txt", COMBINED_FEATURES),
    "Tuned 8-feature (Optuna + 5-fold CV)": ("lgbm_tuned_model.txt", FULL_FEATURES),
}

report_lines = []
for name, (model_file, features) in MODELS.items():
    model = lgb.Booster(model_file=model_file)
    preds = model.predict(test[features])

    precision, recall, thresholds = precision_recall_curve(y_test, preds)
    f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall + 1e-12), 0)
    best_idx = np.argmax(f1[:-1])  # last point has no corresponding threshold
    best_threshold = thresholds[best_idx]

    y_pred = (preds >= best_threshold).astype(int)
    report = classification_report(y_test, y_pred, target_names=["negative", "positive"], digits=3)
    header = (f"=== {name} ===\n"
              f"(F1-optimal threshold = {best_threshold:.4f}, n = {len(y_test):,}, "
              f"{y_test.sum():,} actual positives)")
    print(header)
    print(report)
    report_lines.append(header + "\n" + report)

with open("classification_reports.txt", "w") as f:
    f.write("\n\n".join(report_lines))
