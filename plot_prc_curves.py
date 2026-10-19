"""
Generate PRC (precision-recall) curve plots for each trained model,
to sit alongside the ROC curve plots already in the _FINDINGS.md
files. Loads the already-saved LightGBM models directly - no
retraining needed.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

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
base_rate = y_test.mean()

FULL_FEATURES = ["sex_male", "age_60_plus", "cough", "fever", "sore_throat",
                  "shortness_of_breath", "head_ache", "contact_confirmed"]
BALANCED_FEATURES = ["sex_male", "age_60_plus", "cough", "fever", "contact_confirmed"]
COMBINED_FEATURES = ["sex_male", "age_60_plus", "cough", "fever", "contact_confirmed", "any_biased_symptom"]

CONFIGS = [
    ("Full 8-feature", "lgbm_replication_model.txt", FULL_FEATURES, "17_replication_prc.png"),
    ("Balanced 5-feature", "lgbm_balanced_features_model.txt", BALANCED_FEATURES, "18_balanced_features_prc.png"),
    ("Combined 6-feature", "lgbm_combined_biased_feature_model.txt", COMBINED_FEATURES, "19_combined_biased_feature_prc.png"),
    ("Tuned 8-feature", "lgbm_tuned_model.txt", FULL_FEATURES, "20_tuned_prc.png"),
]

results = {}
curves = {}
for name, model_file, features, out_file in CONFIGS:
    model = lgb.Booster(model_file=model_file)
    preds = model.predict(test[features])
    precision, recall, _ = precision_recall_curve(y_test, preds)
    auprc = average_precision_score(y_test, preds)
    curves[name] = (recall, precision, auprc)

    plt.figure(figsize=(5, 5))
    plt.plot(recall, precision, color="#C44E52", label=f"auPRC = {auprc:.3f}")
    plt.axhline(base_rate, color="lightgray", linestyle="--", label=f"no-skill = {base_rate:.3f}")
    plt.xlabel("Recall (sensitivity)")
    plt.ylabel("Precision (PPV)")
    plt.title(f"Precision-Recall Curve ({name})")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(f"eda_output/{out_file}", dpi=120)
    plt.close()
    print(f"{name}: auPRC = {auprc:.4f} -> eda_output/{out_file}")

# combined overlay plot comparing all four
plt.figure(figsize=(6, 6))
colors = {"Full 8-feature": "#4C72B0", "Balanced 5-feature": "#DD8452",
          "Combined 6-feature": "#55A868", "Tuned 8-feature": "#C44E52"}
for name, (recall, precision, auprc) in curves.items():
    plt.plot(recall, precision, color=colors[name], label=f"{name} (auPRC={auprc:.3f})")
plt.axhline(base_rate, color="lightgray", linestyle="--", label=f"no-skill = {base_rate:.3f}")
plt.xlabel("Recall (sensitivity)")
plt.ylabel("Precision (PPV)")
plt.title("Precision-Recall Curves — all models compared")
plt.legend(loc="upper right", fontsize=8)
plt.tight_layout()
plt.savefig("eda_output/21_all_models_prc_comparison.png", dpi=120)
plt.close()
print("Comparison plot -> eda_output/21_all_models_prc_comparison.png")
