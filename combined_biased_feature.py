"""
Feature engineering experiment: combine the three biased-reporting
symptoms (headache, sore_throat, shortness_of_breath) into a single
"any_biased_symptom" flag (1 if the person reported any one of the
three, else 0), instead of either including them individually (as in
replicate_paper.py) or dropping them entirely (as in balanced_features.py).

Rationale: the paper found these three symptoms are reported almost
exclusively for people already suspected positive (headache 96.2%,
sore_throat 92.3%, shortness_of_breath 92.4% "biased reporting" rates),
making each one individually a noisy, overconfident signal (see the
head_ache SHAP distortion in replicate_paper.py's output). Collapsing
them into a single OR'd flag keeps *some* signal from these symptoms
while diluting the overconfidence any single rare biased feature can
have on the model.

Resulting feature set (6 total): sex, age>=60, cough, fever,
contact with confirmed case, any_biased_symptom.

Compare against:
- Full 8-feature baseline (replicate_paper.py): auROC 0.889
- Balanced 5-feature (drop biased symptoms entirely, balanced_features.py): auROC 0.865

Everything else (date split, hyperparameters, early stopping) is
identical to replicate_paper.py.
"""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

RAW_FEATURES = ["cough", "fever", "sore_throat", "shortness_of_breath", "head_ache"]

df = pd.read_csv("corona_tested_individuals_ver_006.english.csv", low_memory=False)
df["test_date"] = pd.to_datetime(df["test_date"])

# Binary task only: drop inconclusive "other" results
df = df[df["corona_result"].isin(["positive", "negative"])].copy()
df["target"] = (df["corona_result"] == "positive").astype(int)

# Encode the 8 features exactly as in the paper
df["sex_male"] = (df["gender"] == "male").astype("float")
df.loc[df["gender"].isna(), "sex_male"] = np.nan

df["age_60_plus"] = df["age_60_and_above"].map({"Yes": 1.0, "No": 0.0})

df["contact_confirmed"] = (df["test_indication"] == "Contact with confirmed").astype(float)

# Combine the three biased-reporting symptoms into a single OR'd flag.
# Missingness in these columns is ~1 row total across the whole dataset,
# so filling NaN with 0 before the OR is negligible.
df["any_biased_symptom"] = (
    (df["head_ache"].fillna(0) == 1)
    | (df["sore_throat"].fillna(0) == 1)
    | (df["shortness_of_breath"].fillna(0) == 1)
).astype(float)

FEATURES = ["sex_male", "age_60_plus", "cough", "fever", "contact_confirmed", "any_biased_symptom"]

train_val = df[(df.test_date >= "2020-03-22") & (df.test_date <= "2020-03-31")].copy()
test = df[(df.test_date >= "2020-04-01") & (df.test_date <= "2020-04-07")].copy()

print(f"Train+val set: {len(train_val):,} rows, {train_val.target.sum():,} positive "
      f"({train_val.target.mean()*100:.2f}%)")
print(f"Test set:      {len(test):,} rows, {test.target.sum():,} positive "
      f"({test.target.mean()*100:.2f}%)")
print("(Paper reports 51,831 train/val with 4,769 positive; "
      "47,401 test with 3,624 positive — small diffs expected from dataset version drift)")

# 4:1 train:validation split, as in the paper
train_val = train_val.sample(frac=1.0, random_state=42).reset_index(drop=True)
n_train = int(len(train_val) * 0.8)
train = train_val.iloc[:n_train]
val = train_val.iloc[n_train:]

X_train, y_train = train[FEATURES], train["target"]
X_val, y_val = val[FEATURES], val["target"]
X_test, y_test = test[FEATURES], test["target"]

# Hyperparameters from https://github.com/nshomron/covidpred (hyperparameters.txt)
params = {
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 20,
    "min_data_in_leaf": 4,
    "feature_fraction": 0.2,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "learning_rate": 0.05,
    "verbose": -1,
}

train_set = lgb.Dataset(X_train, label=y_train)
val_set = lgb.Dataset(X_val, label=y_val, reference=train_set)

model = lgb.train(
    params,
    train_set,
    num_boost_round=603,
    valid_sets=[val_set],
    callbacks=[lgb.early_stopping(stopping_rounds=5), lgb.log_evaluation(0)],
)

model.save_model("lgbm_combined_biased_feature_model.txt")

# ---- Evaluation on the held-out prospective test week ----
test_preds = model.predict(X_test, num_iteration=model.best_iteration)

auroc = roc_auc_score(y_test, test_preds)
auprc = average_precision_score(y_test, test_preds)

print(f"\nBest iteration: {model.best_iteration}")
print(f"Test auROC: {auroc:.4f}  (vs. full 8-feature baseline: 0.889, balanced 5-feature: 0.865)")
print(f"Test auPRC: {auprc:.4f}  (vs. full 8-feature baseline: 0.585)")

# A couple of working points, like the paper reports
fpr, tpr, thr = roc_curve(y_test, test_preds)
specificity = 1 - fpr
for target_sens in [0.8730, 0.8576]:
    idx = np.argmin(np.abs(tpr - target_sens))
    print(f"At sensitivity {tpr[idx]*100:.2f}%, specificity = {specificity[idx]*100:.2f}%  "
          f"(threshold={thr[idx]:.3f})")

results = {
    "n_train": len(train), "n_val": len(val), "n_test": len(test),
    "test_auroc": auroc, "test_auprc": auprc,
    "best_iteration": int(model.best_iteration),
}
with open("combined_biased_feature_results.json", "w") as f:
    json.dump(results, f, indent=2)

# ---- Plots: ROC curve + SHAP summary, mirroring the paper's Fig 3 ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr, color="#5B4FA3", label=f"auROC = {auroc:.3f}")
plt.plot([0, 1], [0, 1], color="lightgray", linestyle="--")
plt.xlabel("1 - specificity")
plt.ylabel("Sensitivity")
plt.title("Receiver Operating Characteristic (combined biased feature)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("eda_output/12_combined_biased_feature_roc.png", dpi=120)
plt.close()

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

plt.figure(figsize=(7, 5))
shap.summary_plot(shap_values, X_test, show=False)
plt.title("SHAP summary — combined biased feature model")
plt.tight_layout()
plt.savefig("eda_output/13_combined_biased_feature_shap.png", dpi=120)
plt.close()

print("\nPlots saved: eda_output/12_combined_biased_feature_roc.png, eda_output/13_combined_biased_feature_shap.png")
