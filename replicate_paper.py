"""
Replicate Zoabi et al. 2021 (npj Digital Medicine) "Machine learning-based
prediction of COVID-19 diagnosis based on symptoms".

Setup follows the paper's Methods section exactly:
- Train/validation: 2020-03-22 to 2020-03-31, split 4:1 train:val
- Test (prospective, held out): 2020-04-01 to 2020-04-07
- 8 binary features: sex, age>=60, cough, fever, sore_throat,
  shortness_of_breath, headache, contact with confirmed case
- LightGBM gradient-boosted trees, early stopping on validation auROC
- Hyperparameters taken from the paper's repo (hyperparameters.txt)
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

FEATURES = ["sex_male", "age_60_plus", "cough", "fever", "sore_throat",
            "shortness_of_breath", "head_ache", "contact_confirmed"]

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

model.save_model("lgbm_replication_model.txt")

# ---- Evaluation on the held-out prospective test week ----
test_preds = model.predict(X_test, num_iteration=model.best_iteration)

auroc = roc_auc_score(y_test, test_preds)
auprc = average_precision_score(y_test, test_preds)

print(f"\nBest iteration: {model.best_iteration}")
print(f"Test auROC: {auroc:.4f}  (paper: 0.90, 95% CI 0.892-0.905)")
print(f"Test auPRC: {auprc:.4f}  (paper: 0.66, 95% CI 0.647-0.678)")

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
with open("replication_results.json", "w") as f:
    json.dump(results, f, indent=2)

# ---- Plots: ROC curve + SHAP summary, mirroring the paper's Fig 1a / Fig 2 ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr, color="#5B4FA3", label=f"auROC = {auroc:.3f}")
plt.plot([0, 1], [0, 1], color="lightgray", linestyle="--")
plt.xlabel("1 - specificity")
plt.ylabel("Sensitivity")
plt.title("Receiver Operating Characteristic (replication)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("eda_output/08_replication_roc.png", dpi=120)
plt.close()

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

plt.figure(figsize=(7, 5))
shap.summary_plot(shap_values, X_test, show=False)
plt.title("SHAP summary — replication model")
plt.tight_layout()
plt.savefig("eda_output/09_replication_shap.png", dpi=120)
plt.close()

print("\nPlots saved: eda_output/08_replication_roc.png, eda_output/09_replication_shap.png")
