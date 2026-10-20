"""
Hyperparameter tuning for the "balanced features" model — same idea as
tune_hyperparameters.py (Optuna + 5-fold CV, prospective test week held
out strictly throughout the search), but on balanced_features.py's
5-feature set that drops the three biased-reporting symptoms
(head_ache, sore_throat, shortness_of_breath), keeping only:
sex, age>=60, cough, fever, contact with confirmed case.

Baseline to beat (balanced_features.py, paper's own hyperparameters):
  auROC 0.8654, auPRC 0.4562

Goal: see whether a tuned version of the bias-aware 5-feature model can
close the gap with the tuned 8-feature model (auROC 0.902, auPRC 0.700)
while staying free of the head_ache/sore_throat/shortness_of_breath
over-reliance flagged in TUNING_FINDINGS.md.
"""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

optuna.logging.set_verbosity(optuna.logging.WARNING)

df = pd.read_csv("corona_tested_individuals_ver_006.english.csv", low_memory=False)
df["test_date"] = pd.to_datetime(df["test_date"])

df = df[df["corona_result"].isin(["positive", "negative"])].copy()
df["target"] = (df["corona_result"] == "positive").astype(int)

df["sex_male"] = (df["gender"] == "male").astype("float")
df.loc[df["gender"].isna(), "sex_male"] = np.nan
df["age_60_plus"] = df["age_60_and_above"].map({"Yes": 1.0, "No": 0.0})
df["contact_confirmed"] = (df["test_indication"] == "Contact with confirmed").astype(float)

# Balanced-features set: drop head_ache, sore_throat, shortness_of_breath
FEATURES = ["sex_male", "age_60_plus", "cough", "fever", "contact_confirmed"]

train_val = df[(df.test_date >= "2020-03-22") & (df.test_date <= "2020-03-31")].copy()
test = df[(df.test_date >= "2020-04-01") & (df.test_date <= "2020-04-07")].copy()

X_tv, y_tv = train_val[FEATURES].reset_index(drop=True), train_val["target"].reset_index(drop=True)
X_test, y_test = test[FEATURES], test["target"]

print(f"Train/val (CV) set: {len(train_val):,} rows, {y_tv.sum():,} positive")
print(f"Held-out test set:   {len(test):,} rows, {y_test.sum():,} positive "
      "(never used during tuning)")

N_FOLDS = 5
cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

BASE_PARAMS = {"objective": "binary", "metric": "auc", "verbose": -1}


def objective(trial):
    params = {
        **BASE_PARAMS,
        "num_leaves": trial.suggest_int("num_leaves", 4, 64),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 2, 200),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.2, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
        "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
    }

    fold_aucs = []
    for fold_train_idx, fold_val_idx in cv.split(X_tv, y_tv):
        fold_train = lgb.Dataset(X_tv.iloc[fold_train_idx], label=y_tv.iloc[fold_train_idx])
        fold_val = lgb.Dataset(X_tv.iloc[fold_val_idx], label=y_tv.iloc[fold_val_idx], reference=fold_train)

        booster = lgb.train(
            params,
            fold_train,
            num_boost_round=1000,
            valid_sets=[fold_val],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )
        fold_aucs.append(booster.best_score["valid_0"]["auc"])

    return float(np.mean(fold_aucs))


N_TRIALS = 60
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

print(f"\nBest {N_FOLDS}-fold CV mean auROC: {study.best_value:.4f}")
print("Best params:")
for k, v in study.best_params.items():
    print(f"  {k}: {v}")

# ---- Refit on the full train/val set with the best params, using one more
# internal split only to pick num_boost_round via early stopping, then
# retrain on ALL of train/val at that fixed round count before touching test ----
best_params = {**BASE_PARAMS, **study.best_params}

sub_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
fold_train_idx, fold_val_idx = next(sub_cv.split(X_tv, y_tv))
es_train = lgb.Dataset(X_tv.iloc[fold_train_idx], label=y_tv.iloc[fold_train_idx])
es_val = lgb.Dataset(X_tv.iloc[fold_val_idx], label=y_tv.iloc[fold_val_idx], reference=es_train)
es_booster = lgb.train(
    best_params, es_train, num_boost_round=1000, valid_sets=[es_val],
    callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
)
best_num_rounds = es_booster.best_iteration
print(f"\nChosen num_boost_round (via early stopping on a held-out split): {best_num_rounds}")

final_train = lgb.Dataset(X_tv, label=y_tv)
final_model = lgb.train(best_params, final_train, num_boost_round=best_num_rounds)
final_model.save_model("lgbm_tuned_balanced_features_model.txt")

# ---- Evaluate ONCE on the held-out prospective test week ----
test_preds = final_model.predict(X_test)
auroc = roc_auc_score(y_test, test_preds)
auprc = average_precision_score(y_test, test_preds)

print(f"\nTest auROC: {auroc:.4f}  (baseline with paper's hyperparameters: 0.8654)")
print(f"Test auPRC: {auprc:.4f}  (baseline with paper's hyperparameters: 0.4562)")

fpr, tpr, thr = roc_curve(y_test, test_preds)
specificity = 1 - fpr
for target_sens in [0.8730, 0.8576]:
    idx = np.argmin(np.abs(tpr - target_sens))
    print(f"At sensitivity {tpr[idx]*100:.2f}%, specificity = {specificity[idx]*100:.2f}%  "
          f"(threshold={thr[idx]:.3f})")

results = {
    "n_trials": N_TRIALS,
    "n_folds": N_FOLDS,
    "features": FEATURES,
    "best_cv_auroc": study.best_value,
    "best_params": study.best_params,
    "best_num_boost_round": int(best_num_rounds),
    "test_auroc": auroc,
    "test_auprc": auprc,
    "baseline_test_auroc": 0.8654,
    "baseline_test_auprc": 0.4562,
}
with open("tune_balanced_features_results.json", "w") as f:
    json.dump(results, f, indent=2)

# ---- Plots ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr, color="#5B4FA3", label=f"auROC = {auroc:.3f}")
plt.plot([0, 1], [0, 1], color="lightgray", linestyle="--")
plt.xlabel("1 - specificity")
plt.ylabel("Sensitivity")
plt.title("ROC — tuned balanced-features model")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("eda_output/22_tuned_balanced_features_roc.png", dpi=120)
plt.close()

precision, recall, _ = precision_recall_curve(y_test, test_preds)
plt.figure(figsize=(5, 5))
plt.plot(recall, precision, color="#5B4FA3", label=f"auPRC = {auprc:.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall — tuned balanced-features model")
plt.legend(loc="lower left")
plt.tight_layout()
plt.savefig("eda_output/23_tuned_balanced_features_prc.png", dpi=120)
plt.close()

explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_test)
plt.figure(figsize=(7, 5))
shap.summary_plot(shap_values, X_test, show=False)
plt.title("SHAP summary — tuned balanced-features model")
plt.tight_layout()
plt.savefig("eda_output/24_tuned_balanced_features_shap.png", dpi=120)
plt.close()

fig = optuna.visualization.matplotlib.plot_optimization_history(study)
plt.tight_layout()
plt.savefig("eda_output/25_tuned_balanced_features_tuning_history.png", dpi=120)
plt.close()

print("\nPlots saved: eda_output/22_tuned_balanced_features_roc.png, "
      "eda_output/23_tuned_balanced_features_prc.png, "
      "eda_output/24_tuned_balanced_features_shap.png, "
      "eda_output/25_tuned_balanced_features_tuning_history.png")
