"""
Hyperparameter tuning for the full 8-feature model, using Optuna +
k-fold cross-validation on the train/validation period, with the
paper's prospective test week (Apr 1-7) held out strictly throughout
the search — it is only touched once, at the very end, to score the
final tuned model.

Baseline to beat (replicate_paper.py, paper's own hyperparameters):
  auROC 0.889, auPRC 0.585

This keeps the same features, date split and model family (LightGBM)
as replicate_paper.py — isolating hyperparameter choice as the only
variable.
"""
import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

optuna.logging.set_verbosity(optuna.logging.WARNING)

df = pd.read_csv("corona_tested_individuals_ver_006.english.csv", low_memory=False)
df["test_date"] = pd.to_datetime(df["test_date"])

df = df[df["corona_result"].isin(["positive", "negative"])].copy()
df["target"] = (df["corona_result"] == "positive").astype(int)

df["sex_male"] = (df["gender"] == "male").astype("float")
df.loc[df["gender"].isna(), "sex_male"] = np.nan
df["age_60_plus"] = df["age_60_and_above"].map({"Yes": 1.0, "No": 0.0})
df["contact_confirmed"] = (df["test_indication"] == "Contact with confirmed").astype(float)

FEATURES = ["sex_male", "age_60_plus", "cough", "fever", "sore_throat",
            "shortness_of_breath", "head_ache", "contact_confirmed"]

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
final_model.save_model("lgbm_tuned_model.txt")

# ---- Evaluate ONCE on the held-out prospective test week ----
test_preds = final_model.predict(X_test)
auroc = roc_auc_score(y_test, test_preds)
auprc = average_precision_score(y_test, test_preds)

print(f"\nTest auROC: {auroc:.4f}  (baseline with paper's hyperparameters: 0.889)")
print(f"Test auPRC: {auprc:.4f}  (baseline with paper's hyperparameters: 0.585)")

fpr, tpr, thr = roc_curve(y_test, test_preds)
specificity = 1 - fpr
for target_sens in [0.8730, 0.8576]:
    idx = np.argmin(np.abs(tpr - target_sens))
    print(f"At sensitivity {tpr[idx]*100:.2f}%, specificity = {specificity[idx]*100:.2f}%  "
          f"(threshold={thr[idx]:.3f})")

results = {
    "n_trials": N_TRIALS,
    "n_folds": N_FOLDS,
    "best_cv_auroc": study.best_value,
    "best_params": study.best_params,
    "best_num_boost_round": int(best_num_rounds),
    "test_auroc": auroc,
    "test_auprc": auprc,
    "baseline_test_auroc": 0.889,
    "baseline_test_auprc": 0.585,
}
with open("tune_hyperparameters_results.json", "w") as f:
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
plt.title("ROC — hyperparameter-tuned model")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("eda_output/14_tuned_roc.png", dpi=120)
plt.close()

explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_test)
plt.figure(figsize=(7, 5))
shap.summary_plot(shap_values, X_test, show=False)
plt.title("SHAP summary — hyperparameter-tuned model")
plt.tight_layout()
plt.savefig("eda_output/15_tuned_shap.png", dpi=120)
plt.close()

fig = optuna.visualization.matplotlib.plot_optimization_history(study)
plt.tight_layout()
plt.savefig("eda_output/16_tuning_history.png", dpi=120)
plt.close()

print("\nPlots saved: eda_output/14_tuned_roc.png, eda_output/15_tuned_shap.png, "
      "eda_output/16_tuning_history.png")
