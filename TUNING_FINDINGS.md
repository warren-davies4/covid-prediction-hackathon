# Hyperparameter tuning (Optuna + 5-fold CV)

Script: `tune_hyperparameters.py`. Model: `lgbm_tuned_model.txt`. Results: `tune_hyperparameters_results.json`.

## Method

- Same 8 features, same date split as `replicate_paper.py` — only the hyperparameters change, so any improvement is isolated to tuning rather than feature changes.
- Search: **Optuna** (TPE sampler), 60 trials, maximizing mean auROC across **5-fold stratified cross-validation** on the train/validation period (2020-03-22 to 2020-03-31). Each fold trains with LightGBM's own early stopping (30 rounds patience, up to 1000 rounds).
- The prospective test week (2020-04-01 to 2020-04-07) was **never touched during the search** — same strict holdout as the paper. It was used exactly once, at the end, to score the final tuned model.
- Final model: refit on the full train/val set with the best params found, using a boosting-round count chosen via one more early-stopping pass on a held-out split (so the test week still isn't used to pick anything).

## Result: beats the paper's hyperparameters on both metrics

| Metric | Paper's hyperparameters (baseline) | **Tuned** |
|---|---|---|
| **auROC** | 0.889 | **0.902** |
| **auPRC** | 0.585 | **0.700** |
| Boosting rounds used | 10 (of max 603) | 24 (of max 1000) |

![Tuned ROC curve](eda_output/14_tuned_roc.png)

5-fold CV mean auROC during search: **0.9356** (expected to be somewhat higher than the test score, since CV folds share the same ~10-day period and distribution as each other, while the test week is a genuinely different, later time period).

Best hyperparameters found:

| Param | Paper's value | Tuned value |
|---|---|---|
| `num_leaves` | 20 | 7 |
| `min_data_in_leaf` | 4 | 59 |
| `feature_fraction` | 0.2 | 0.72 |
| `bagging_fraction` | 0.8 | 0.97 |
| `bagging_freq` | 5 | 4 |
| `learning_rate` | 0.05 | 0.14 |
| `lambda_l1` | (not set) | 0.24 |
| `lambda_l2` | (not set) | 0.09 |

The tuned model prefers **much smaller, more regularized trees** (`num_leaves` 7 vs 20, `min_data_in_leaf` 59 vs 4) trained **faster** (higher learning rate, far fewer boosting rounds — 24 vs the paper's implied ~hundreds) with **more features visible per tree** (`feature_fraction` 0.72 vs 0.2 — recall there are only 8 features total, so 0.2 means each tree only sees ~1-2 of them, which looks like under-fitting). This combination of fewer, shallower trees but more features per split and a touch of L1/L2 regularization generalizes noticeably better to the held-out week than the paper's original configuration.

![Optuna optimization history](eda_output/16_tuning_history.png)

The search converges quickly (within ~10 trials) and stays flat — suggesting we're close to this feature set's ceiling for a single LightGBM model, rather than needing a much larger search budget.

## Caveat: the bias-exploitation effect gets *stronger*, not weaker

![Tuned SHAP summary](eda_output/15_tuned_shap.png)

This is the important catch. In `replicate_paper.py`'s SHAP plot, `head_ache` alone already looked suspiciously dominant (SHAP values up to ~1.4) — flagged there as a likely artifact of the paper's own documented reporting bias (headache was recorded almost exclusively for people already suspected positive). **In the tuned model, this effect is substantially larger**: `head_ache`, `sore_throat` and `shortness_of_breath` now show SHAP (margin) values up to **~4–5**, while `cough`/`fever`/`contact_confirmed`/`sex_male`/`age_60_plus` stay in the roughly ±0.5 range they were in before.

In other words: part of why the tuned model scores higher is that the search found a hyperparameter configuration that lets the model lean on the three most biased-but-separating symptoms *even harder* than the paper's own settings did. This isn't a bug in the tuning — it's a legitimate improvement on the stated metrics (auROC/auPRC on this specific test week) — but it means the tuned model is arguably **less** trustworthy as a genuine clinical signal than the paper's original, even though it scores better.

## Takeaway

- **On raw metrics, tuning clearly helps**: auROC 0.889 → 0.902, auPRC 0.585 → 0.700, using the exact same data and features as the paper.
- **This reframes the combined-biased-feature experiment** (`FEATURE_ENGINEERING_FINDINGS.md`): if the untuned full model already leaned on `head_ache` due to bias, and tuning makes that worse, the combined `any_biased_symptom` feature becomes more attractive, not less — it caps how much the model can over-trust any single biased symptom. **Worth re-running the Optuna search on the 6-feature combined-feature set** to see whether a tuned version of that model can close the gap with the tuned 8-feature model, while staying more trustworthy. That's a natural next step.
- If the hackathon's goal rewards leaderboard metrics only, the tuned 8-feature model is the strongest result so far. If it also rewards a defensible, bias-aware model (closer to the paper's own stated intent), the combined-feature approach — ideally also hyperparameter-tuned — is the safer one to present.
