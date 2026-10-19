# Using these models (no retraining needed)

Four trained LightGBM models are checked into this repo, saved in LightGBM's native
text format (`model.save_model(...)` / `lgb.Booster(model_file=...)`) rather than
pickle — this format is self-contained and doesn't require matching the exact
library versions used to train it, and carries none of pickle's
arbitrary-code-execution risk if a file ever came from somewhere less trusted.

| File | Script that produced it | auROC | auPRC |
|---|---|---|---|
| `lgbm_replication_model.txt` | `replicate_paper.py` | 0.889 | 0.585 |
| `lgbm_balanced_features_model.txt` | `balanced_features.py` | 0.865 | 0.456 |
| `lgbm_combined_biased_feature_model.txt` | `combined_biased_feature.py` | 0.881 | 0.453 |
| `lgbm_tuned_model.txt` | `tune_hyperparameters.py` | **0.902** | **0.700** |

**Use `predict.py`** rather than loading these files directly — it also carries the
exact feature engineering (which raw CSV columns map to which model inputs, in what
order) and the F1-optimal decision threshold for each model, both of which the raw
model file doesn't know about:

```python
from predict import predict
import pandas as pd

raw_df = pd.read_csv("some_new_export.csv")  # same columns as the original
                                              # MoH export: cough, fever,
                                              # sore_throat, shortness_of_breath,
                                              # head_ache, age_60_and_above,
                                              # gender, test_indication
result = predict(raw_df, model_name="tuned")
# result now has covid_probability and covid_prediction columns
```

**Which model to pick:** `tuned` has the best raw metrics (see `TUNING_FINDINGS.md`)
and the best precision/recall/F1 (see `classification_reports.txt`), but it leans
more heavily on three symptoms (`head_ache`, `sore_throat`, `shortness_of_breath`)
that the paper identified as having biased reporting — i.e. part of its score comes
from exploiting a known data quality issue, not purely genuine signal. If an
explainable, bias-aware model matters more than squeezing out the last bit of
auROC/auPRC, use `combined` instead.

**If new data has a very different class balance** than the paper's test week
(~7.5% positive), the thresholds baked into `predict.py` may no longer be optimal —
rerun `classification_reports.py`'s F1-threshold search on the new data rather than
reusing these numbers blindly.
