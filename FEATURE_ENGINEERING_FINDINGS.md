# Feature engineering experiments

Baselines to compare against (from `REPLICATION_FINDINGS.md`):
- **Full 8 features** (`replicate_paper.py`): auROC **0.889**, auPRC **0.585**
- **Balanced 5 features**, biased symptoms dropped entirely (`balanced_features.py`): auROC **0.865**, auPRC **0.456**

## Combined biased-symptom feature

Script: `combined_biased_feature.py`. Model: `lgbm_combined_biased_feature_model.txt`.

**Idea:** instead of including `head_ache`, `sore_throat`, `shortness_of_breath` individually (full model) or dropping them entirely (balanced model), combine them into a single `any_biased_symptom` flag = 1 if the person reports *any* of the three, else 0. This keeps some signal from the three biased-reporting symptoms while diluting the overconfidence any single rare, biased feature can introduce (recall: in the full model, `head_ache` alone — reported for very few people, and reported almost only when later confirmed positive — ends up with a SHAP impact 3–4x larger than any other feature, which is a known artifact of its 96.2% "biased reporting" rate, not a genuinely stronger clinical signal).

Resulting feature set (6 total): `sex`, `age_60_plus`, `cough`, `fever`, `contact_confirmed`, `any_biased_symptom`.

| Metric | Full 8-feature | Balanced 5-feature | **Combined 6-feature** |
|---|---|---|---|
| auROC | 0.889 | 0.865 | **0.881** |
| auPRC | 0.585 | 0.456 | **0.453** |

**Result: a reasonable middle ground, but doesn't beat the full model.** The combined feature recovers most of the auROC gap vs. dropping the symptoms entirely (0.881 vs 0.865, closer to the full model's 0.889), confirming there's real signal in those three symptoms worth keeping in some form. But it doesn't beat the full 8-feature model on either metric, and auPRC in particular stays low (0.453, close to the balanced model's 0.456, well below the full model's 0.585).

**Why it likely falls short of the full model:** collapsing three symptoms into one OR flag throws away information about *which* symptom was reported and loses the ability to combine them (e.g. headache + sore throat together is more informative than either alone, which an OR can't distinguish from just one being true). The full model's main problem was that `head_ache` individually was over-trusted due to reporting bias — but LightGBM's tree splits can still extract genuine value from the other two biased symptoms when they're separate, more than an OR captures.

**SHAP ranking with the combined feature:** fever > cough > contact_confirmed > sex_male > any_biased_symptom > age_60_plus. The combined feature now sits mid-pack rather than dominating — the "overconfidence" artifact from standalone `head_ache` is gone, and the SHAP importance ordering now looks qualitatively similar to the paper's own (cough/fever/contact/sex/age all ranked above the biased-symptom group), while still using the information that was lost in the balanced-features drop.

**Takeaway to feed back:** the combined feature is a sensible way to *de-bias* the signal (confirmed by the SHAP plot looking much more sane than the full model's), but it costs some predictive power compared to keeping the three symptoms separate — at least with this hyperparameter set and this LightGBM model. If the goal is best possible auROC/auPRC, keep the symptoms separate; if the goal is a model whose feature-importance ranking is trustworthy/explainable (not overfit to a reporting artifact) while still using more information than just dropping them, this combined feature is the better trade-off. Worth retesting after hyperparameter tuning, since the "head_ache dominates" effect in the full model was partly a symptom of very few boosting iterations (10) being used before early stopping — a better-tuned full model might not lean on head_ache as heavily, which would change this comparison.
