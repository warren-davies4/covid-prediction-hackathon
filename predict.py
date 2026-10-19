"""
Load a trained model and run predictions on new data, without needing
to retrain anything. Point teammates here instead of the training
scripts (replicate_paper.py / balanced_features.py /
combined_biased_feature.py / tune_hyperparameters.py) - those are for
reproducing/extending the experiments; this is for just using a model.

Usage:
    from predict import predict
    import pandas as pd

    raw_df = pd.read_csv("some_new_data.csv")  # same columns as the
                                                # original MoH export:
                                                # cough, fever, sore_throat,
                                                # shortness_of_breath, head_ache,
                                                # age_60_and_above, gender,
                                                # test_indication
    result = predict(raw_df, model_name="tuned")
    # result has two new columns: "covid_probability" and "covid_prediction"

Available model_name values and the F1-optimal decision thresholds
found in classification_reports.py (recompute these for your own data
if the class balance is very different from the paper's test week):

    "full"      - all 8 original features, paper's hyperparameters   (threshold 0.099)
    "balanced"  - 5 features, 3 biased-reporting symptoms dropped    (threshold 0.131)
    "combined"  - 6 features, 3 biased symptoms OR'd into one flag   (threshold 0.102)
    "tuned"     - all 8 features, Optuna-tuned hyperparameters       (threshold 0.384,
                   best precision/recall/F1 of the four - see TUNING_FINDINGS.md
                   for the caveat about it leaning harder on the biased symptoms)
"""
import numpy as np
import pandas as pd
import lightgbm as lgb

MODELS = {
    "full": {
        "file": "lgbm_replication_model.txt",
        "features": ["sex_male", "age_60_plus", "cough", "fever", "sore_throat",
                     "shortness_of_breath", "head_ache", "contact_confirmed"],
        "threshold": 0.0991,
    },
    "balanced": {
        "file": "lgbm_balanced_features_model.txt",
        "features": ["sex_male", "age_60_plus", "cough", "fever", "contact_confirmed"],
        "threshold": 0.1308,
    },
    "combined": {
        "file": "lgbm_combined_biased_feature_model.txt",
        "features": ["sex_male", "age_60_plus", "cough", "fever", "contact_confirmed",
                     "any_biased_symptom"],
        "threshold": 0.1019,
    },
    "tuned": {
        "file": "lgbm_tuned_model.txt",
        "features": ["sex_male", "age_60_plus", "cough", "fever", "sore_throat",
                      "shortness_of_breath", "head_ache", "contact_confirmed"],
        "threshold": 0.3835,
    },
}


def engineer_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Derive the model's input columns from the original MoH export columns."""
    df = raw_df.copy()
    df["sex_male"] = (df["gender"] == "male").astype("float")
    df.loc[df["gender"].isna(), "sex_male"] = np.nan
    df["age_60_plus"] = df["age_60_and_above"].map({"Yes": 1.0, "No": 0.0})
    df["contact_confirmed"] = (df["test_indication"] == "Contact with confirmed").astype(float)
    df["any_biased_symptom"] = (
        (df["head_ache"].fillna(0) == 1)
        | (df["sore_throat"].fillna(0) == 1)
        | (df["shortness_of_breath"].fillna(0) == 1)
    ).astype(float)
    return df


def predict(raw_df: pd.DataFrame, model_name: str = "tuned", threshold: float | None = None) -> pd.DataFrame:
    """Return raw_df with two added columns: covid_probability and covid_prediction."""
    if model_name not in MODELS:
        raise ValueError(f"model_name must be one of {list(MODELS)}, got {model_name!r}")

    config = MODELS[model_name]
    engineered = engineer_features(raw_df)

    booster = lgb.Booster(model_file=config["file"])
    probabilities = booster.predict(engineered[config["features"]])

    result = raw_df.copy()
    result["covid_probability"] = probabilities
    result["covid_prediction"] = (probabilities >= (threshold or config["threshold"])).astype(int)
    return result


if __name__ == "__main__":
    # Smoke test: run the tuned model over a handful of real rows and
    # print the result, so you can sanity-check this loads correctly.
    df = pd.read_csv("corona_tested_individuals_ver_006.english.csv", low_memory=False, nrows=20)
    print(predict(df, model_name="tuned")[
        ["cough", "fever", "head_ache", "corona_result", "covid_probability", "covid_prediction"]
    ])
