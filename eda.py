import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

pd.set_option("display.width", 120)

df = pd.read_csv("corona_tested_individuals_ver_006.english.csv")

report = []
def log(x=""):
    report.append(str(x))

log("# EDA raw dump\n")

log("## Shape")
log(df.shape)

log("\n## Dtypes")
log(df.dtypes)

log("\n## Head")
log(df.head(10))

log("\n## Missing values (count)")
log(df.isna().sum())

log("\n## Unique values per column")
for c in df.columns:
    log(f"{c}: {df[c].nunique()} unique -> {df[c].unique()[:10]}")

log("\n## corona_result value counts")
log(df["corona_result"].value_counts(dropna=False))

log("\n## age_60_and_above value counts")
log(df["age_60_and_above"].value_counts(dropna=False))

log("\n## gender value counts")
log(df["gender"].value_counts(dropna=False))

log("\n## test_indication value counts")
log(df["test_indication"].value_counts(dropna=False))

log("\n## test_date range")
log((df["test_date"].min(), df["test_date"].max(), df["test_date"].nunique()))

symptom_cols = ["cough","fever","sore_throat","shortness_of_breath","head_ache"]
log("\n## symptom columns value counts")
for c in symptom_cols:
    log(f"{c}:")
    log(df[c].value_counts(dropna=False))

with open("eda_output/raw_dump.txt","w") as f:
    f.write("\n".join(log_line if isinstance(log_line,str) else repr(log_line) for log_line in report))

print("done basic dump")
