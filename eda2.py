import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("corona_tested_individuals_ver_006.english.csv", low_memory=False)
df["test_date"] = pd.to_datetime(df["test_date"])

symptom_cols = ["cough","fever","sore_throat","shortness_of_breath","head_ache"]

# Following the paper: drop "other" result rows, keep positive/negative only
df_bin = df[df["corona_result"].isin(["positive","negative"])].copy()
df_bin["target"] = (df_bin["corona_result"] == "positive").astype(int)

print("Rows total:", len(df), "| rows after dropping 'other':", len(df_bin), "dropped:", len(df)-len(df_bin))

# ---- 1. Class balance ----
plt.figure(figsize=(5,4))
counts = df_bin["target"].value_counts().sort_index()
plt.bar(["Negative","Positive"], counts.values, color=["#4C72B0","#DD8452"])
for i,v in enumerate(counts.values):
    plt.text(i, v+1000, f"{v:,}\n({v/len(df_bin)*100:.1f}%)", ha="center")
plt.title("Class balance: COVID test result")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("eda_output/01_class_balance.png", dpi=120)
plt.close()

# ---- 2. Symptom prevalence overall vs positive vs negative ----
rates_overall = df_bin[symptom_cols].mean()
rates_pos = df_bin[df_bin.target==1][symptom_cols].mean()
rates_neg = df_bin[df_bin.target==0][symptom_cols].mean()

plt.figure(figsize=(8,5))
x = np.arange(len(symptom_cols))
w = 0.35
plt.bar(x-w/2, rates_neg.values*100, width=w, label="Negative", color="#4C72B0")
plt.bar(x+w/2, rates_pos.values*100, width=w, label="Positive", color="#DD8452")
plt.xticks(x, symptom_cols, rotation=20)
plt.ylabel("% reporting symptom")
plt.title("Symptom prevalence by test result")
plt.legend()
plt.tight_layout()
plt.savefig("eda_output/02_symptom_prevalence.png", dpi=120)
plt.close()

# ---- 3. Positivity rate by number of symptoms ----
df_bin["n_symptoms"] = df_bin[symptom_cols].sum(axis=1)
sym_rate = df_bin.groupby("n_symptoms")["target"].agg(["mean","count"])
plt.figure(figsize=(6,4))
plt.bar(sym_rate.index.astype(str), sym_rate["mean"]*100, color="#55A868")
for i,(m,c) in enumerate(zip(sym_rate["mean"],sym_rate["count"])):
    plt.text(i, m*100+0.5, f"n={c:,}", ha="center", fontsize=8)
plt.xlabel("Number of symptoms reported")
plt.ylabel("Positivity rate (%)")
plt.title("Positivity rate vs number of symptoms")
plt.tight_layout()
plt.savefig("eda_output/03_positivity_by_nsymptoms.png", dpi=120)
plt.close()

# ---- 4. Positivity by test_indication ----
ind_rate = df_bin.groupby("test_indication")["target"].agg(["mean","count"]).sort_values("mean")
plt.figure(figsize=(6,4))
plt.barh(ind_rate.index, ind_rate["mean"]*100, color="#C44E52")
for i,(m,c) in enumerate(zip(ind_rate["mean"],ind_rate["count"])):
    plt.text(m*100+0.3, i, f"n={c:,}", va="center", fontsize=8)
plt.xlabel("Positivity rate (%)")
plt.title("Positivity rate by test indication")
plt.tight_layout()
plt.savefig("eda_output/04_positivity_by_indication.png", dpi=120)
plt.close()

# ---- 5. Positivity by age_60_and_above and gender ----
fig, axes = plt.subplots(1,2, figsize=(10,4))
age_rate = df_bin.dropna(subset=["age_60_and_above"]).groupby("age_60_and_above")["target"].mean()*100
axes[0].bar(age_rate.index, age_rate.values, color="#8172B2")
axes[0].set_title("Positivity rate by age 60+")
axes[0].set_ylabel("%")
gender_rate = df_bin.dropna(subset=["gender"]).groupby("gender")["target"].mean()*100
axes[1].bar(gender_rate.index, gender_rate.values, color="#937860")
axes[1].set_title("Positivity rate by gender")
plt.tight_layout()
plt.savefig("eda_output/05_positivity_age_gender.png", dpi=120)
plt.close()

# ---- 6. Daily tests and positivity over time ----
daily = df_bin.set_index("test_date").resample("D").agg(tests=("target","count"), positives=("target","sum"))
daily["rate"] = daily["positives"]/daily["tests"]*100
fig, ax1 = plt.subplots(figsize=(10,4))
ax1.bar(daily.index, daily["tests"], color="#B0C4DE", label="Tests/day")
ax1.set_ylabel("Tests per day")
ax2 = ax1.twinx()
ax2.plot(daily.index, daily["rate"], color="#C44E52", marker="o", markersize=3, label="Positivity rate %")
ax2.set_ylabel("Positivity rate (%)")
fig.suptitle("Daily testing volume and positivity rate (Mar 11 - Apr 30, 2020)")
fig.legend(loc="upper right")
plt.tight_layout()
plt.savefig("eda_output/06_daily_trend.png", dpi=120)
plt.close()

# ---- 7. Correlation heatmap (symptoms + target) ----
corr_df = df_bin[symptom_cols+["target"]].copy()
corr = corr_df.corr()
plt.figure(figsize=(6,5))
im = plt.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
plt.colorbar(im)
plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
plt.yticks(range(len(corr.columns)), corr.columns)
for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        plt.text(j,i,f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=8)
plt.title("Correlation matrix (symptoms vs target)")
plt.tight_layout()
plt.savefig("eda_output/07_correlation_heatmap.png", dpi=120)
plt.close()

# save numeric summaries to reuse in markdown
summary = {}
summary["n_total"] = len(df)
summary["n_other_dropped"] = len(df) - len(df_bin)
summary["n_bin"] = len(df_bin)
summary["class_counts"] = counts.to_dict()
summary["class_pct_pos"] = counts[1]/len(df_bin)*100
summary["rates_overall"] = rates_overall.to_dict()
summary["rates_pos"] = rates_pos.to_dict()
summary["rates_neg"] = rates_neg.to_dict()
summary["sym_rate"] = sym_rate.to_dict()
summary["ind_rate"] = ind_rate.to_dict()
summary["age_rate"] = age_rate.to_dict()
summary["gender_rate"] = gender_rate.to_dict()
summary["age_missing"] = df_bin["age_60_and_above"].isna().sum()
summary["gender_missing"] = df_bin["gender"].isna().sum()
summary["corr"] = corr.to_dict()
summary["daily_mean_tests"] = daily["tests"].mean()
summary["daily_rate_min"] = daily["rate"].min()
summary["daily_rate_max"] = daily["rate"].max()
summary["n_symptoms_zero_rate"] = sym_rate.loc[0,"mean"]*100 if 0 in sym_rate.index else None
summary["n_symptoms_zero_count"] = int(sym_rate.loc[0,"count"]) if 0 in sym_rate.index else None

import json
with open("eda_output/summary.json","w") as f:
    json.dump(summary, f, indent=2, default=str)

print("done full analysis")
print(json.dumps(summary, indent=2, default=str))
