# ── UNEMPLOYMENT RATE ANALYSIS IN INDIA ─────────────────────────
# Dataset: Kaggle - Unemployment in India
# Goal: Analyze trends, Covid-19 impact, regional patterns

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

# ── STEP 1: Load and Clean Data ──────────────────────────────────
print("=" * 55)
print("  UNEMPLOYMENT ANALYSIS - INDIA")
print("=" * 55)

df1 = pd.read_csv("Unemployment in India.csv")
df2 = pd.read_csv("Unemployment_Rate_upto_11_2020.csv")

# Clean column names
df1.columns = df1.columns.str.strip()
df2.columns = df2.columns.str.strip()

# Clean string values
for col in df1.select_dtypes("object").columns:
    df1[col] = df1[col].str.strip()
for col in df2.select_dtypes("object").columns:
    df2[col] = df2[col].str.strip()

# Convert dates
df1["Date"] = pd.to_datetime(df1["Date"], format="%d-%m-%Y")
df2["Date"] = pd.to_datetime(df2["Date"], format="%d-%m-%Y")

# Rename columns
col_map = {
    "Estimated Unemployment Rate (%)": "Unemployment_Rate",
    "Estimated Employed": "Employed",
    "Estimated Labour Participation Rate (%)": "Labour_Participation",
}
df1.rename(columns=col_map, inplace=True)
df2.rename(columns=col_map, inplace=True)

# Drop nulls
df1.dropna(subset=["Unemployment_Rate"], inplace=True)
df2.dropna(subset=["Unemployment_Rate"], inplace=True)

print(
    f"\nFile 1: {df1.shape[0]} records | {df1['Date'].min().date()} to {df1['Date'].max().date()}"
)
print(
    f"File 2: {df2.shape[0]} records | {df2['Date'].min().date()} to {df2['Date'].max().date()}"
)
print(f"Regions covered: {df1['Region'].nunique()}")

# ── STEP 2: Basic Statistics ─────────────────────────────────────
print("\n--- Overall Unemployment Statistics ---")
print(df1["Unemployment_Rate"].describe().round(2))

# ── STEP 3: Prepare Data for Charts ─────────────────────────────
monthly = df2.groupby("Date")["Unemployment_Rate"].mean().reset_index()
monthly.sort_values("Date", inplace=True)

regional = (
    df2.groupby("Region")["Unemployment_Rate"].mean().sort_values(ascending=False)
)

df2["Period"] = df2["Date"].apply(
    lambda x: (
        "During Covid\n(Apr-Nov 2020)"
        if x >= pd.Timestamp("2020-04-01")
        else "Pre Covid\n(Jan-Mar 2020)"
    )
)
covid_impact = df2.groupby("Period")["Unemployment_Rate"].mean()

area_trend = df1.groupby(["Date", "Area"])["Unemployment_Rate"].mean().reset_index()

print("\n--- Covid Impact ---")
print(covid_impact.round(2))

# ════════════════════════════════════════════════════════
# CHART 1: Monthly Trend with Covid marker
# ════════════════════════════════════════════════════════
print("\nShowing Chart 1 of 5 - Monthly Trend...")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(
    monthly["Date"],
    monthly["Unemployment_Rate"],
    color="#2980b9",
    linewidth=2.5,
    marker="o",
    markersize=5,
)
ax.axvline(
    x=pd.Timestamp("2020-03-25"),
    color="red",
    linestyle="--",
    linewidth=2,
    label="India Lockdown (Mar 25)",
)
ax.axvspan(
    pd.Timestamp("2020-03-25"),
    pd.Timestamp("2020-06-30"),
    alpha=0.15,
    color="red",
    label="Lockdown Period",
)
ax.set_title(
    "Chart 1 — Monthly Unemployment Rate in India (2020)",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
ax.set_xlabel("Month", fontsize=12)
ax.set_ylabel("Unemployment Rate (%)", fontsize=12)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("chart1_monthly_trend.png", dpi=150, bbox_inches="tight")
plt.show()

# ════════════════════════════════════════════════════════
# CHART 2: Regional Unemployment
# ════════════════════════════════════════════════════════
print("Showing Chart 2 of 5 - Regional Unemployment...")
fig, ax = plt.subplots(figsize=(14, 8))
colors = [
    (
        "#e74c3c"
        if x == regional.max()
        else "#2ecc71" if x == regional.min() else "#3498db"
    )
    for x in regional.values
]
bars = ax.barh(regional.index, regional.values, color=colors, edgecolor="white")
ax.set_title(
    "Chart 2 — Average Unemployment Rate by State (2020)",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
ax.set_xlabel("Average Unemployment Rate (%)", fontsize=12)
for bar, val in zip(bars, regional.values):
    ax.text(
        bar.get_width() + 0.3,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.1f}%",
        va="center",
        fontsize=10,
    )
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig("chart2_regional.png", dpi=150, bbox_inches="tight")
plt.show()

# ════════════════════════════════════════════════════════
# CHART 3: Covid Impact Comparison
# ════════════════════════════════════════════════════════
print("Showing Chart 3 of 5 - Covid Impact...")
fig, ax = plt.subplots(figsize=(8, 6))
bars = ax.bar(
    covid_impact.index,
    covid_impact.values,
    color=["#2ecc71", "#e74c3c"],
    width=0.4,
    edgecolor="white",
)
ax.set_title(
    "Chart 3 — Covid-19 Impact on Unemployment", fontsize=14, fontweight="bold", pad=15
)
ax.set_ylabel("Average Unemployment Rate (%)", fontsize=12)
ax.set_ylim(0, max(covid_impact.values) * 1.3)
for bar, val in zip(bars, covid_impact.values):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.5,
        f"{val:.2f}%",
        ha="center",
        fontsize=13,
        fontweight="bold",
    )
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("chart3_covid_impact.png", dpi=150, bbox_inches="tight")
plt.show()

# ════════════════════════════════════════════════════════
# CHART 4: Rural vs Urban Trend
# ════════════════════════════════════════════════════════
print("Showing Chart 4 of 5 - Rural vs Urban...")
fig, ax = plt.subplots(figsize=(12, 6))
for area, color, style in zip(["Rural", "Urban"], ["#27ae60", "#e67e22"], ["-", "--"]):
    data = area_trend[area_trend["Area"] == area].sort_values("Date")
    ax.plot(
        data["Date"],
        data["Unemployment_Rate"],
        label=area,
        color=color,
        linewidth=2.5,
        linestyle=style,
        marker="o",
        markersize=5,
    )
ax.axvline(
    x=pd.Timestamp("2020-03-25"),
    color="red",
    linestyle=":",
    linewidth=2,
    label="Lockdown Start",
)
ax.set_title(
    "Chart 4 — Rural vs Urban Unemployment Over Time",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Unemployment Rate (%)", fontsize=12)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("chart4_rural_urban.png", dpi=150, bbox_inches="tight")
plt.show()

# ════════════════════════════════════════════════════════
# CHART 5: Top 5 vs Bottom 5 States
# ════════════════════════════════════════════════════════
print("Showing Chart 5 of 5 - Top & Bottom States...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
top5 = regional.head(5)
bottom5 = regional.tail(5)

axes[0].barh(top5.index, top5.values, color="#e74c3c", edgecolor="white")
axes[0].set_title("Highest Unemployment States", fontsize=13, fontweight="bold")
axes[0].set_xlabel("Avg Unemployment Rate (%)", fontsize=11)
for i, (idx, val) in enumerate(top5.items()):
    axes[0].text(val + 0.2, i, f"{val:.1f}%", va="center", fontsize=11)

axes[1].barh(bottom5.index, bottom5.values, color="#2ecc71", edgecolor="white")
axes[1].set_title("Lowest Unemployment States", fontsize=13, fontweight="bold")
axes[1].set_xlabel("Avg Unemployment Rate (%)", fontsize=11)
for i, (idx, val) in enumerate(bottom5.items()):
    axes[1].text(val + 0.1, i, f"{val:.1f}%", va="center", fontsize=11)

for ax in axes:
    ax.grid(axis="x", alpha=0.3)

plt.suptitle(
    "Chart 5 — Best & Worst Performing States", fontsize=14, fontweight="bold", y=1.02
)
plt.tight_layout()
plt.savefig("chart5_top_bottom_states.png", dpi=150, bbox_inches="tight")
plt.show()

# ── FINAL SUMMARY ────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  KEY INSIGHTS")
print("=" * 55)
pre = covid_impact.iloc[0]
post = covid_impact.iloc[1]
print(f"\n1. Overall avg unemployment:  {df2['Unemployment_Rate'].mean():.2f}%")
print(f"2. Pre-Covid avg:             {pre:.2f}%")
print(f"3. During-Covid avg:          {post:.2f}%")
print(f"4. Covid spike:               +{abs(post - pre):.2f}%")
print(f"5. Highest state: {regional.idxmax()} ({regional.max():.2f}%)")
print(f"6. Lowest state:  {regional.idxmin()} ({regional.min():.2f}%)")
print(f"\nAll 5 charts saved in your project folder!")
