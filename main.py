import time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, Markdown

from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import copy, random
import torch
from torch import nn
import torch.nn.functional as F

warnings.filterwarnings("ignore")

SEED = 42
def set_all_seeds(seed=SEED):
  """Seed python, numpy and PyTorch so that runs are repeatable"""
  random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
set_all_seeds(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 100
pd.set_option("display.max_columns", 50)
print("pandas",pd.__version__, "| Pytorch", torch.__version__, "| device:",DEVICE)

# Load the dataset

raw_df = pd.read_csv("/content/data_2.csv")

# Make the column names match the project brief ("partners" instead of dashers)
RENAME_MAP ={"total_onshit_dasher": "total_onshit_partners",
             "total_busy_dasher": "total_busy_partners"}

raw_df = raw_df.rename(columns=RENAME_MAP)

df = raw_df.copy()

print("First five records")
display(df.head())
print("="*50)
print("Last five records")
display(df.tail())

print(f"\nShape: {df.shape[0]:,} rows x {df.shape[1]} columns\n")
print("="*50)
df.info()
print("="*50)
print("\nDuplicate rows:", df.duplicated().sum())

print("Descriptive statistics - numerical columns")
display(df.describe().T)

print("="*100)
print("Descriptive statistics - timestamp columns (currently stored as object)")
display(df[["created_at", "actual_delivery_time"]].describe())

TIMESTAMP_COLS = ["created_at", "actual_delivery_time"]
CODE_COLS = [c for c in ["market_id", "order_protocol", "store_primary_category", "store_id"] if c in df.columns]
NUMERIC_COLS = [ c for c in df.columns if c not in TIMESTAMP_COLS + CODE_COLS]

print("Timestamp columns (stored as object):", TIMESTAMP_COLS)
print("\nCategorical columns (numeric codes) and number of distinct values:")
for c in CODE_COLS:
  print(f" {c:25s} {df[c].nunique():>4} distinct values")
print("\nNumerical Columns:")
for c in NUMERIC_COLS:
  print(" ",c)

missing = pd.DataFrame({
    "missing_count": df.isnull().sum(),
    "missing_pct": (df.isnull().mean() * 100).round(3),
})
print("*"*100)
display(missing.sort_values("missing_pct",ascending=False))
print("Total missing cells:", int(missing["missing_count"].sum()))

# convert text -> datetime64
for col in TIMESTAMP_COLS:
  df[col] = pd.to_datetime(df[col], errors="coerce")
print("Unparsable timestamp (NaT):", int(df[TIMESTAMP_COLS].isnull().sum().sum()))
df = df.dropna(subset=TIMESTAMP_COLS).reset_index(drop=True)

# Demonstration on the first order 
start, end = df.loc[0, "created_at"], df.loc[0, "actual_delivery_time"]
print("\nTimestamp :", start, "->", type(start).__name__)
print("Timedelta :", end - start, "->", type(end - start).__name__)
print("Period    :", start.to_period("M"), "->", type(start.to_period("M")).__name__)

print("\nOrders per week, using Period('W') for grouping:")
display(df.groupby(df["created_at"].dt.to_period("W")).size().rename("orders"))
print(df[TIMESTAMP_COLS].dtypes)

df["delivery_time_min"] = (df["actual_delivery_time"] - df["created_at"]).dt.total_seconds() / 60

MIN_MINUTES, MAX_MINUTES = 5, 240
too_fast = (df["delivery_time_min"] < MIN_MINUTES).sum()
too_slow = (df["delivery_time_min"] > MAX_MINUTES).sum()
print(f"Durations < {MIN_MINUTES} min: {too_fast} | > {MAX_MINUTES} min: {too_slow}")

df = df[df["delivery_time_min"].between(MIN_MINUTES, MAX_MINUTES)].reset_index(drop=True)
print(f"Rows kept: {len(df):,}\n")
display(df["delivery_time_min"].describe(percentiles=[.01, .05, .25, .5, .75, .95, .99]).round(2).to_frame().T)

# time based features from "created_at"
df["hour"] = df["created_at"].dt.hour
df["day_of_week"] = df["created_at"].dt.day_name()
df["day_of_month"] = df["created_at"].dt.day
df["month"] = df["created_at"].dt.month
df["is_weekend"] = df["created_at"].dt.day_of_week.isin([5,6]).astype(int)
df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

print("Order dates:", df["created_at"].min(), "->", df["created_at"].max())
display(df[["created_at", "hour", "day_of_week", "day_of_month", "month", "is_weekend", "hour_sin", "hour_cos"]].head())

# Cleaning invalid values and data-type conversion
validity_rules = {
    "total_items":              lambda s: s <= 0,
    "num_distinct_items":       lambda s: s <= 0,
    "subtotal":                 lambda s: s <= 0,
    "min_item_price":           lambda s: s <= 0,
    "max_item_price":           lambda s: s <= 0,
    "total_onshift_partners":   lambda s: s < 0,
    "total_busy_partners":      lambda s: s < 0,
    "total_outstanding_orders": lambda s: s < 0,
    "estimated_store_to_consumer_driving_duration": lambda s: s <= 0,
}

report = []
for col, rule in validity_rules.items():
    if col in df.columns:
        bad = rule(df[col])
        report.append({"column": col, "invalid_values": int(bad.sum()), "pct": round(bad.mean() * 100, 3)})
        df.loc[bad, col] = np.nan
display(pd.DataFrame(report))

# Logical inconsistency: cheapest item priced higher than the costliest item
inconsistent = (df["min_item_price"] > df["max_item_price"]).sum()
print(f"\nRows where min_item_price > max_item_price: {inconsistent:,} "
      "(reported, left unchanged: the gaps are small and tree/neural models are robust to this)")

# Code columns -> integer labels (only if no missing values)
for col in ["market_id", "order_protocol"]:
    if col in df.columns and df[col].notna().all():
        df[col] = df[col].astype(int)
print(f"\nMissing cells after validity cleaning (to be imputed in the pipeline): {int(df.isnull().sum().sum()):,}")

# Fleet-load features
df["onshift"] = df["total_onshift_dashers"].where(df["total_onshift_dashers"] > 0)   # 0 or NaN -> NaN (avoids divide-by-zero)
df["available_partners"] = df["total_onshift_dashers"] - df["total_busy_dashers"]
df["busy_ratio"] = (df["total_busy_dashers"] / df["onshift"]).replace([np.inf, -np.inf], np.nan)
df["outstanding_per_partner"] = (df["total_outstanding_orders"] / df["onshift"]).replace([np.inf, -np.inf], np.nan)
display(df[["total_onshift_dashers", "total_busy_dashers", "total_outstanding_orders","available_partners", "busy_ratio", "outstanding_per_partner"]].describe().T.round(2))

# Binary Flags
b, o = df["total_busy_partners"], df["total_onshift_partners"]
df["partners_overloaded"] = (b > o).astype(float).where(b.notna() & o.notna())
df["single_item_order"] = (df["num_distinct_items"] == 1).astype(float).where(df["num_distinct_items"].notna())
display(df[["is_weekend", "partners_overloaded", "single_item_order"]].mean().mul(100).round(1).to_frame("% of orders with flag = 1"))

# Choosng features and splittingthe data (before any fittingg)
TARGET = "delivery_time_min"

NUMERIC_FEATURES = [c for c in [
    "total_items", "subtotal", "num_distinct_items", "min_item_price", "max_item_price",
    "total_onshift_partners", "total_busy_partners", "total_outstanding_orders",
    "estimated_store_to_consumer_driving_duration",
    "available_partners", "busy_ratio", "outstanding_per_partner",
    "hour", "hour_sin", "hour_cos",
    "is_weekend", "partners_overloaded", "single_item_order"] if c in df.columns]
CATEGORICAL_FEATURES = [c for c in ["market_id", "order_protocol", "store_primary_category", "day_of_week"] if c in df.columns]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ---- Leakage guard: these must never be model inputs ----
FORBIDDEN = {"actual_delivery_time", "created_at", TARGET}
assert FORBIDDEN.isdisjoint(FEATURES), "Target leakage: a forbidden column is in the feature list!"
print("Leakage check passed.")
print(f"{len(NUMERIC_FEATURES)} numeric + {len(CATEGORICAL_FEATURES)} categorical features")

X = df[FEATURES]
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED)
X_train, X_test = X_train.copy(), X_test.copy()
print(f"Train: {X_train.shape[0]:,} rows | Test: {X_test.shape[0]:,} rows")

# Peak-hour flag: the busiest hours are learned from the TRAINING set only, then applied to both sets
PEAK_HOURS = sorted(X_train["hour"].value_counts().head(5).index.tolist())
for frame in (X_train, X_test):
    frame["is_peak_hour"] = frame["hour"].isin(PEAK_HOURS).astype(int)
NUMERIC_FEATURES.append("is_peak_hour"); FEATURES.append("is_peak_hour")
print("Peak hours (5 busiest in training data, hours as stored):", PEAK_HOURS)
print(f"{len(NUMERIC_FEATURES)} numeric + {len(CATEGORICAL_FEATURES)} categorical features in total")

# Preprocessing pipeline: imputation outlier capping, scaling encoding
class IQRCapper(BaseEstimator, TransformerMixin):
    """Winsorise each column to [Q1 - k*IQR, Q3 + k*IQR].
    The fences are learned in fit() (training data only) and re-used in transform()."""
    def __init__(self, factor=3.0):
        self.factor = factor

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1, q3 = np.nanpercentile(X, [25, 75], axis=0)
        iqr = q3 - q1
        self.lower_ = np.where(iqr > 0, q1 - self.factor * iqr, -np.inf)   # IQR = 0 -> do not cap
        self.upper_ = np.where(iqr > 0, q3 + self.factor * iqr, np.inf)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def make_preprocessor(scale=False, cap_factor=None):
    """Build a fresh (unfitted) preprocessing pipeline."""
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if cap_factor is not None:
        num_steps.append(("cap", IQRCapper(factor=cap_factor)))
    if scale:
        num_steps.append(("scale", StandardScaler()))
    cat_steps = [("impute", SimpleImputer(strategy="most_frequent")),
                 ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
    return ColumnTransformer([("num", Pipeline(num_steps), NUMERIC_FEATURES),
                              ("cat", Pipeline(cat_steps), CATEGORICAL_FEATURES)])

# Quick sanity test: fit on TRAIN only, transform both, confirm no NaN remains
demo_prep = make_preprocessor(scale=True, cap_factor=3.0)
demo_train = demo_prep.fit_transform(X_train)
demo_test = demo_prep.transform(X_test)
print("Transformed train:", demo_train.shape, "| test:", demo_test.shape)
print("NaN left in train/test:", int(np.isnan(demo_train).sum()), "/", int(np.isnan(demo_test).sum()))

# Exo=ploratory Data Analysis and Outliner Treatment
eda = X_train.copy()
eda[TARGET] = y_train
BASE_NUMERIC = [c for c in ["total_items", "subtotal", "num_distinct_items", "min_item_price", "max_item_price",
                            "total_onshift_partners", "total_busy_partners", "total_outstanding_orders",
                            "estimated_store_to_consumer_driving_duration"] if c in eda.columns]
print("EDA rows (training set):", f"{len(eda):,}")

# Target Distribution and boxplot
fig, axes = plt.subplots(1, 2, figsize=(14, 3.6))
sns.histplot(eda[TARGET], bins=60, kde=True, color="steelblue", ax=axes[0])
axes[0].axvline(eda[TARGET].mean(), color="red", ls="--", label=f"Mean = {eda[TARGET].mean():.1f}")
axes[0].axvline(eda[TARGET].median(), color="green", ls="--", label=f"Median = {eda[TARGET].median():.1f}")
axes[0].set(title="Distribution of delivery time", xlabel="Delivery time (minutes)", ylabel="Number of orders")
axes[0].legend()
sns.boxplot(x=eda[TARGET], color="lightsteelblue", ax=axes[1])
axes[1].set(title="Boxplot of delivery time", xlabel="Delivery time (minutes)")
plt.tight_layout(); plt.show()

lo, hi = eda[TARGET].quantile(0.25) - 1.5 * (eda[TARGET].quantile(0.75) - eda[TARGET].quantile(0.25)), \
         eda[TARGET].quantile(0.75) + 1.5 * (eda[TARGET].quantile(0.75) - eda[TARGET].quantile(0.25))
print(f"Mean {eda[TARGET].mean():.1f} min | median {eda[TARGET].median():.1f} | std {eda[TARGET].std():.1f} | "
      f"min {eda[TARGET].min():.0f} | max {eda[TARGET].max():.0f} | skewness {eda[TARGET].skew():.2f}")
print(f"{(eda[TARGET] > hi).mean()*100:.2f}% of orders are above the 1.5xIQR upper fence ({hi:.1f} min).")

# Univariate Analysis: distribution of every continoues variables
UNI_COLS = [c for c in BASE_NUMERIC + ["busy_ratio", "outstanding_per_partner", TARGET] if c in eda.columns]
ncols = 3; nrows = int(np.ceil(len(UNI_COLS) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(15, 2.3 * nrows))
for ax, col in zip(axes.ravel(), UNI_COLS):
    sns.histplot(eda[col].dropna(), bins=40, color="steelblue", ax=ax)
    ax.set(title=f"{col} (skew {eda[col].skew():.2f})", xlabel=col, ylabel="Orders")
for ax in axes.ravel()[len(UNI_COLS):]:
    ax.axis("off")
plt.tight_layout(); plt.show()

def describe_skew(s):
    if abs(s) < 0.5:
        return "roughly symmetric"
    side = "right-skewed (long tail of large values)" if s > 0 else "left-skewed (long tail of small values)"
    return side + (", strongly" if abs(s) > 1 else ", moderately")

lines = []
for col in UNI_COLS:
    s = eda[col].dropna(); med = s.median()
    ratio = f", max = {s.max() / med:.1f}x the median" if med > 0 else ""
    lines.append(f"* **{col}**: skewness {s.skew():.2f}, {describe_skew(s.skew())}; median {med:.1f}, mean {s.mean():.1f}{ratio}.")
display(Markdown("\n".join(lines)))

# Countplots of categorical features
fig, axes = plt.subplots(2, 2, figsize=(15, 8))

# Subplot 0, 0: Market ID
sns.countplot(data=eda, x="market_id", color="steelblue", ax=axes[0, 0])
axes[0, 0].set(title="Orders per market", xlabel="Market ID", ylabel="Orders")
axes[0, 0].bar_label(axes[0, 0].containers[0], fmt='%.0f')

# Subplot 0, 1: Order Protocol
sns.countplot(data=eda, x="order_protocol", color="seagreen", ax=axes[0, 1])
axes[0, 1].set(title="Orders per order protocol", xlabel="Order protocol (code)", ylabel="Orders")
axes[0, 1].bar_label(axes[0, 1].containers[0], fmt='%.0f')

# Subplot 1, 0: Day of Week
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
sns.countplot(data=eda, x="day_of_week", order=day_order, color="darkorange", ax=axes[1, 0])
axes[1, 0].set(title="Orders per day of week", xlabel="Day of week", ylabel="Orders")
axes[1, 0].tick_params(axis="x", rotation=30)
axes[1, 0].bar_label(axes[1, 0].containers[0], fmt='%.0f')

# Subplot 1, 1: Top 15 Categories
top_cats = eda["store_primary_category"].value_counts().head(15).index
sns.countplot(data=eda[eda["store_primary_category"].isin(top_cats)], x="store_primary_category",
              order=top_cats, color="mediumpurple", ax=axes[1, 1])
axes[1, 1].set(title="Top 15 restaurant categories by order volume", xlabel="Store category (code)", ylabel="Orders")
axes[1, 1].tick_params(axis="x", rotation=60)
axes[1, 1].bar_label(axes[1, 1].containers[0], fmt='%.0f', rotation=90, padding=3)

plt.tight_layout()
plt.show()

print("Markets: top market holds", f"{eda['market_id'].value_counts(normalize=True).iloc[0]*100:.1f}% of orders")
print("Order protocols: top protocol holds", f"{eda['order_protocol'].value_counts(normalize=True).iloc[0]*100:.1f}% of orders")
print(f"Store categories: {eda['store_primary_category'].nunique()} in total; top 15 cover "
      f"{eda['store_primary_category'].isin(top_cats).mean()*100:.1f}% of orders")

# Bivariate Analysis : delivery time by category (box plots)
fig, axes = plt.subplots(2, 2, figsize=(15, 6.2))
sns.boxplot(data=eda, x="market_id", y=TARGET, color="steelblue", fliersize=1, ax=axes[0, 0])
axes[0, 0].set(title="Delivery time by market", xlabel="Market ID", ylabel="Delivery time (min)")
sns.boxplot(data=eda, x="order_protocol", y=TARGET, color="seagreen", fliersize=1, ax=axes[0, 1])
axes[0, 1].set(title="Delivery time by order protocol", xlabel="Order protocol (code)", ylabel="Delivery time (min)")
sns.boxplot(data=eda, x="day_of_week", y=TARGET, order=day_order, color="darkorange", fliersize=1, ax=axes[1, 0])
axes[1, 0].set(title="Delivery time by day of week", xlabel="Day of week", ylabel="Delivery time (min)")
axes[1, 0].tick_params(axis="x", rotation=30)
sns.boxplot(data=eda, x="partners_overloaded", y=TARGET, color="indianred", fliersize=1, ax=axes[1, 1])
axes[1, 1].set(title="Delivery time: partners overloaded (busy > on-shift)?", xlabel="0 = no, 1 = yes", ylabel="Delivery time (min)")
plt.tight_layout(); plt.show()

lines = []
for col in ["market_id", "order_protocol", "day_of_week", "partners_overloaded"]:
    med = eda.groupby(col)[TARGET].median()
    lines.append(f"* **{col}**: highest median delivery time at `{med.idxmax()}` ({med.max():.1f} min), lowest at `{med.idxmin()}` "
                 f"({med.min():.1f} min); spread between medians {med.max() - med.min():.1f} min.")
display(Markdown("\n".join(lines)))

# Correlation Heatmap
corr_cols = [c for c in NUMERIC_FEATURES if c not in ("hour_sin", "hour_cos")] + [TARGET]
corr = eda[corr_cols].corr()
plt.figure(figsize=(10, 7))
sns.heatmap(corr, annot=True, annot_kws={"size": 7}, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.4, cbar_kws={"label": "Pearson correlation"})
plt.title("Correlation heatmap (training set)")
plt.tight_layout(); plt.show()

target_corr = corr[TARGET].drop(TARGET).sort_values(key=np.abs, ascending=False)
print("Features ranked by |correlation| with delivery time:")
display(target_corr.round(3).to_frame("correlation"))
pairs = corr.drop(index=TARGET, columns=TARGET).where(np.triu(np.ones((len(corr)-1, len(corr)-1)), 1).astype(bool)).stack()
print("Most correlated feature pairs (possible redundancy):")
display(pairs.reindex(pairs.abs().sort_values(ascending=False).index).head(5).round(3).to_frame("correlation"))

# Scatterplots: numeric features vs delivery time
sample = eda.sample(5000, random_state=SEED)
fig, axes = plt.subplots(3, 3, figsize=(16, 8.5))
for ax, col in zip(axes.ravel(), BASE_NUMERIC):
    sns.regplot(data=sample, x=col, y=TARGET, scatter_kws={"alpha": 0.15, "s": 8},
                line_kws={"color": "red"}, ax=ax)
    ax.set(title=f"Delivery time vs {col}", xlabel=col, ylabel="Delivery time (min)")
for ax in axes.ravel()[len(BASE_NUMERIC):]:
    ax.axis("off")
plt.tight_layout(); plt.show()

# Delivery time by hour of day and by day of week

def mean_ci(frame, by):
    g = frame.groupby(by)[TARGET].agg(["mean", "std", "count"])
    g["ci"] = 1.96 * g["std"] / np.sqrt(g["count"])
    return g

by_hour = mean_ci(eda, "hour")
by_day = mean_ci(eda, "day_of_week").reindex(day_order)

fig, axes = plt.subplots(1, 2, figsize=(16, 3.8))
ax2 = axes[0].twinx()
ax2.bar(by_hour.index, by_hour["count"], color="lightgrey", alpha=0.6)
ax2.set_ylabel("Number of orders"); ax2.grid(False)
axes[0].errorbar(by_hour.index, by_hour["mean"], yerr=by_hour["ci"], marker="o", color="steelblue", capsize=3)
axes[0].set_zorder(ax2.get_zorder() + 1); axes[0].patch.set_visible(False)
axes[0].set(title="Mean delivery time by hour of day", xlabel="Hour of day (as stored in the data)", ylabel="Mean delivery time (min)")

bars = axes[1].bar(by_day.index, by_day["mean"], yerr=by_day["ci"], color="darkorange", capsize=4)
axes[1].bar_label(bars, fmt='%.1f', padding=3)
axes[1].set(title="Mean delivery time by day of week", xlabel="Day of week", ylabel="Mean delivery time (min)",
            ylim=(by_day["mean"].min() - 3, by_day["mean"].max() + 3))
axes[1].tick_params(axis="x", rotation=30)
plt.tight_layout(); plt.show()

print(f"Slowest hour: {by_hour['mean'].idxmax()}:00 ({by_hour['mean'].max():.1f} min) | "f"fastest hour: {by_hour['mean'].idxmin()}:00 ({by_hour['mean'].min():.1f} min)")
print(f"Slowest day: {by_day['mean'].idxmax()} ({by_day['mean'].max():.1f} min) | "f"fastest day: {by_day['mean'].idxmin()} ({by_day['mean'].min():.1f} min)")

def decile_plot(frame, col, ax, color, series_source=None):
    # If the column is missing in 'frame' but present in the main 'df', we can pull it using the matching index
    if col not in frame.columns and series_source is not None and col in series_source.columns:
        col_data = series_source.loc[frame.index, col]
    else:
        col_data = frame[col]

    bins = pd.qcut(col_data, q=10, duplicates="drop")
    g = frame.groupby(bins, observed=True)[TARGET].agg(["mean", "std", "count"])
    g["ci"] = 1.96 * g["std"] / np.sqrt(g["count"])
    
    # Plot line chart
    ax.errorbar(range(len(g)), g["mean"], yerr=g["ci"], marker="o", color=color, capsize=3)
    
    # Add data labels for each point
    for idx, row in enumerate(g.itertuples()):
        ax.annotate(f"{row.mean:.1f}", 
                    (idx, row.mean), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center', 
                    fontsize=8, 
                    fontweight='bold', 
                    color=color)
        
    ax.set_xticks(range(len(g)))
    ax.set_xticklabels([f"{iv.left:.0f}-{iv.right:.0f}" for iv in g.index], rotation=45)
    ax.set(title=f"Mean delivery time by {col} (deciles)", xlabel=f"{col} (range per decile)", ylabel="Mean delivery time (min)")
    return g

fig, axes = plt.subplots(1, 2, figsize=(16, 4.2))
g_out = decile_plot(eda, "total_outstanding_orders", axes[0], "crimson", series_source=df)
g_busy = decile_plot(eda, "total_busy_dashers", axes[1], "teal", series_source=df)
plt.tight_layout(); plt.show()

print(f"Outstanding orders: lowest decile {g_out['mean'].iloc[0]:.1f} min -> highest decile {g_out['mean'].iloc[-1]:.1f} min "
      f"(change {g_out['mean'].iloc[-1]-g_out['mean'].iloc[0]:+.1f})")
print(f"Busy partners:      lowest decile {g_busy['mean'].iloc[0]:.1f} min -> highest decile {g_busy['mean'].iloc[-1]:.1f} min "
      f"(change {g_busy['mean'].iloc[-1]-g_busy['mean'].iloc[0]:+.1f})")

cat_stats = eda.groupby("store_primary_category")[TARGET].agg(["mean", "std", "count"])
cat_stats = cat_stats[cat_stats["count"] >= 500].sort_values("mean")
print(f"{len(cat_stats)} categories have >= 500 orders")

show = pd.concat([cat_stats.head(8), cat_stats.tail(8)]).drop_duplicates()
plt.figure(figsize=(15, 4.5))
colors = ["seagreen" if v < eda[TARGET].mean() else "indianred" for v in show["mean"]]
bars = plt.bar(show.index.astype(str), show["mean"], color=colors)
plt.bar_label(bars, fmt='%.1f', padding=3)
plt.axhline(eda[TARGET].mean(), color="black", ls="--", label=f"Overall mean = {eda[TARGET].mean():.1f}")
plt.ylim(show["mean"].min() - 2, show["mean"].max() + 2)
plt.title("Fastest and slowest restaurant categories (mean delivery time)")
plt.xlabel("Store primary category (code)"); plt.ylabel("Mean delivery time (min)"); plt.legend()
plt.tight_layout(); plt.show()
print(f"Spread between fastest and slowest category means: {cat_stats['mean'].max() - cat_stats['mean'].min():.1f} minutes")

# Outlier analysis with the IQR method
def iqr_bounds(series, k=1.5):
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr

def outlier_mask(series, k=1.5):
    lo, hi = iqr_bounds(series, k)
    return (series < lo) | (series > hi)          # NaN compares False -> never flagged

rows = []
for col in BASE_NUMERIC + [TARGET]:
    s = eda[col].dropna()
    q1, q3 = s.quantile([0.25, 0.75]); lo, hi = iqr_bounds(s, 1.5)
    rows.append({"column": col, "Q1": q1, "Q3": q3, "IQR": q3 - q1, "lower_fence": lo, "upper_fence": hi,
                 "max": s.max(), "outliers_1.5xIQR_%": outlier_mask(s, 1.5).mean() * 100,
                 "outliers_3xIQR_%": outlier_mask(s, 3).mean() * 100})
outlier_table = pd.DataFrame(rows).set_index("column").round(2)
display(outlier_table)

def before_after_boxplots(before, after, cols, title):
    """Horizontal boxplots per column. If `after` is None only the 'before' box is drawn."""
    n = len(cols); ncols = 3; nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 2.4 * nrows))
    axes = np.atleast_1d(axes)
    for ax, col in zip(axes.ravel(), cols):
        groups = [before[col].dropna()] + ([after[col].dropna()] if after is not None else [])
        ax.boxplot(groups, vert=False, widths=0.6, flierprops=dict(marker=".", markersize=2, alpha=0.3))
        ax.set_yticks(range(1, len(groups) + 1))
        ax.set_yticklabels(["Before", "After"][:len(groups)] if after is not None else ["Before"])
        ax.set(title=col, xlabel="Value")
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=14, y=1.01)
    plt.tight_layout(); plt.show()

OUT_COLS = BASE_NUMERIC

CAP_FACTOR = 3.0

# Treatment decision
# (a) Extreme-target removal: TRAINING rows only
t_lo, t_hi = iqr_bounds(y_train, CAP_FACTOR)
keep = y_train.between(t_lo, t_hi)
X_train_clean, y_train_clean = X_train[keep], y_train[keep]
print(f"Target fences (3xIQR): {t_lo:.1f} to {t_hi:.1f} minutes")
print(f"Training rows removed: {(~keep).sum():,} of {len(keep):,} ({(~keep).mean()*100:.2f}%)")
print(f"Test set untouched: {len(X_test):,} rows")

# (b) Feature capping fences (same maths the pipeline will use), for plotting only
plot_capper = IQRCapper(CAP_FACTOR).fit(eda[OUT_COLS].to_numpy(dtype=float))
eda_after = eda.copy()
eda_after[OUT_COLS] = plot_capper.transform(eda[OUT_COLS].to_numpy(dtype=float))
eda_after = eda_after[keep]
capped_share = pd.Series({c: ((eda[c] < lo_) | (eda[c] > hi_)).mean() * 100
                          for c, lo_, hi_ in zip(OUT_COLS, plot_capper.lower_, plot_capper.upper_)})
print("\nShare of values capped per column (%):")
display(capped_share.round(2).to_frame("capped_%").T)

# Before vs After Plots:
before_after_boxplots(eda, eda_after, OUT_COLS, "Boxplots of numeric features: BEFORE vs AFTER treatment")

fig, axes = plt.subplots(1, 2, figsize=(14, 3.6))
sns.histplot(eda[TARGET], bins=60, color="grey", alpha=0.5, label="Before", ax=axes[0])
sns.histplot(eda_after[TARGET], bins=60, color="steelblue", alpha=0.6, label="After", ax=axes[0])
axes[0].set(title="Delivery time distribution", xlabel="Delivery time (min)", ylabel="Orders"); axes[0].legend()

comp = pd.DataFrame({"Before": eda[TARGET].describe(), "After": eda_after[TARGET].describe()}).round(2)
axes[1].axis("off")
tbl = axes[1].table(cellText=comp.values, rowLabels=comp.index, colLabels=comp.columns, loc="center")
tbl.scale(1, 1.5); axes[1].set_title("Target summary statistics")
plt.tight_layout(); plt.show()

skew_cols = ["total_items", "subtotal", "max_item_price"]
print("Skewness before -> after treatment:")
for c in [c for c in skew_cols if c in eda.columns]:
    print(f"  {c:20s} {eda[c].skew():6.2f} -> {eda_after[c].skew():6.2f}")

# Validation slice from TRAIN only
X_fit, X_val, y_fit, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=SEED)

def regression_report(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float).ravel()
    mse = mean_squared_error(y_true, y_pred)
    return {"MAE": mean_absolute_error(y_true, y_pred), "MSE": mse, "RMSE": float(np.sqrt(mse)),
            "R2": r2_score(y_true, y_pred), "Within_5min_%": float(np.mean(np.abs(y_true - y_pred) <= 5) * 100)}

def try_strategy(label, keep_mask, cap_factor):
    prep = make_preprocessor(scale=False, cap_factor=cap_factor)
    Xf, yf = X_fit[keep_mask], y_fit[keep_mask]
    A = prep.fit_transform(Xf)
    model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, random_state=SEED).fit(A, yf)
    rep = regression_report(y_val, model.predict(prep.transform(X_val)))
    return {"strategy": label, "train_rows": int(keep_mask.sum()), **rep}

all_rows = pd.Series(True, index=X_fit.index)
lo3, hi3 = iqr_bounds(y_fit, 3)
keep_B = y_fit.between(lo3, hi3)
keep_C = pd.Series(True, index=X_fit.index)
for col in BASE_NUMERIC:
    keep_C &= ~outlier_mask(X_fit[col], 1.5)
keep_C &= ~outlier_mask(y_fit, 1.5)

impact = pd.DataFrame([
    try_strategy("A: no treatment", all_rows, None),
    try_strategy("B: cap 3xIQR + drop extreme targets (chosen)", keep_B, CAP_FACTOR),
    try_strategy("C: aggressive 1.5xIQR row deletion", keep_C, None),
]).set_index("strategy")
display(impact.round(3))

def print_metrics(name, rep):
    print(f"{name:28s} MAE {rep['MAE']:.3f} | MSE {rep['MSE']:.3f} | RMSE {rep['RMSE']:.3f} | "
          f"R2 {rep['R2']:.4f} | within 5 min {rep['Within_5min_%']:.1f}%")

dummy = DummyRegressor(strategy="mean").fit(X_train_clean, y_train_clean)
baseline_report = regression_report(y_test, dummy.predict(X_test))
print_metrics("Predict-the-mean baseline", baseline_report)
