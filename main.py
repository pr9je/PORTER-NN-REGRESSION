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
