import os, time, warnings
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

SEED = 42 # one seed used everywhere
def set_all_seeds(seed=SEED):
  """Seed python, numpy and Pytorch so taht run are repeatable"""
  random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
set_all_seeds(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 100
pd.set_option("display.max_columns", 50)
print("pandas", pd.__version__, "|Pytorch", torch.__version__, "| device:", DEVICE)

# loading dataset

raw_df = pd.read_csv("/content/data_2.csv")

# Make the column names match the projecr brief ("partners" instead of "dashers")
RENAME_MAP = {"total_onshift_dashers": "total_onshift_partners",
              "total_busy_dashers": "total_busy_partners"}

raw_df = raw_df.rename(columns=RENAME_MAP)

df = raw_df.copy() # work on a copy so the raw data is always available
print(f"Loaded {df.head}")

print("First five records")
display(df.head())
print("Last five records")
display(df.tail())

print(f"\nShape: {df.shape[0]:,} rows x {df.shape[1]} columns\n")
df.info()
print("\nDuplicate rows: ", df.duplicated().sum())

print("Descriptive statistics - numeric columns")
display(df.describe().T)

print("Descriptive statistics - timestamp columns (currently stored as text)")
display(df[["created_at", "actual_delivery_time"]].describe())

TIMESTAMP_COLS = ["created_at", "actual_delivery_time"]
CODE_COLS = [c for c in ["market_id", "order_protocol", "store_primary_category", "store_id"] if c in df.columns]
NUMERIC_COLS = [c for c in df.columns if c not in TIMESTAMP_COLS + CODE_COLS]

print("Timestamp columns (stored as text):", TIMESTAMP_COLS)
print("\nCategorical columns (numeric codes) and number of distinct values:")
for c in CODE_COLS:
    print(f"  {c:25s} {df[c].nunique():>4} distinct values")
print("\nNumerical columns:")
for c in NUMERIC_COLS:
    print("  ", c)

missing = pd.DataFrame({
    "missing_count": df.isnull().sum(),
    "missing_pct": ((df.isnull().sum() / len(df)) * 100).round(3)
})

display(missing.sort_values("missing_pct", ascending=False))
print("Total missing cells: ", int(missing["missing_count"].sum()))

rows_before = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Rows before: {rows_before:,} | after dropping duplicates: {len(df):,} | removed: {rows_before - len(df):,}")

# Convert text -> datetime64
for col in TIMESTAMP_COLS:
    df[col] = pd.to_datetime(df[col], errors="coerce")
print("Unparsable timestamps (NaT):", int(df[TIMESTAMP_COLS].isnull().sum().sum()))
df = df.dropna(subset=TIMESTAMP_COLS).reset_index(drop=True)

# Demonstration on the first order
start, end = df.loc[0, "created_at"], df.loc[0, "actual_delivery_time"]
print("\nTimestamp :", start, "->", type(start).__name__)
print("Timedelta :", end - start, "->", type(end - start).__name__)
print("Period    :", start.to_period("M"), "->", type(start.to_period("M")).__name__)

print("\nOrders per week, using Period('W') for grouping:")
display(df.groupby(df["created_at"].dt.to_period("W")).size().rename("orders"))
print(df[TIMESTAMP_COLS].dtypes)
