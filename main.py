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
