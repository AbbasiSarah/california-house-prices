"""California housing: loading, feature engineering and model definitions."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA = Path(__file__).resolve().parents[1] / "data" / "housing.csv"
TARGET = "median_house_value"
CAP = 500_001  # values were top-coded at this amount in the 1990 census release
NUMERIC = ["longitude", "latitude", "housing_median_age", "total_rooms", "total_bedrooms", "population",
           "households", "median_income", "rooms_per_household", "bedrooms_per_room", "people_per_household"]
CATEGORICAL = ["ocean_proximity"]


def load_data(path=DATA) -> pd.DataFrame:
    df = pd.read_csv(path)
    return add_features(df)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ratios are more informative than raw totals, which mostly measure district size."""
    df = df.copy()
    df["rooms_per_household"] = df["total_rooms"] / df["households"]
    df["bedrooms_per_room"] = df["total_bedrooms"] / df["total_rooms"]
    df["people_per_household"] = df["population"] / df["households"]
    return df


def spatial_groups(df: pd.DataFrame, cell=0.5) -> np.ndarray:
    """Grid cells of `cell` degrees: used to test on whole areas the model has not seen."""
    return (np.floor(df["latitude"] / cell).astype(int) * 1000 + np.floor(df["longitude"] / cell).astype(int)).values


def preprocessor(scale=True):
    num = [SimpleImputer(strategy="median")] + ([StandardScaler()] if scale else [])
    return ColumnTransformer([
        ("num", make_pipeline(*num), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])


def models() -> dict:
    return {
        "Baseline (median price)": make_pipeline(preprocessor(), DummyRegressor(strategy="median")),
        "Ridge regression": make_pipeline(preprocessor(), Ridge(alpha=1.0)),
        "Random forest": make_pipeline(preprocessor(scale=False),
                                       RandomForestRegressor(300, min_samples_leaf=2, n_jobs=-1, random_state=0)),
        "Gradient boosting": make_pipeline(preprocessor(scale=False),
                                           HistGradientBoostingRegressor(max_iter=600, learning_rate=0.06,
                                                                         max_leaf_nodes=48, random_state=0)),
    }
