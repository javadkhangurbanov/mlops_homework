# src/features.py
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from category_encoders import CatBoostEncoder


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """
    This function is for all transformations and feature engineering we do before
    applying the model. Like creating new features from existing ones and so on.
    """

    def __init__(
        self,
        trf_min_freq: float = 0.01,
        dev_man_min_freq: float = 0.01,
        group_rare_os: bool = True,
        os_min_freq: float = 0.01,
        cap_dev_num_at: int = 9,
        make_age_dev_cat: bool = True,
        age_dev_bin_days: int = 365,
    ):
        self.trf_min_freq = trf_min_freq
        self.dev_man_min_freq = dev_man_min_freq
        self.group_rare_os = group_rare_os
        self.os_min_freq = os_min_freq
        self.cap_dev_num_at = cap_dev_num_at
        self.make_age_dev_cat = make_age_dev_cat
        self.age_dev_bin_days = age_dev_bin_days

        # learned artifacts
        self._rare_trf_ = None
        self._rare_dev_man_ = None
        self._rare_os_ = None

    def fit(self, X: pd.DataFrame, y=None):
        df = X.copy()

        # Tariff rare groups (frequency on train)
        trf_freq = df["trf"].value_counts(normalize=True)
        self._rare_trf_ = set(trf_freq[trf_freq < self.trf_min_freq].index)

        # Device manufacturer rare groups
        # Normalize text case/spaces to reduce duplicates
        dev_man_clean = (
            df["dev_man"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True).str.upper()
        )
        dm_freq = dev_man_clean.value_counts(normalize=True)
        self._rare_dev_man_ = set(dm_freq[dm_freq < self.dev_man_min_freq].index)

        # Optional: OS rare groups (you mostly have Android/iOS/Proprietary; keep others as 'Other')
        if self.group_rare_os and "device_os_name" in df.columns:
            os_freq = df["device_os_name"].astype(str).value_counts(normalize=True)
            self._rare_os_ = set(os_freq[os_freq < self.os_min_freq].index)
        else:
            self._rare_os_ = set()

        return self

    def transform(self, X: pd.DataFrame):
        df = X.copy()

        # --- dev_num numeric + binning (9 means 9+) ---
        df["dev_num"] = pd.to_numeric(df["dev_num"], errors="coerce")
        df["dev_num_bin"] = df["dev_num"].apply(
            lambda x: np.nan if pd.isna(x) else (int(x) if int(x) <= self.cap_dev_num_at - 1 else self.cap_dev_num_at)
        )

        # --- dual_smart_combo ('0_0','0_1','1_0','1_1') ---
        # Coerce to int first to avoid 'True'/'False' strings
        df["is_dualsim"] = pd.to_numeric(df["is_dualsim"], errors="coerce").fillna(0).astype(int)
        df["is_smartphone"] = pd.to_numeric(df["is_smartphone"], errors="coerce").fillna(0).astype(int)
        df["dual_smart_combo"] = df["is_dualsim"].astype(str) + "_" + df["is_smartphone"].astype(str)

        # --- age_dev_cat (years) ---
        if self.make_age_dev_cat and "age_dev" in df.columns:
            df["age_dev"] = pd.to_numeric(df["age_dev"], errors="coerce")
            df["age_dev_cat"] = np.floor(df["age_dev"] / float(self.age_dev_bin_days))

        # --- trf_grouped ---
        df["trf_grouped"] = df["trf"].where(~df["trf"].isin(self._rare_trf_), "Other")

        # --- dev_man_grouped (normalize casing first to match fit) ---
        dev_man_clean = (
            df["dev_man"].astype(str).str.strip().str.replace(r"\s+", " ", regex=True).str.upper()
        )
        df["dev_man_grouped"] = dev_man_clean.where(~dev_man_clean.isin(self._rare_dev_man_), "Other")

        # --- device_os_name rare grouping (optional) ---
        if self._rare_os_:
            df["device_os_name"] = df["device_os_name"].astype(str)
            df["device_os_name"] = df["device_os_name"].where(
                ~df["device_os_name"].isin(self._rare_os_), "Other"
            )

        # Keep original region as-is (you chose not to group)
        return df


def make_preprocessor():
    """
    Full preprocessing:
      1) FeatureBuilder creates engineered columns from raw input.
      2) ColumnTransformer imputes/encodes for XGBoost.
    We return a Pipeline(fe -> ct) so train.py can use it directly.
    """
    # === categorical splits ===
    low_cardinal_cat_features = [
        'gndr',
        'is_dualsim',
        'is_smartphone',
        'dual_smart_combo',
        'device_os_name',
        'simcard_type'
    ]
    high_cardinal_cat_features = [
        'trf_grouped',
        'dev_man_grouped',
        'region'
    ]

    # === numeric splits ===
    numeric_continuous_features = [
        'age',
        'tenure',
        'age_dev'
    ]
    numeric_discrete_features = [
        'dev_num'
    ]

    # === pipelines ===
    numeric_cont_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean'))
    ])
    numeric_disc_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent'))
    ])

    low_cardinal_cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('ohe', OneHotEncoder(handle_unknown='ignore'))
    ])

    high_cardinal_cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('cb', CatBoostEncoder(cols=None, random_state=10))
    ])

    ct = ColumnTransformer(transformers=[
        ("num_cont", numeric_cont_pipeline, numeric_continuous_features),
        ("num_disc", numeric_disc_pipeline, numeric_discrete_features),
        ("cat_low",  low_cardinal_cat_pipeline,  low_cardinal_cat_features),
        ("cat_high", high_cardinal_cat_pipeline, high_cardinal_cat_features),
    ])

    pre = Pipeline(steps=[
        ("fe", FeatureBuilder(
            trf_min_freq=0.01,
            dev_man_min_freq=0.01,
            group_rare_os=True,
            os_min_freq=0.01,
            cap_dev_num_at=9,
            make_age_dev_cat=True,
            age_dev_bin_days=365,
        )),
        ("ct", ct),
    ])
    return pre
