"""
Preprocessing and feature engineering for the CDC Diabetes Health Indicators data.

I keep these pieces in a module (instead of inside the notebook) so the fitted
pipeline can be saved with joblib and loaded again later without redefining
anything. Both the scikit-learn models and the Keras models use the exact same
transformations.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET = "Diabetes_012"
CLASS_NAMES = ["No diabetes", "Prediabetes", "Diabetes"]

BINARY_COLS = [
    "HighBP", "HighChol", "CholCheck", "Smoker", "Stroke",
    "HeartDiseaseorAttack", "PhysActivity", "Fruits", "Veggies",
    "HvyAlcoholConsump", "AnyHealthcare", "NoDocbcCost", "DiffWalk", "Sex",
]
ORDINAL_COLS = ["GenHlth", "Age", "Education", "Income"]
CONTINUOUS_COLS = ["BMI", "MentHlth", "PhysHlth"]
RAW_FEATURES = BINARY_COLS + ORDINAL_COLS + CONTINUOUS_COLS

ENGINEERED_NUMERIC = [
    "BMI_Category", "CardioRiskCount", "LifestyleScore", "UnhealthyDays", "Age_x_BMI",
]
ENGINEERED_BINARY = [
    "FreqPhysDistress", "FreqMentDistress", "CareBarrier", "HighBP_and_HighChol",
]


def split_data(df, seed=42, val_size=0.15, test_size=0.15):
    """Stratified train / validation / test split (70 / 15 / 15 by default)."""
    X = df[RAW_FEATURES].copy()
    y = df[TARGET].astype(int).copy()
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=val_size + test_size, stratify=y, random_state=seed
    )
    rel_test = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=rel_test, stratify=y_temp, random_state=seed
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


class BMICapper(BaseEstimator, TransformerMixin):
    """Caps BMI at training-set percentiles.

    Very high BMI values in a self-reported phone survey are more likely to be
    reporting or entry errors than real measurements, but dropping the rows would
    throw away the rest of that person's answers. Capping keeps the row and limits
    the influence of the extreme value. Bounds are learned on training data only.
    """

    def __init__(self, lower_q=0.005, upper_q=0.995):
        self.lower_q = lower_q
        self.upper_q = upper_q

    def fit(self, X, y=None):
        self.lower_ = float(X["BMI"].quantile(self.lower_q))
        self.upper_ = float(X["BMI"].quantile(self.upper_q))
        return self

    def transform(self, X):
        X = X.copy()
        X["BMI"] = X["BMI"].clip(self.lower_, self.upper_)
        return X


def engineer_features(X):
    """Adds domain-driven features. Purely row-wise, so there is no leakage risk."""
    X = X.copy()
    # WHO BMI bands: underweight, normal, overweight, obese I, II, III
    X["BMI_Category"] = pd.cut(
        X["BMI"], bins=[0, 18.5, 25, 30, 35, 40, np.inf], labels=False, right=False
    ).astype(float)
    # How many cardiometabolic conditions the person already has (0-4)
    X["CardioRiskCount"] = (
        X["HighBP"] + X["HighChol"] + X["HeartDiseaseorAttack"] + X["Stroke"]
    )
    # Healthy habits minus unhealthy ones (0-5)
    X["LifestyleScore"] = (
        X["PhysActivity"] + X["Fruits"] + X["Veggies"]
        + (1 - X["Smoker"]) + (1 - X["HvyAlcoholConsump"])
    )
    X["UnhealthyDays"] = X["MentHlth"] + X["PhysHlth"]
    # CDC defines "frequent distress" as 14 or more bad days in the last 30
    X["FreqPhysDistress"] = (X["PhysHlth"] >= 14).astype(int)
    X["FreqMentDistress"] = (X["MentHlth"] >= 14).astype(int)
    # No coverage, or could not afford a doctor in the past year
    X["CareBarrier"] = ((X["AnyHealthcare"] == 0) | (X["NoDocbcCost"] == 1)).astype(int)
    X["HighBP_and_HighChol"] = (X["HighBP"] * X["HighChol"]).astype(int)
    # Older age and higher BMI together raise risk more than either alone;
    # linear models cannot learn this interaction on their own
    X["Age_x_BMI"] = X["Age"] * X["BMI"]
    return X


class FeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return engineer_features(X)


def build_preprocessor(use_engineered=True):
    """Full preprocessing pipeline: cap BMI -> (engineer features) -> scale.

    Continuous and ordinal columns are standardised. Binary columns are passed
    through as 0/1, since scaling them adds nothing and makes them harder to read.
    """
    numeric = CONTINUOUS_COLS + ORDINAL_COLS + (ENGINEERED_NUMERIC if use_engineered else [])
    binary = BINARY_COLS + (ENGINEERED_BINARY if use_engineered else [])
    scaler = ColumnTransformer(
        [("num", StandardScaler(), numeric), ("bin", "passthrough", binary)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    steps = [("cap_bmi", BMICapper())]
    if use_engineered:
        steps.append(("engineer", FeatureEngineer()))
    steps.append(("scale", scaler))
    return Pipeline(steps)


def get_feature_names(preprocessor):
    return list(preprocessor.named_steps["scale"].get_feature_names_out())
