"""
preprocess.py
Handles all data cleaning, feature engineering (RFM scoring, tenure segmentation),
and preprocessing for the churn prediction pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df):,} records | Churn rate: {df['churn'].mean():.1%}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    RFM-inspired feature engineering:
    - Recency  → payment_delay (lower = better)
    - Frequency → num_products, support_calls
    - Monetary  → monthly_charges, total_charges
    Plus tenure segmentation buckets.
    """
    df = df.copy()

    # Tenure segmentation
    df["tenure_segment"] = pd.cut(
        df["tenure_months"],
        bins=[0, 12, 36, 60, 120],
        labels=["New (0-12m)", "Growing (1-3y)", "Established (3-5y)", "Loyal (5y+)"]
    )

    # Charge ratio — monthly as % of average total
    df["charge_ratio"] = df["monthly_charges"] / (df["total_charges"] / df["tenure_months"].clip(lower=1))

    # Risk flag — high support + high delay
    df["high_risk_flag"] = (
        (df["support_calls"] > 5) & (df["payment_delay"] > 10)
    ).astype(int)

    # Revenue tier
    df["revenue_tier"] = pd.qcut(
        df["monthly_charges"], q=4,
        labels=["Budget", "Standard", "Premium", "VIP"]
    )

    return df


def preprocess(df: pd.DataFrame):
    """
    Encode categoricals, scale numerics, return X, y, scaler, encoders.
    """
    df = engineer_features(df)

    cat_cols = ["gender", "contract_type", "internet_service",
                "tenure_segment", "revenue_tier"]
    num_cols = ["age", "tenure_months", "monthly_charges", "total_charges",
                "num_products", "support_calls", "payment_delay",
                "charge_ratio", "high_risk_flag", "senior_citizen"]

    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    X = df[num_cols + cat_cols]
    y = df["churn"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    return X_scaled, y, scaler, encoders
