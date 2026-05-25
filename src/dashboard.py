"""
dashboard.py
Streamlit app for the Churn Prediction & Revenue Analytics Dashboard.
Loads trained model, scores new customers, visualizes churn risk segments and KPIs.

Run: streamlit run src/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import load_data, engineer_features, preprocess

st.set_page_config(page_title="Churn Analytics Dashboard", layout="wide", page_icon="📊")

# ── Load model ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("models/churn_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_and_score():
    df_raw = load_data("data/customers.csv")
    bundle = load_model()
    X, y, _, _ = preprocess(df_raw)
    df_raw["churn_probability"] = bundle["model"].predict_proba(X)[:, 1]
    df_raw["churn_predicted"]   = (df_raw["churn_probability"] > 0.5).astype(int)
    df_raw["risk_tier"] = pd.cut(
        df_raw["churn_probability"],
        bins=[0, 0.30, 0.60, 1.0],
        labels=["Low Risk", "Medium Risk", "High Risk"]
    )
    df_feat = engineer_features(df_raw)
    return df_raw, df_feat

# ── Header ───────────────────────────────────────────────────────────────────
st.title("📊 Customer Churn Prediction & Revenue Analytics")
st.caption("Powered by Random Forest + SMOTE | Data Analyst Portfolio — Swaroop Batchu")

try:
    df, df_feat = load_and_score()
except FileNotFoundError:
    st.error("⚠️ Model not found. Run `python src/train.py` first, then relaunch.")
    st.stop()

# ── KPI Cards ────────────────────────────────────────────────────────────────
total       = len(df)
churned     = df["churn"].sum()
churn_rate  = churned / total
high_risk   = (df["risk_tier"] == "High Risk").sum()
avg_revenue = df["monthly_charges"].mean()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Customers",   f"{total:,}")
k2.metric("Churn Rate",        f"{churn_rate:.1%}")
k3.metric("High Risk Accounts",f"{high_risk:,}")
k4.metric("Avg Monthly Revenue",f"${avg_revenue:.2f}")

st.markdown("---")

# ── Row 1: Risk Tier Distribution + Contract Type Churn ──────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Churn Risk Tier Distribution")
    tier_counts = df["risk_tier"].value_counts()
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(tier_counts.index, tier_counts.values, color=colors)
    ax.set_ylabel("Customer Count")
    ax.set_title("Risk Tier Breakdown")
    for i, v in enumerate(tier_counts.values):
        ax.text(i, v + 100, f"{v:,}", ha="center", fontsize=9)
    st.pyplot(fig)

with col2:
    st.subheader("Churn Rate by Contract Type")
    contract_churn = df.groupby("contract_type")["churn"].mean().sort_values(ascending=False)
    fig2, ax2 = plt.subplots(figsize=(5, 3))
    bars = ax2.barh(contract_churn.index, contract_churn.values * 100, color="#3498db")
    ax2.set_xlabel("Churn Rate (%)")
    ax2.set_title("Churn % by Contract")
    for bar, val in zip(bars, contract_churn.values):
        ax2.text(val * 100 + 0.5, bar.get_y() + bar.get_height()/2,
                 f"{val:.1%}", va="center", fontsize=9)
    st.pyplot(fig2)

# ── Row 2: Revenue Trend + Tenure Segment ────────────────────────────────────
col3, col4 = st.columns(2)

with col3:
    st.subheader("Revenue by Risk Tier")
    rev_tier = df.groupby("risk_tier")["monthly_charges"].sum().reset_index()
    fig3, ax3 = plt.subplots(figsize=(5, 3))
    ax3.bar(rev_tier["risk_tier"], rev_tier["monthly_charges"] / 1000,
            color=["#2ecc71", "#f39c12", "#e74c3c"])
    ax3.set_ylabel("Revenue ($K)")
    ax3.set_title("Monthly Revenue at Risk")
    st.pyplot(fig3)

with col4:
    st.subheader("Churn Rate by Tenure Segment")
    tenure_churn = df_feat.groupby("tenure_segment", observed=True)["churn"].mean()
    fig4, ax4 = plt.subplots(figsize=(5, 3))
    ax4.bar(tenure_churn.index, tenure_churn.values * 100, color="#9b59b6")
    ax4.set_ylabel("Churn Rate (%)")
    ax4.set_title("Churn by Customer Tenure")
    plt.xticks(rotation=15, fontsize=8)
    st.pyplot(fig4)

st.markdown("---")

# ── High Risk Customer Table ──────────────────────────────────────────────────
st.subheader("🚨 High Risk Customers — Action Required")
high_risk_df = df[df["risk_tier"] == "High Risk"][[
    "customer_id", "tenure_months", "monthly_charges",
    "contract_type", "support_calls", "churn_probability"
]].sort_values("churn_probability", ascending=False).head(20)
high_risk_df["churn_probability"] = high_risk_df["churn_probability"].map("{:.1%}".format)
st.dataframe(high_risk_df, use_container_width=True)

# ── Export ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📥 Export Data")
csv = df[["customer_id","churn_probability","risk_tier","monthly_charges","contract_type"]]\
      .to_csv(index=False)
st.download_button("Download Risk Scores CSV", csv, "churn_risk_scores.csv", "text/csv")
