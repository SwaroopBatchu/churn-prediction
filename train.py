"""
train.py
Trains a Random Forest churn classifier with SMOTE oversampling.
Compares against a logistic regression baseline.
Saves the best model to models/churn_model.pkl
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score
)
from imblearn.over_sampling import SMOTE
from preprocess import load_data, preprocess


def train_baseline(X_train, y_train, X_test, y_test):
    """Logistic Regression baseline."""
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n── Baseline (Logistic Regression) ──")
    print(f"  Accuracy : {acc:.1%}")
    print(f"  ROC-AUC  : {roc_auc_score(y_test, lr.predict_proba(X_test)[:,1]):.3f}")
    return lr, acc


def train_random_forest(X_train, y_train, X_test, y_test):
    """Random Forest with SMOTE for class imbalance."""
    print("\n── Applying SMOTE for class balancing ──")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"  Before SMOTE: {y_train.value_counts().to_dict()}")
    print(f"  After  SMOTE: {pd.Series(y_res).value_counts().to_dict()}")

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_res, y_res)

    y_pred  = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]

    acc     = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    print(f"\n── Random Forest Results ──")
    print(f"  Accuracy  : {acc:.1%}")
    print(f"  ROC-AUC   : {roc_auc:.3f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['No Churn','Churn'])}")

    # Feature importance
    feat_imp = pd.Series(rf.feature_importances_, index=X_train.columns)
    print("\nTop 10 Feature Importances:")
    print(feat_imp.nlargest(10).to_string())

    return rf, acc


def save_model(model, scaler, encoders, path="models/churn_model.pkl"):
    os.makedirs("models", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "encoders": encoders}, f)
    print(f"\nModel saved → {path}")


def main():
    df = load_data("data/customers.csv")
    X, y, scaler, encoders = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    # Baseline
    lr_model, lr_acc = train_baseline(X_train, y_train, X_test, y_test)

    # Random Forest + SMOTE
    rf_model, rf_acc = train_random_forest(X_train, y_train, X_test, y_test)

    print(f"\n── Accuracy Improvement ──")
    print(f"  Baseline  : {lr_acc:.1%}")
    print(f"  RF + SMOTE: {rf_acc:.1%}")
    print(f"  Gain      : +{(rf_acc - lr_acc)*100:.1f} percentage points")

    save_model(rf_model, scaler, encoders)


if __name__ == "__main__":
    main()
