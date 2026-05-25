"""
generate_data.py
Generates a synthetic customer churn dataset (50,000 records) for demo purposes.
Run this once before training: python src/generate_data.py
"""

import pandas as pd
import numpy as np
import os

np.random.seed(42)
N = 50000

def generate_churn_data(n=N):
    customer_id     = [f"CUST_{i:06d}" for i in range(1, n+1)]
    age             = np.random.randint(18, 75, n)
    tenure_months   = np.random.randint(1, 120, n)
    monthly_charges = np.round(np.random.uniform(20, 150, n), 2)
    total_charges   = np.round(monthly_charges * tenure_months * np.random.uniform(0.85, 1.0, n), 2)
    num_products    = np.random.randint(1, 6, n)
    support_calls   = np.random.randint(0, 15, n)
    payment_delay   = np.random.randint(0, 30, n)
    contract_type   = np.random.choice(["Month-to-Month", "One Year", "Two Year"], n, p=[0.55, 0.25, 0.20])
    internet_service= np.random.choice(["DSL", "Fiber Optic", "No"], n, p=[0.35, 0.45, 0.20])
    gender          = np.random.choice(["Male", "Female"], n)
    senior_citizen  = np.random.choice([0, 1], n, p=[0.84, 0.16])

    # Churn probability driven by business-realistic features
    churn_prob = (
        0.05
        + 0.20 * (contract_type == "Month-to-Month")
        + 0.15 * (support_calls > 5)
        + 0.10 * (payment_delay > 15)
        - 0.10 * (tenure_months > 60)
        - 0.08 * (num_products > 3)
        + 0.05 * (monthly_charges > 100)
    )
    churn_prob = np.clip(churn_prob, 0.02, 0.85)
    churn      = (np.random.uniform(0, 1, n) < churn_prob).astype(int)

    df = pd.DataFrame({
        "customer_id"     : customer_id,
        "age"             : age,
        "gender"          : gender,
        "senior_citizen"  : senior_citizen,
        "tenure_months"   : tenure_months,
        "contract_type"   : contract_type,
        "internet_service": internet_service,
        "num_products"    : num_products,
        "monthly_charges" : monthly_charges,
        "total_charges"   : total_charges,
        "support_calls"   : support_calls,
        "payment_delay"   : payment_delay,
        "churn"           : churn
    })
    return df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    df = generate_churn_data()
    df.to_csv("data/customers.csv", index=False)
    print(f"Generated {len(df):,} records → data/customers.csv")
    print(f"Churn rate: {df['churn'].mean():.1%}")
