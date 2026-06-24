"""
Generates a realistic sample transactions.csv for testing the
Spend Insights Assistant RAG pipeline.

Run: python generate_sample_data.py
Output: transactions.csv (about 1,200 rows, 6 months of data, 50 customers)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_CUSTOMERS = 50
N_TRANSACTIONS = 1200
START_DATE = datetime(2025, 9, 1)
END_DATE = datetime(2026, 2, 28)

MERCHANT_CATEGORIES = [
    "Groceries", "Travel", "Dining", "Electronics", "Fuel",
    "Healthcare", "Entertainment", "Utilities", "Apparel", "Online Shopping"
]

# Give each category a realistic spend range so the data tells a believable story
CATEGORY_AMOUNT_RANGES = {
    "Groceries":        (15, 180),
    "Travel":           (80, 2200),
    "Dining":           (10, 150),
    "Electronics":      (50, 1800),
    "Fuel":             (20, 90),
    "Healthcare":       (25, 600),
    "Entertainment":    (10, 200),
    "Utilities":        (40, 250),
    "Apparel":          (20, 400),
    "Online Shopping":  (10, 500),
}

# Make Travel and Online Shopping trend upward over the period (a "trend" for the
# assistant to discover) and Electronics spike in December (holiday season)
def seasonal_multiplier(category, txn_date):
    month_progress = (txn_date - START_DATE).days / (END_DATE - START_DATE).days
    if category == "Travel":
        return 1.0 + month_progress * 0.8          # steady increase
    if category == "Online Shopping":
        return 1.0 + month_progress * 0.5
    if category == "Electronics" and txn_date.month == 12:
        return 2.2                                  # holiday spike
    return 1.0

# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------
customer_ids = [f"CUST{1000 + i}" for i in range(N_CUSTOMERS)]
date_range_days = (END_DATE - START_DATE).days

rows = []
for i in range(N_TRANSACTIONS):
    txn_date = START_DATE + timedelta(days=int(np.random.uniform(0, date_range_days)))
    category = np.random.choice(MERCHANT_CATEGORIES, p=[
        0.18, 0.10, 0.15, 0.08, 0.12, 0.07, 0.08, 0.09, 0.08, 0.05
    ])
    low, high = CATEGORY_AMOUNT_RANGES[category]
    base_amount = np.random.uniform(low, high)
    amount = round(base_amount * seasonal_multiplier(category, txn_date), 2)
    customer_id = np.random.choice(customer_ids)

    rows.append({
        "transaction_date": txn_date.strftime("%Y-%m-%d"),
        "customer_id": customer_id,
        "merchant_category": category,
        "amount": amount
    })

df = pd.DataFrame(rows).sort_values("transaction_date").reset_index(drop=True)
df.to_csv("transactions.csv", index=False)

print(f"Generated {len(df)} transactions across {N_CUSTOMERS} customers.")
print(f"Date range: {df['transaction_date'].min()} to {df['transaction_date'].max()}")
print(f"Categories: {df['merchant_category'].nunique()}")
print("\nSample rows:")
print(df.head(8).to_string(index=False))
print("\nSaved to transactions.csv")
