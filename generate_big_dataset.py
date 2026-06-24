"""
Generates a large, clean, realistic credit card transactions dataset —
designed so that almost any natural-language question asked of the
Spend Insights chatbot has rich, real patterns behind it (trends,
seasonality, customer behavior segments, weekday effects, named merchants).

Run: python generate_big_dataset.py
Output: transactions_dashboard.csv  (~8,000 rows, 1 year, 200 customers)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(7)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
N_CUSTOMERS = 200
START_DATE = datetime(2025, 1, 1)
END_DATE = datetime(2025, 12, 31)
TOTAL_DAYS = (END_DATE - START_DATE).days

# Real-sounding merchant names grouped under each category (adds realism —
# a chatbot/dashboard answering "top merchant" gets a real name, not a category label)
MERCHANTS = {
    "Groceries": ["Whole Foods Market", "Trader Joe's", "Kroger", "Safeway", "Costco Wholesale"],
    "Travel": ["Delta Air Lines", "Marriott Hotels", "Expedia", "Airbnb", "United Airlines"],
    "Dining": ["Chipotle", "Starbucks", "Olive Garden", "Cheesecake Factory", "Local Bistro Co"],
    "Electronics": ["Best Buy", "Apple Store", "Amazon Electronics", "Samsung Store", "Micro Center"],
    "Fuel": ["Shell", "Chevron", "ExxonMobil", "BP", "Costco Gas"],
    "Healthcare": ["CVS Pharmacy", "Walgreens", "Kaiser Permanente", "Quest Diagnostics", "One Medical"],
    "Entertainment": ["Netflix", "AMC Theatres", "Spotify", "Live Nation", "Steam"],
    "Utilities": ["Pacific Gas & Electric", "Comcast Xfinity", "AT&T", "Verizon", "Waste Management"],
    "Apparel": ["Nike", "Zara", "Nordstrom", "H&M", "Lululemon"],
    "Online Shopping": ["Amazon", "Target.com", "Etsy", "Walmart.com", "eBay"],
}
CATEGORIES = list(MERCHANTS.keys())

CATEGORY_AMOUNT_RANGES = {
    "Groceries":        (15, 180),
    "Travel":           (90, 2400),
    "Dining":           (8, 140),
    "Electronics":      (40, 2000),
    "Fuel":             (20, 95),
    "Healthcare":       (20, 650),
    "Entertainment":    (8, 220),
    "Utilities":        (35, 260),
    "Apparel":          (18, 420),
    "Online Shopping":  (10, 550),
}
CATEGORY_WEIGHTS = [0.16, 0.11, 0.16, 0.09, 0.11, 0.07, 0.08, 0.08, 0.08, 0.06]

CITIES = [
    ("New York", "NY"), ("Los Angeles", "CA"), ("Chicago", "IL"), ("Houston", "TX"),
    ("Phoenix", "AZ"), ("Austin", "TX"), ("Seattle", "WA"), ("Denver", "CO"),
    ("Atlanta", "GA"), ("Boston", "MA"), ("Miami", "FL"), ("San Francisco", "CA"),
]
CARD_TYPES = ["Platinum", "Gold", "Green", "Business Green"]
CHANNELS = ["In-Store", "Online", "Mobile App"]

# ---------------------------------------------------------------------------
# Customer profiles — each customer gets a realistic, persistent behavior
# pattern rather than every transaction being pure independent noise
# ---------------------------------------------------------------------------
customers = []
for i in range(N_CUSTOMERS):
    customer_id = f"CUST{2000 + i}"
    # Behavior archetype shapes how often and how much this customer spends
    archetype = np.random.choice(
        ["frequent_high", "frequent_moderate", "occasional", "one_time_high", "churned_mid_year"],
        p=[0.12, 0.38, 0.30, 0.08, 0.12]
    )
    city, state = CITIES[np.random.randint(len(CITIES))]
    card_type = np.random.choice(CARD_TYPES, p=[0.2, 0.35, 0.35, 0.10])
    join_offset = np.random.randint(0, 15)  # nearly all customers active from the start
    customers.append({
        "customer_id": customer_id,
        "archetype": archetype,
        "city": city,
        "state": state,
        "card_type": card_type,
        "join_day": join_offset,
    })

ARCHETYPE_MONTHLY_TXNS = {
    "frequent_high": (10, 16),
    "frequent_moderate": (5, 9),
    "occasional": (1, 4),
    "one_time_high": (1, 2),
    "churned_mid_year": (4, 8),   # active first half, then stops
}
ARCHETYPE_SPEND_MULTIPLIER = {
    "frequent_high": 1.6,
    "frequent_moderate": 1.0,
    "occasional": 0.7,
    "one_time_high": 2.8,
    "churned_mid_year": 1.0,
}


def seasonal_multiplier(category, txn_date):
    """Adds believable seasonality: holiday electronics/shopping spike,
    summer travel bump, January gym/health bump, steady online growth."""
    month = txn_date.month
    m = 1.0
    if category == "Electronics" and month == 12:
        m *= 2.3
    if category == "Online Shopping" and month in (11, 12):
        m *= 1.9
    if category == "Travel" and month in (6, 7, 8):
        m *= 1.6
    if category == "Healthcare" and month == 1:
        m *= 1.3
    if category == "Online Shopping":
        # gentle upward trend across the whole year
        m *= 1.0 + (txn_date - START_DATE).days / TOTAL_DAYS * 0.4
    return m


def weekday_multiplier(category, weekday):
    """Dining/Entertainment skew toward weekends; Fuel/Groceries are steadier."""
    is_weekend = weekday in (5, 6)
    if category in ("Dining", "Entertainment") and is_weekend:
        return 1.4
    if category == "Groceries" and weekday in (5, 6):
        return 1.2
    return 1.0


# ---------------------------------------------------------------------------
# Generate transactions per customer based on their archetype
# ---------------------------------------------------------------------------
rows = []
txn_counter = 1

for cust in customers:
    archetype = cust["archetype"]
    min_txn, max_txn = ARCHETYPE_MONTHLY_TXNS[archetype]
    spend_mult = ARCHETYPE_SPEND_MULTIPLIER[archetype]

    active_start = START_DATE + timedelta(days=int(cust["join_day"]))
    active_end = END_DATE
    if archetype == "churned_mid_year":
        # stop being active partway through the year
        churn_day = np.random.randint(120, 280)
        active_end = START_DATE + timedelta(days=int(churn_day))

    months_active = max(1, (active_end - active_start).days // 30)

    for _ in range(months_active):
        n_txns_this_month = np.random.randint(min_txn, max_txn + 1)
        for _ in range(n_txns_this_month):
            day_offset = np.random.uniform(0, (active_end - active_start).days or 1)
            txn_date = active_start + timedelta(days=day_offset)
            if txn_date > END_DATE:
                continue

            category = np.random.choice(CATEGORIES, p=CATEGORY_WEIGHTS)
            merchant = np.random.choice(MERCHANTS[category])
            low, high = CATEGORY_AMOUNT_RANGES[category]
            base_amount = np.random.uniform(low, high)

            amount = (
                base_amount
                * spend_mult
                * seasonal_multiplier(category, txn_date)
                * weekday_multiplier(category, txn_date.weekday())
            )
            amount = round(amount, 2)

            rows.append({
                "transaction_id": f"TXN{100000 + txn_counter}",
                "transaction_date": txn_date.strftime("%Y-%m-%d"),
                "customer_id": cust["customer_id"],
                "merchant_name": merchant,
                "merchant_category": category,
                "amount": amount,
                "city": cust["city"],
                "state": cust["state"],
                "card_type": cust["card_type"],
                "channel": np.random.choice(CHANNELS, p=[0.45, 0.40, 0.15]),
            })
            txn_counter += 1

df = pd.DataFrame(rows).sort_values("transaction_date").reset_index(drop=True)

# ---------------------------------------------------------------------------
# Add calculated/dashboard fields (same enrichment as before, kept consistent)
# ---------------------------------------------------------------------------
df["transaction_date"] = pd.to_datetime(df["transaction_date"])
df["month_name"] = df["transaction_date"].dt.strftime("%b %Y")
df["month_sort"] = df["transaction_date"].dt.strftime("%Y-%m")
df["quarter"] = df["transaction_date"].dt.to_period("Q").astype(str)
df["weekday"] = df["transaction_date"].dt.day_name()
df["is_weekend"] = df["transaction_date"].dt.weekday >= 5
df["is_holiday_season"] = df["transaction_date"].dt.month.isin([11, 12])

customer_totals = df.groupby("customer_id")["amount"].sum().reset_index()
customer_totals.columns = ["customer_id", "customer_total_spend"]
q1, q2 = customer_totals["customer_total_spend"].quantile([0.33, 0.66])

def segment(spend):
    if spend >= q2:
        return "High Value"
    elif spend >= q1:
        return "Medium Value"
    return "Low Value"

customer_totals["customer_segment"] = customer_totals["customer_total_spend"].apply(segment)
df = df.merge(customer_totals, on="customer_id", how="left")

df["amount"] = df["amount"].round(2)
df["customer_total_spend"] = df["customer_total_spend"].round(2)
df["transaction_date"] = df["transaction_date"].dt.strftime("%Y-%m-%d")

# Reorder columns sensibly
df = df[[
    "transaction_id", "transaction_date", "month_name", "month_sort", "quarter",
    "weekday", "is_weekend", "is_holiday_season",
    "customer_id", "customer_segment", "customer_total_spend", "card_type",
    "merchant_name", "merchant_category", "amount", "city", "state", "channel",
]]

df.to_csv("transactions_dashboard.csv", index=False)

print(f"Generated {len(df):,} transactions across {df['customer_id'].nunique()} customers.")
print(f"Date range: {df['transaction_date'].min()} to {df['transaction_date'].max()}")
print(f"Categories: {df['merchant_category'].nunique()}, Merchants: {df['merchant_name'].nunique()}")
print(f"Cities: {df['city'].nunique()}")
print(f"\nSegment distribution:\n{df.groupby('customer_segment')['customer_id'].nunique()}")
print(f"\nTotal spend: ${df['amount'].sum():,.2f}")
print(f"\nSample rows:")
print(df.head(5).to_string(index=False))
