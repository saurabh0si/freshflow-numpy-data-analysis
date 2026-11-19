
import numpy as np
import pandas as pd

np.random.seed(42)

rows = 5000

products = np.random.randint(1, 9, rows)

category_map = {
    1: 1, 2: 1, 3: 1, 4: 1,
    5: 2, 6: 2,
    7: 3,
    8: 4
}

price_map = {
    1: 120, 2: 140, 3: 110, 4: 100,
    5: 160, 6: 190, 7: 80, 8: 90
}

cost_map = {
    1: 65, 2: 75, 3: 60, 4: 55,
    5: 90, 6: 110, 7: 40, 8: 45
}

data = pd.DataFrame({
    "transaction_id": np.arange(1001, 6001),
    "day_index": np.random.randint(1, 31, rows),
    "outlet_id": np.random.randint(1, 4, rows),
    "product_code": products,
    "category_code": [category_map[p] for p in products],
    "quantity": np.random.randint(1, 16, rows),
    "unit_price": [price_map[p] for p in products],
    "unit_cost": [cost_map[p] for p in products],
    "discount_rate": np.random.choice([0, 0.05, 0.10, 0.15, 0.20], rows),
    "stock_before_units": np.random.randint(5, 150, rows),
    "payment_code": np.random.randint(1, 5, rows),
    "customer_rating": np.random.choice([1, 2, 3, 4, 5], rows).astype(float)
})

data.loc[np.random.choice(rows, 4, replace=False), "unit_price"] = np.nan
data.loc[np.random.choice(rows, 4, replace=False), "unit_cost"] = np.nan
data.loc[np.random.choice(rows, 20, replace=False), "customer_rating"] = np.nan

duplicate_rows = np.random.choice(rows, 5, replace=False)
data.loc[duplicate_rows, "transaction_id"] = data.loc[duplicate_rows - 1, "transaction_id"].values

data.loc[np.random.choice(rows, 4, replace=False), "quantity"] = 0
data.loc[np.random.choice(rows, 2, replace=False), "unit_price"] = -10
data.loc[np.random.choice(rows, 2, replace=False), "unit_cost"] = -5
data.loc[np.random.choice(rows, 2, replace=False), "discount_rate"] = 1.5
data.loc[np.random.choice(rows, 3, replace=False), "customer_rating"] = 6
data.loc[np.random.choice(rows, 4, replace=False), "category_code"] = 4
data.loc[np.random.choice(rows, 2, replace=False), "outlet_id"] = 9
data.loc[np.random.choice(rows, 2, replace=False), "unit_price"] = 999
data.loc[np.random.choice(rows, 2, replace=False), "discount_rate"] = 0.90
data.loc[np.random.choice(rows, 5, replace=False), "stock_before_units"] = 0

data.to_csv("data/raw/freshflow_sales.csv", index=False)

print("Dataset created successfully")
print(data.shape)
print(data.head())
