import numpy as np

raw_data = np.genfromtxt("data/raw/freshflow_sales.csv", delimiter=",", skip_header=1, ndmin=2)

print("=== TASK A: DATA PROFILE ===")
print(f"Shape: {raw_data.shape} (Expected: (5000, 12))")
print(f"Dtype: {raw_data.dtype}, Memory: {raw_data.nbytes / 1024:.2f} KB")

col_names = [
    "transaction_id", "day_index", "outlet_id", "product_code",
    "category_code", "quantity", "unit_price", "unit_cost",
    "discount_rate", "stock_before_units", "payment_code", "customer_rating"
]

missing_counts = np.isnan(raw_data).sum(axis=0)
for name, count in zip(col_names, missing_counts):
    if count > 0:
        print(f"Missing in {name}: {count}")

print("\n=== TASK B: INTEGRITY AUDIT ===")
tx_id = raw_data[:, 0]
day_idx = raw_data[:, 1]
outlet_id = raw_data[:, 2]
prod_code = raw_data[:, 3]
cat_code = raw_data[:, 4]
qty = raw_data[:, 5]
price = raw_data[:, 6]
cost = raw_data[:, 7]
disc = raw_data[:, 8]
stock = raw_data[:, 9]
pay_code = raw_data[:, 10]
rating = raw_data[:, 11]

unique_ids, counts = np.unique(tx_id[~np.isnan(tx_id)], return_counts=True)
dup_ids = unique_ids[counts > 1]
dup_mask = np.isin(tx_id, dup_ids)
print(f"Duplicate transaction_id records: {dup_mask.sum()} (Unique duplicate IDs: {len(dup_ids)})")

inv_day = np.isnan(day_idx) | (day_idx < 1) | (day_idx > 30) | (day_idx % 1 != 0)
inv_outlet = ~np.isin(outlet_id, [1, 2, 3])
inv_prod = ~np.isin(prod_code, np.arange(1, 9))
inv_pay = ~np.isin(pay_code, [1, 2, 3, 4])

expected_cat = np.select(
    [np.isin(prod_code, [1, 2, 3, 4]), np.isin(prod_code, [5, 6]), prod_code == 7, prod_code == 8],
    [1, 2, 3, 4],
    default=-1
)
inv_cat_mismatch = (~inv_prod) & (cat_code != expected_cat)

inv_qty = np.isnan(qty) | (qty <= 0) | (qty % 1 != 0)
inv_price = np.isnan(price) | (price <= 0)
inv_cost = np.isnan(cost) | (cost < 0)
inv_disc = np.isnan(disc) | (disc < 0) | (disc > 1)
inv_stock = np.isnan(stock) | (stock < 0)
inv_rating = (~np.isnan(rating)) & ((rating < 1) | (rating > 5))

print(f"Invalid day_index: {inv_day.sum()}")
print(f"Unknown outlet_id: {inv_outlet.sum()}")
print(f"Unknown product_code: {inv_prod.sum()}")
print(f"Product/Category mismatch: {inv_cat_mismatch.sum()}")
print(f"Zero/negative quantity: {inv_qty.sum()}")
print(f"Missing or <=0 unit_price: {inv_price.sum()}")
print(f"Missing or <0 unit_cost: {inv_cost.sum()}")
print(f"Discount outside [0, 1]: {inv_disc.sum()}")
print(f"Negative stock_before_units: {inv_stock.sum()}")
print(f"Unknown payment_code: {inv_pay.sum()}")
print(f"Invalid customer_rating (outside 1-5): {inv_rating.sum()}")

fatal_error_mask = (
    dup_mask | inv_day | inv_outlet | inv_prod | inv_cat_mismatch |
    inv_qty | inv_price | inv_cost | inv_disc | inv_stock | inv_pay
)

clean_data = raw_data[~fatal_error_mask].copy()
print("\n=== TASK C: CLEANING SUMMARY ===")
print(f"Total raw rows: {len(raw_data)}")
print(f"Excluded rows (fatal rule violations): {fatal_error_mask.sum()}")
print(f"Clean valid rows retained: {len(clean_data)}")

np.save("data/processed/clean_sales.npy", clean_data)
print("Saved clean array to data/processed/clean_sales.npy")