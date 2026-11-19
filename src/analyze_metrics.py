import numpy as np

clean_data = np.load("data/processed/clean_sales.npy")

tx_id = clean_data[:, 0].astype(int)
day_idx = clean_data[:, 1].astype(int)
outlet_id = clean_data[:, 2].astype(int)
prod_code = clean_data[:, 3].astype(int)
cat_code = clean_data[:, 4].astype(int)
qty = clean_data[:, 5]
price = clean_data[:, 6]
cost = clean_data[:, 7]
disc = clean_data[:, 8]
stock = clean_data[:, 9]
pay_code = clean_data[:, 10].astype(int)
rating = clean_data[:, 11]

gross_sales = qty * price
discount_amount = gross_sales * disc
net_revenue = gross_sales - discount_amount
cogs = qty * cost
gross_profit = net_revenue - cogs

gross_margin_pct = np.divide(
    gross_profit * 100.0,
    net_revenue,
    out=np.full_like(net_revenue, np.nan, dtype=float),
    where=(net_revenue > 0)
)
avg_selling_price = np.divide(
    net_revenue,
    qty,
    out=np.full_like(net_revenue, np.nan, dtype=float),
    where=(qty > 0)
)

total_net_rev = np.sum(net_revenue)
total_gross_profit = np.sum(gross_profit)
overall_margin_pct = (total_gross_profit / total_net_rev) * 100
weighted_disc_pct = (np.sum(discount_amount) / np.sum(gross_sales)) * 100

print("=== TASK D: OVERALL METRICS ===")
print(f"Total Gross Sales:   INR {np.sum(gross_sales):,.2f}")
print(f"Total Discounts:     INR {np.sum(discount_amount):,.2f} (Weighted Disc: {weighted_disc_pct:.2f}%)")
print(f"Total Net Revenue:   INR {total_net_rev:,.2f}")
print(f"Total COGS:          INR {np.sum(cogs):,.2f}")
print(f"Total Gross Profit:  INR {total_gross_profit:,.2f}")
print(f"Overall Gross Margin: {overall_margin_pct:.2f}%")

print("\n=== TASK E1: OUTLET RANKING (BY NET REVENUE) ===")
outlet_names = {1: "Outlet A", 2: "Outlet B", 3: "Outlet C"}
outlet_stats = []
for oid in [1, 2, 3]:
    mask = (outlet_id == oid)
    rev = np.sum(net_revenue[mask])
    prof = np.sum(gross_profit[mask])
    units = np.sum(qty[mask])
    margin = (prof / rev) * 100
    outlet_stats.append((oid, rev, prof, units, margin))

outlet_stats.sort(key=lambda x: x[1], reverse=True)
for oid, rev, prof, units, margin in outlet_stats:
    print(f"{outlet_names[oid]} (ID {oid}): Net Rev = INR {rev:,.2f} | Profit = INR {prof:,.2f} | Margin = {margin:.2f}% | Units = {int(units)}")

print("\n=== TASK E2: PRODUCT RANKING (BY GROSS PROFIT) ===")
prod_names = {
    1: "Orange Juice", 2: "Mango Juice", 3: "Apple Juice", 4: "Watermelon Juice",
    5: "Banana Smoothie", 6: "Mixed Fruit Smoothie", 7: "Lemonade", 8: "Iced Tea"
}
prod_stats = []
for pid in range(1, 9):
    mask = (prod_code == pid)
    rev = np.sum(net_revenue[mask])
    prof = np.sum(gross_profit[mask])
    units = np.sum(qty[mask])
    margin = (prof / rev) * 100
    prod_stats.append((pid, rev, prof, units, margin))

prod_stats.sort(key=lambda x: x[2], reverse=True)
for pid, rev, prof, units, margin in prod_stats:
    print(f"P{pid} ({prod_names[pid]:<20}): Profit = INR {prof:,.2f} | Net Rev = INR {rev:,.2f} | Units = {int(units):4d} | Margin = {margin:.2f}%")

print("\n=== TASK F: DESCRIPTIVE STATISTICS (TRANSACTION LEVEL) ===")
for name, arr in [("Net Revenue (INR)", net_revenue), ("Gross Profit (INR)", gross_profit), ("Quantity (Units)", qty)]:
    q25, med, q75 = np.percentile(arr, [25, 50, 75])
    print(f"{name:<20} -> Mean: {np.mean(arr):8.2f} | Median: {med:8.2f} | Std: {np.std(arr, ddof=1):8.2f} | IQR: {q75 - q25:8.2f} (Q1={q25:.2f}, Q3={q75:.2f})")

valid_ratings = rating[(~np.isnan(rating)) & (rating >= 1) & (rating <= 5)]
print(f"Customer Rating      -> Mean: {np.mean(valid_ratings):.2f} | Median: {np.median(valid_ratings):.2f} | Valid Count: {len(valid_ratings)}")

print("\n=== TASK G: DAILY TREND & UNUSUAL DAYS ===")
days = np.arange(1, 31)
daily_rev = np.array([np.sum(net_revenue[day_idx == d]) for d in days])
daily_diff = np.diff(daily_rev)
daily_pct_change = np.divide(
    daily_diff * 100.0,
    daily_rev[:-1],
    out=np.full_like(daily_diff, np.nan, dtype=float),
    where=(daily_rev[:-1] > 0)
)
rolling_3d = np.convolve(daily_rev, np.ones(3) / 3, mode="valid")

mean_daily = np.mean(daily_rev)
std_daily = np.std(daily_rev, ddof=1)
print(f"Average Daily Net Revenue: INR {mean_daily:,.2f} (Std Dev: INR {std_daily:,.2f})")
print(f"Highest Sales Day: Day {days[np.argmax(daily_rev)]} (INR {np.max(daily_rev):,.2f})")
print(f"Lowest Sales Day:  Day {days[np.argmin(daily_rev)]} (INR {np.min(daily_rev):,.2f})")

unusual_days = days[np.abs(daily_rev - mean_daily) > 1.5 * std_daily]
print(f"Unusual Days (>1.5 SD from mean): {unusual_days.tolist()}")

print("\n=== TASK H: DISCOUNT INVESTIGATION ===")
bands = [
    ("0% (No Disc)", disc == 0),
    ("1% - 10%", (disc > 0) & (disc <= 0.10)),
    ("11% - 20%", (disc > 0.10) & (disc <= 0.20)),
    (">20% (Extreme)", disc > 0.20)
]
for label, b_mask in bands:
    cnt = np.sum(b_mask)
    avg_q = np.mean(qty[b_mask]) if cnt > 0 else 0
    rev = np.sum(net_revenue[b_mask])
    prof = np.sum(gross_profit[b_mask])
    marg = (prof / rev * 100) if rev > 0 else 0
    print(f"{label:<15}: Txns = {cnt:4d} | Avg Qty = {avg_q:5.2f} | Net Rev = INR {rev:10,.2f} | Gross Margin = {marg:6.2f}%")

corr_disc_qty = np.corrcoef(disc, qty)[0, 1]
corr_disc_margin = np.corrcoef(disc, gross_margin_pct)[0, 1]
print(f"Correlation (Discount vs Quantity): {corr_disc_qty:.4f}")
print(f"Correlation (Discount vs Margin %): {corr_disc_margin:.4f}")

print("\n=== TASK I: ANOMALY REGISTER (SUSPICIOUS RECORDS IN CLEAN DATA) ===")
high_price_mask = price > 500
high_disc_mask = disc > 0.50
zero_stock_mask = stock == 0

print(f"Unusually high unit_price (>500): {high_price_mask.sum()} records -> Tx IDs: {tx_id[high_price_mask].tolist()}")
print(f"Unusually high discount (>50%):   {high_disc_mask.sum()} records -> Tx IDs: {tx_id[high_disc_mask].tolist()}")
print(f"Zero stock before sale (stock=0): {zero_stock_mask.sum()} records -> Tx IDs: {tx_id[zero_stock_mask].tolist()}")

print("\n=== TASK J: DISCOUNT SCENARIO ANALYSIS (STATIC DEMAND ASSUMPTION) ===")
disc_s1 = np.minimum(disc, 0.10)
rev_s1 = np.sum(gross_sales * (1 - disc_s1))
prof_s1 = rev_s1 - np.sum(cogs)

disc_s2 = np.full_like(disc, 0.05)
rev_s2 = np.sum(gross_sales * (1 - disc_s2))
prof_s2 = rev_s2 - np.sum(cogs)

print(f"Baseline             -> Net Rev: INR {total_net_rev:11,.2f} | Gross Profit: INR {total_gross_profit:11,.2f}")
print(f"Scenario 1 (Cap 10%) -> Net Rev: INR {rev_s1:11,.2f} ({rev_s1 - total_net_rev:+,.2f}) | Gross Profit: INR {prof_s1:11,.2f} ({prof_s1 - total_gross_profit:+,.2f})")
print(f"Scenario 2 (Flat 5%) -> Net Rev: INR {rev_s2:11,.2f} ({rev_s2 - total_net_rev:+,.2f}) | Gross Profit: INR {prof_s2:11,.2f} ({prof_s2 - total_gross_profit:+,.2f})")