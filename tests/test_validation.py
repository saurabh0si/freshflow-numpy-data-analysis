import numpy as np

def calculate_metrics(data_array):
    qty = data_array[:, 5]
    price = data_array[:, 6]
    cost = data_array[:, 7]
    disc = data_array[:, 8]

    gross_sales = qty * price
    discount_amt = gross_sales * disc
    net_rev = gross_sales - discount_amt
    cogs = qty * cost
    gross_profit = net_rev - cogs
    margin_pct = np.divide(
        gross_profit * 100.0,
        net_rev,
        out=np.full_like(net_rev, np.nan, dtype=float),
        where=(net_rev > 0)
    )
    return gross_sales, discount_amt, net_rev, cogs, gross_profit, margin_pct

def run_tests():
    clean_data = np.load("data/processed/clean_sales.npy")
    assert clean_data.ndim == 2
    assert clean_data.shape[1] == 12

    sample_5 = clean_data[:5]
    gs, da, nr, cogs, gp, mp = calculate_metrics(sample_5)

    expected_nr = []
    expected_gp = []
    for row in sample_5:
        q, p, c, d = row[5], row[6], row[7], row[8]
        manual_net = (q * p) * (1.0 - d)
        manual_prof = manual_net - (q * c)
        expected_nr.append(manual_net)
        expected_gp.append(manual_prof)

    np.testing.assert_allclose(nr, expected_nr, rtol=1e-7)
    np.testing.assert_allclose(gp, expected_gp, rtol=1e-7)
    print("PASS: 5 sample rows match independent manual calculations!")

    edge_cases = np.array([
        [9001, 1, 1, 1, 1, 5, np.nan, 50.0, 0.10, 20, 1, 4.0],
        [9002, 1, 1, 1, 1, 5, 100.0, 50.0, 1.00, 20, 1, 4.0]
    ], dtype=float)

    e_gs, e_da, e_nr, e_cogs, e_gp, e_mp = calculate_metrics(edge_cases)
    assert np.isnan(e_nr[0]) and np.isnan(e_gp[0])
    assert e_nr[1] == 0.0 and np.isnan(e_mp[1])
    print("PASS: NaN propagation and zero-denominator edge cases handled safely!")

    single_row = np.genfromtxt("data/raw/freshflow_sales.csv", delimiter=",", skip_header=1, max_rows=1, ndmin=2)
    assert single_row.shape == (1, 12)
    print("PASS: Single-row CSV loading preserves 2D shape (1, 12)!")

if __name__ == "__main__":
    run_tests()