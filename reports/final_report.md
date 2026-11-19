# FreshFlow Juice Company — NumPy Data Analysis

End-to-end numeric CSV data inspection, cleaning, statistical analysis, anomaly detection, and discount scenario modeling built using Python and NumPy.

## Project Structure

- `data/raw/freshflow_sales.csv` — 5,000 raw transaction records
- `data/processed/clean_sales.npy` — 4,966 validated records saved as a NumPy array
- `src/generate_data.py` — Synthetic dataset generator
- `src/audit_and_clean.py` — NumPy CSV loader, data audit, and boolean masking
- `src/analyze_metrics.py` — Metrics, rankings, trends, anomalies, and discount scenarios
- `tests/test_validation.py` — Manual calculation assertions and edge-case tests
- `reports/final_report.md` — 12-section executive report

## How to Run

```bash
source .venv/bin/activate
python src/audit_and_clean.py
python src/analyze_metrics.py
python tests/test_validation.py