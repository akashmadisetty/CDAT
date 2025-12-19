# E-commerce Dataset Integration Guide (UPDATED)

This guide shows the recommended steps to convert your e-commerce CSV into RFM, run the transfer-learning CLI, and generate clean, machine-readable reports. It also documents the quick runner behavior: the runner can now accept an already-converted RFM CSV and will skip conversion when the filename contains `_RFM`.

## 📊 Dataset → RFM mapping

Keep an RFM file with these required columns:

- `customer_id` (string)
- `Recency` (numeric, days since last purchase — lower is better)
- `Frequency` (numeric, number of purchases/items — higher is better)
- `Monetary` (numeric, total spend — higher is better)

You can keep optional metadata columns (these are preserved by the converter): `gender`, `age`, `city`, `membership_type`, `avg_rating`, `discount_applied`, `satisfaction`.


## 🚀 Quick Start (recommended)

There are two main ways to run the analysis.

Option A — One-command end-to-end (recommended if you want automation):

1. If you have the raw CSV and want the full pipeline (convert → create pairs → analyze):

```powershell
python .\quick_ecommerce_analysis.py "E-commerce Customer Behavior - Sheet1.csv"
```

2. If you already converted the dataset and have the RFM CSV (filename contains `_RFM`), call the quick runner which will skip conversion and run analysis on the existing domain-pair files in `data/ecommerce/`:

```powershell
python .\quick_ecommerce_analysis.py "E-commerce Customer Behavior - Sheet1_RFM.csv"
```

What the quick runner does now:
- If input filename contains `_RFM`, it will skip calling the converter and assume `data/ecommerce/` already contains the pair CSVs.
- For each domain pair it calls the CLI with `--save-report <path>` so the CLI itself writes a clean, UTF-8 report (no ANSI/color escapes).
- It then parses those clean reports and writes `results/ecommerce/SUMMARY_REPORT.txt` (human-readable summary).


Option B — Manual (fine-grained control)

If you prefer to run conversion and CLI yourself, run these steps manually.

Step 1 — Convert (optional):

```powershell
# Convert full dataset to RFM
python convert_ecommerce_to_rfm.py "E-commerce Customer Behavior - Sheet1.csv"

# Or create the four recommended domain pairs automatically
python convert_ecommerce_to_rfm.py "E-commerce Customer Behavior - Sheet1.csv" --create-pairs
```

The `--create-pairs` option saves files into `data/ecommerce/`:
- `pair1_gold_source_RFM.csv`
- `pair1_silver_target_RFM.csv`
- `pair2_satisfied_source_RFM.csv`
- `pair2_neutral_target_RFM.csv`
- `pair3_newyork_source_RFM.csv`
- `pair3_losangeles_target_RFM.csv`
- `pair4_highspend_source_RFM.csv`
- `pair4_lowspend_target_RFM.csv`

Step 2 — Run CLI per pair (use `--save-report` to get a clean report file):

```powershell
# Pair 1: Gold -> Silver
python .\src\week3\cli.py --mode rfm --source data\ecommerce\pair1_gold_source_RFM.csv --target data\ecommerce\pair1_silver_target_RFM.csv --save-report results\ecommerce\pair1_clean_report.txt

# Pair 2: Satisfied -> Neutral
python .\src\week3\cli.py --mode rfm --source data\ecommerce\pair2_satisfied_source_RFM.csv --target data\ecommerce\pair2_neutral_target_RFM.csv --save-report results\ecommerce\pair2_clean_report.txt

# Pair 3: New York -> Los Angeles
python .\src\week3\cli.py --mode rfm --source data\ecommerce\pair3_newyork_source_RFM.csv --target data\ecommerce\pair3_losangeles_target_RFM.csv --save-report results\ecommerce\pair3_clean_report.txt

# Pair 4: High Spend -> Low Spend
python .\src\week3\cli.py --mode rfm --source data\ecommerce\pair4_highspend_source_RFM.csv --target data\ecommerce\pair4_lowspend_target_RFM.csv --save-report results\ecommerce\pair4_clean_report.txt
```

Step 3 — Review results

- The CLI `--save-report` output files are saved under `results/ecommerce/` and are pure UTF-8 text files without ANSI escape sequences. Example: `results/ecommerce/pair1_clean_report.txt`.
- If you used the quick runner, `results/ecommerce/SUMMARY_REPORT.txt` will be generated automatically. If you ran the CLI manually, see the optional aggregator section below.

---

## 🔧 Optional: Remove the quick runner?

Yes — the `quick_ecommerce_analysis.py` is purely a convenience script. If you delete it, make sure to:

- Keep the RFM pair files in `data/ecommerce/` (or create them manually with `convert_ecommerce_to_rfm.py`).
- Call the CLI per pair with `--save-report` to get clean reports.
- Optionally add a small aggregator (I can add `aggregate_reports.py`) that reads the clean reports and writes `SUMMARY_REPORT.txt` in the same format the runner produces.

If you want, I can remove the quick runner for you (git rm + commit) or add the lightweight aggregator. Tell me which.

---

## 🧭 Recommended Domain Pairs (brief)

- Membership: Gold → Silver — expected MODERATE to HIGH
- Satisfaction: Satisfied → Neutral — expected MODERATE
- Geography: New York → Los Angeles — expected HIGH
- Spend tiers: High → Low — expected MODERATE to LOW

These are suggestions; actual transferability depends on the data distributions and sample sizes.

---

## ⚠️ Important notes & troubleshooting

1. Encoding on Windows: the CLI and runner set stdout/stderr to UTF-8 where possible. Use `--save-report` to avoid console-encoding artifacts.
2. Column names: the converter strips whitespace from headers. If your column names differ, either rename them or run the converter with a short script that maps your columns.
3. Missing RFM values: the converter drops rows missing Recency/Frequency/Monetary. Check `convert_ecommerce_to_rfm.py` logs if you lose many rows.
4. Small sample sizes (<100) reduce confidence — treat composite scores with caution.

---

## 📚 Next steps I can help with

- Add `aggregate_reports.py` to produce `SUMMARY_REPORT.txt` from existing clean reports (recommended if you delete the quick runner).
- Add a small README snippet listing the exact commands (I can commit it).
- Wire a simple unit test that validates conversion creates the `_RFM.csv` file and that CLI `--save-report` writes a file.

Tell me which of the above you'd like me to do next and I'll implement it.
