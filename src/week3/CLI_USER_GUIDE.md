# Transfer Learning Framework CLI - User Guide

## Overview

A multifunctional command-line tool for analyzing customer segmentation transfer learning across domains. Supports built-in domain pairs, custom RFM data, and transaction data.

---

## Installation

No additional installation needed! The CLI uses the existing framework components.

```bash
cd src/week3
python cli.py --help
```

---

## Features

✅ **3 Input Modes:**
1. **Built-in Domain Pairs** - Use pre-analyzed grocery domain pairs (1-7)
2. **Custom RFM Files** - Bring your own RFM CSV files  
3. **Transaction Files** - Auto-calculates RFM from transaction history

✅ **Comprehensive Analysis:**
- Transferability score calculation
- Domain similarity metrics
- Strategy recommendation
- Confidence scoring
- Risk assessment

✅ **Beautiful Terminal Output:**
- Color-coded results
- Formatted tables
- Clear recommendations

✅ **Report Generation:**
- Save detailed reports to file
- Includes all metrics and reasoning

---

## Usage Examples

### 1. List Available Built-in Pairs

```bash
python cli.py --list-pairs
```

**Output:**
```
Pair 1: Cleaning & Household → Foodgrains
  Category: HIGH
  Transferability Score: 0.9028

Pair 2: Snacks → Kitchen, Garden & Pets
  Category: LOW
  Transferability Score: 0.7254
...
```

### 2. Analyze Built-in Domain Pair

```bash
# Analyze Pair 7
python cli.py --mode builtin --pair 7

# With report generation
python cli.py --mode builtin --pair 7 --save-report report.txt

# Skip comparison table (faster)
python cli.py --mode builtin --pair 7 --no-comparison
```

**Output:**
```
Strategy: Transfer As Is
Transferability Level: HIGH
Confidence: 99.0%
Target Data Required: 0%

Reasoning:
  Excellent transferability (score: 0.9110)...
```

### 3. Custom RFM Files

```bash
python cli.py --mode rfm \
  --source my_source_rfm.csv \
  --target my_target_rfm.csv
```

**Requirements:**
- CSV files with columns: `Recency`, `Frequency`, `Monetary`
- Optional: `customer_id` column

**Example CSV:**
```csv
customer_id,Recency,Frequency,Monetary
1001,15,8,2500
1002,30,12,5000
...
```

### 4. Transaction Files (Auto-RFM Calculation)

```bash
python cli.py --mode transactions \
  --source source_transactions.csv \
  --target target_transactions.csv
```

**Requirements:**
- Columns detected automatically:
  - Customer ID: `customer_id`, `CustomerID`, `user_id`
  - Date: `InvoiceDate`, `date`, `transaction_date`
  - Amount: `amount`, `total`, `sale_price`

**Example Transaction CSV:**
```csv
customer_id,InvoiceDate,amount
1001,2024-01-15,150
1001,2024-02-20,200
1002,2024-01-10,500
...
```

---

## Command-Line Options

| Option | Description | Example |
|--------|-------------|---------|
| `--help` | Show help message | `python cli.py --help` |
| `--list-pairs` | List all built-in pairs | `python cli.py --list-pairs` |
| `--mode` | Input mode (builtin/rfm/transactions) | `--mode builtin` |
| `--pair` | Built-in pair number (1-7) | `--pair 7` |
| `--source` | Path to source CSV | `--source src.csv` |
| `--target` | Path to target CSV | `--target tgt.csv` |
| `--no-comparison` | Skip domain comparison table | `--no-comparison` |
| `--save-report` | Save report to file | `--save-report out.txt` |

---

## Interpreting Results

### Transferability Score

- **0.90 - 1.00**: HIGH - Domains very similar, transfer works well
- **0.75 - 0.90**: MODERATE-HIGH - Good transfer with light fine-tuning
- **0.60 - 0.75**: MODERATE - Requires moderate fine-tuning
- **< 0.60**: LOW - Train from scratch recommended

### Strategy Recommendations

| Strategy | Data Required | When to Use |
|----------|--------------|-------------|
| **Transfer As-Is** | 0% | High transferability (>0.90) |
| **Fine-Tune Light** | 10-20% | Moderate-high transferability |
| **Fine-Tune Moderate** | 20-40% | Moderate transferability |
| **Fine-Tune Heavy** | 40-60% | Low-moderate transferability |
| **Train From Scratch** | 100% | Very low transferability (<0.60) |

### Confidence Score

- **90-100%**: Very confident recommendation
- **70-90%**: Confident, minor uncertainties
- **50-70%**: Moderate confidence, consider alternatives
- **< 50%**: Low confidence, proceed with caution

---

## Example Workflow

### Scenario: Transfer from "Beverages" to "Snacks"

**Step 1: Check if pair exists**
```bash
python cli.py --list-pairs
```

**Step 2: Run analysis**
```bash
python cli.py --mode builtin --pair 2 --save-report beverages_to_snacks.txt
```

**Step 3: Review recommendation**
```
Strategy: Fine Tune Light
Target Data Required: 10-20%
Confidence: 85%
```

**Step 4: Follow recommendation**
- Collect 10-20% of target domain data (Snacks customers)
- Fine-tune source model (Beverages) with this data
- Expected: 70-85% performance of from-scratch model

---

## Troubleshooting

### Error: "Data files not found"
- **Solution**: Check file paths are correct
- Built-in pairs look in: `src/week2/`
- Use absolute paths for custom files

### Error: "Missing column: Recency"
- **Solution**: Ensure RFM CSV has columns: `Recency`, `Frequency`, `Monetary`
- Column names are **case-sensitive** (capitalize first letter)

### Error: "Could not auto-detect transaction columns"
- **Solution**: Rename columns to standard names:
  - `customer_id` for customer IDs
  - `InvoiceDate` or `date` for dates
  - `amount` or `total` for purchase amounts

### Warning: "Only 1 cluster predicted"
- **Cause**: Target domain too small or very homogeneous
- **Solution**: Collect more diverse target data

---

## Advanced Usage

### Batch Processing

Analyze all 7 pairs and save reports:

**Windows (PowerShell):**
```powershell
foreach ($i in 1..7) {
    python cli.py --mode builtin --pair $i --save-report "report_pair${i}.txt"
}
```

**Linux/Mac:**
```bash
for i in {1..7}; do
    python cli.py --mode builtin --pair $i --save-report "report_pair${i}.txt"
done
```

### Custom Metric Weights

To adjust metric importance, modify `decision_engine.py`:
```python
# Line ~115
metric_weights = {
    'mmd': 0.35,            # Maximum Mean Discrepancy
    'js_divergence': 0.25,  # Jensen-Shannon
    'correlation': 0.20,     # Correlation Stability
    'ks_statistic': 0.10,    # Kolmogorov-Smirnov
    'wasserstein': 0.10      # Wasserstein Distance
}
```

---

## Output Files

### Report File Structure

```
================================================================================
TRANSFER LEARNING FRAMEWORK - ANALYSIS REPORT
================================================================================

Generated: 2024-11-08 18:30:45

Domain Pair: Beverages → Gourmet & World Food
Transferability Score: 0.8951

Transfer Learning Recommendation
======================================================================
Transferability Level: HIGH
Composite Score: 0.9110
Confidence: 99.0%

Recommended Strategy: Transfer As Is
Target Data Required: 0%

Reasoning:
Excellent transferability...

Potential Risks:
  • No major risks identified

Expected Performance:
Expected to maintain 85-95% of source domain performance...
```

---

## Performance Tips

1. **Use `--no-comparison`** for faster analysis (skips domain comparison table)
2. **Pre-calculate RFM** for transaction data (faster on repeated runs)
3. **Use built-in pairs** when possible (models pre-trained)

---

## Support

For issues or questions:
1. Check this guide
2. Review `MEMBER3_GUIDE.md` for framework details
3. Examine example outputs in `src/week3/results/`

---

## Credits

**Member 4 Deliverable** - Week 3-4
- CLI Tool Development
- Multi-mode input support
- Report generation
- User interface design

**Framework by Member 3** - Transfer Learning Core
**Data by Member 1** - Synthetic customer generation  
**Models by Member 2** - Baseline clustering models

---

**Version:** 1.0  
**Last Updated:** November 8, 2024
