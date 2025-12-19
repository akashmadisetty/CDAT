# UK Retail Framework Validation - Quick Start Guide

## 📋 What This Does

Validates the Transfer Learning Framework on **real UK Retail transaction data** and generates comprehensive insights.

**Deliverables:**
- ✅ UK Retail experiments results (3 experiments)
- ✅ Validation report: Framework accuracy on new dataset
- ✅ Statistical analysis: Correlation between predictions & reality
- ✅ Raw vs Scaled feature comparison
- ✅ Framework accuracy metrics

---

## 🚀 Step-by-Step Execution

### **Step 1: Generate UK Retail RFM Data** (if not done yet)

```bash
cd src\week5_6
python uk_rfm_generator_FIXED.py
```

**Expected output:** 13 files including:
- `exp5_uk_source_RFM_scaled.csv`
- `exp5_france_target_RFM_scaled.csv`
- `exp6_uk_source_RFM_scaled.csv`
- `exp6_germany_target_RFM_scaled.csv`
- `exp7_highvalue_source_RFM_scaled.csv`
- `exp7_mediumvalue_target_RFM_scaled.csv`

**Time:** ~5-10 minutes

---

### **Step 2: Validate RFM Fixes** (recommended)

```bash
python validate_rfm_fixes.py
```

**Expected output:** ✅ ALL VALIDATION CHECKS PASSED!

**Time:** ~2 minutes

---

### **Step 3: Run Framework Validation** (main analysis)

```bash
python validate_framework_on_uk_retail.py
```

**What it does:**
1. ✅ Loads all 3 UK Retail experiments
2. ✅ Calculates transferability scores (MMD, JS Divergence, Correlation)
3. ✅ Compares Raw vs Scaled features
4. ✅ Generates recommendations for each experiment
5. ✅ Analyzes framework accuracy
6. ✅ Creates comprehensive validation report

**Expected output:**
```
╔═══════════════════════════════════════════════════════════════════════════════╗
║           UK RETAIL DATASET VALIDATION - TRANSFER LEARNING FRAMEWORK          ║
║                    Week 5-6: Real Transaction Data Testing                    ║
╚═══════════════════════════════════════════════════════════════════════════════╝

🔍 Checking for UK Retail RFM files...
✅ All RFM files found!

################################################################################
# EXPERIMENT 5: UK → France
################################################################################
...
```

**Time:** ~3-5 minutes

---

## 📊 Output Files

All results saved to `src/week5_6/validation_results/`:

| File | Description |
|------|-------------|
| `uk_retail_validation_results.csv` | Detailed metrics for all experiments |
| `framework_validation_report.md` | Comprehensive validation report (markdown) |

---

## 🎯 What You'll Learn

### **1. Experiment Results**

For each experiment (5, 6, 7):
- Transferability score (0-1)
- MMD, JS Divergence, Correlation Stability metrics
- Recommended strategy (Transfer As-Is / Fine-tune / Train New)
- Confidence level (%)
- Target data required (%)

### **2. Raw vs Scaled Comparison**

**Example output:**
```
COMPARISON: Raw vs Scaled Features
================================================================================

Composite Score:
  Raw:    0.6234
  Scaled: 0.7812
  Δ:      +0.1578

MMD Score (lower = more similar):
  Raw:    342.56
  Scaled: 0.0234
  Δ:      -342.54

Recommendation:
  Raw:    Fine-tune with 20% target data
  Scaled: Transfer As-Is (0% target data)

Confidence:
  Raw:    62.3%
  Scaled: 85.7%
```

**Insight:** Scaling dramatically improves metric validity!

### **3. Framework Accuracy**

```
FRAMEWORK ACCURACY ANALYSIS
================================================================================

✅ Exp 5: UK → France
   Expected: MODERATE
   Predicted: MODERATE (score: 0.7234)
   Recommendation: Fine-tune with 10% target data

✅ Exp 6: UK → Germany
   Expected: MODERATE
   Predicted: MODERATE (score: 0.7156)
   Recommendation: Fine-tune with 10% target data

✅ Exp 7: High-Value → Medium-Value
   Expected: HIGH
   Predicted: HIGH (score: 0.8523)
   Recommendation: Transfer As-Is

Overall Framework Accuracy: 3/3 = 100.0%
```

### **4. Statistical Analysis**

- Mean/median/std for all metrics
- Confidence interval statistics
- Comparison with Week 2 synthetic data results

---

## 🔍 Understanding the Results

### **Composite Score** (0-1, higher = better transferability)

| Range | Transferability | Recommended Strategy |
|-------|----------------|----------------------|
| 0.80 - 1.00 | HIGH | Transfer As-Is |
| 0.60 - 0.79 | MODERATE | Fine-tune with 10-20% |
| 0.40 - 0.59 | LOW | Fine-tune with 50%+ |
| 0.00 - 0.39 | VERY LOW | Train New |

### **MMD Score** (lower = more similar)

- **< 0.05:** Very similar distributions
- **0.05 - 0.15:** Moderately similar
- **> 0.15:** Different distributions

### **Confidence** (%)

- **> 80%:** High confidence
- **60-79%:** Moderate confidence
- **< 60%:** Low confidence (more data needed)

---

## 🐛 Troubleshooting

### **Error: "Missing RFM files"**

**Solution:** Run Step 1 first
```bash
python uk_rfm_generator_FIXED.py
```

### **Error: "Import framework could not be resolved"**

**Solution:** Make sure you have week3 folder with framework.py
```bash
# Check if file exists
dir ..\week3\framework.py
```

### **Error: "No module named sklearn"**

**Solution:** Install dependencies
```bash
pip install pandas numpy scikit-learn
```

---

## 📈 Next Steps After Validation

1. **Review the validation report:**
   ```bash
   notepad validation_results\framework_validation_report.md
   ```

2. **Analyze the CSV results:**
   ```bash
   # Open in Excel or any CSV viewer
   start validation_results\uk_retail_validation_results.csv
   ```

3. **Use insights for final report (Week 8):**
   - Framework successfully validated on real data ✅
   - Scaling methodology proven effective ✅
   - Accuracy metrics documented ✅

4. **Compare with Week 2 results:**
   - Synthetic data (BigBasket): 7 domain pairs
   - Real data (UK Retail): 3 experiments
   - Framework generalizes well across both! ✅

---

## 💡 Key Insights You Should See

1. **Scaling is Critical:**
   - Raw features give misleading scores (currency mismatch)
   - Scaled features enable fair comparison
   - Difference can be 15-30% in composite score

2. **Sample Size Matters:**
   - UK (3,920) → France (87): Moderate transferability
   - UK (3,920) → Germany (94): Moderate transferability  
   - High-value (980) → Medium (1,960): High transferability
   - Larger target samples = better transfer confidence

3. **Framework Robustness:**
   - Works on both synthetic and real data
   - Predictions align with expected categories
   - Confidence scores reflect uncertainty appropriately

---

## ✅ Success Checklist

- [ ] UK RFM data generated (13 files)
- [ ] Validation checks passed (outlier capping, scaling, etc.)
- [ ] Framework validation completed (3 experiments)
- [ ] Validation report generated (.md file)
- [ ] Results CSV created
- [ ] Understood raw vs scaled comparison
- [ ] Framework accuracy documented

**All done?** You're ready for Week 7 (Final Integration) and Week 8 (Report Writing)! 🎉

---

**Questions?**
- Check the validation report for detailed analysis
- Review the CSV for raw numbers
- Re-run with `--help` flag for options

**Good luck! 🚀**
