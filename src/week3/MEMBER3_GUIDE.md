# Member 3: Transferability Framework Core - Week 3 & 4
## Updated for 7 Domain Pairs

---

## 📋 Overview

You're building the **core decision-making framework** that determines:
1. ✅ **Can we transfer?** (Calculate transferability score)
2. ✅ **How should we transfer?** (Recommend strategy)
3. ✅ **How well does it work?** (Validate predictions)

---

## 🎯 Your Deliverables

### Week 3 Deliverables:
- ✅ `framework.py` - Main framework class (ALREADY EXISTS)
- ✅ `decision_engine.py` - Strategy recommendation logic (ALREADY EXISTS)
- ✅ `metrics.py` - Transferability metrics (ALREADY EXISTS)

### Week 4 Deliverables (TO DO):
- [ ] `calibrate_and_validate.py` - **NEW! Calibrate for 7 pairs**
- [ ] Framework validation report
- [ ] Updated thresholds based on 7-pair experiments
- [ ] Accuracy analysis

---

## 🚀 Quick Start - What to Run

### Step 1: Make sure experiments are complete
```bash
# Check if experiment results exist
dir src\week3\results\ALL_EXPERIMENTS_RESULTS.csv
```

If the file doesn't exist, run experiments first:
```bash
python src\week3\run_all_experiments.py
```

---

### Step 2: Calibrate and Validate Framework
```bash
cd src\week3
python calibrate_and_validate.py
```

**This will:**
- ✅ Load results from all 35 experiments (7 pairs × 5 tests)
- ✅ Analyze correlation between predicted transferability and actual performance
- ✅ Determine optimal HIGH/MODERATE/LOW thresholds
- ✅ Validate framework accuracy
- ✅ Generate calibration report

**Expected Output:**
```
FRAMEWORK ACCURACY: XX.X%
Optimal Thresholds:
  HIGH:     >= 0.XXXX
  MODERATE: >= 0.XXXX
  LOW:      <  0.XXXX
```

---

### Step 3: Test Framework on Individual Pairs
```bash
python demo_framework.py --pair 1
```

This shows how the framework works for a specific domain pair.

---

## 📊 What Changed from 4 to 7 Pairs?

### Old Plan (4 pairs):
- Pair 1: Cleaning & Household → Foodgrains
- Pair 2: Snacks → Bakery
- Pair 3: Premium → Budget
- Pair 4: Popular → Niche

### New Plan (7 pairs):
- **Pair 1**: Cleaning & Household → Foodgrains (HIGH, 0.9028)
- **Pair 2**: Snacks → Kitchen, Garden & Pets (LOW, 0.7254)
- **Pair 3**: Beauty & Hygiene → Snacks (MODERATE, 0.8159)
- **Pair 4**: Gourmet → Beauty & Hygiene (MODERATE-HIGH, 0.8958)
- **Pair 5**: Eggs, Meat & Fish → Baby Care (LOW, 0.8036)
- **Pair 6**: Baby Care → Bakery (LOW, 0.7414)
- **Pair 7**: Beverages → Gourmet (MODERATE-HIGH, 0.8951)

### Impact:
- ✅ More data points for calibration (better threshold accuracy)
- ✅ Better coverage of transferability spectrum
- ✅ More LOW transferability pairs to test
- ⚠️ Thresholds need recalibration (was tuned for 4 pairs)

---

## 🔧 Key Files You're Working With

### 1. `framework.py` (Already exists)
The main framework class that integrates everything.

**Key methods:**
```python
framework = TransferLearningFramework(source_model, source_data, target_data)
score = framework.calculate_transferability()
recommendation = framework.recommend_strategy()
```

### 2. `decision_engine.py` (Already exists)
Makes strategy recommendations based on transferability scores.

**Key class:**
```python
engine = DecisionEngine(
    high_threshold=0.85,      # These will be recalibrated
    moderate_threshold=0.75,   # Based on 7-pair experiments
    low_threshold=0.50
)
recommendation = engine.recommend_strategy(composite_score=0.82)
```

### 3. `calibrate_and_validate.py` (NEW - Just created for you!)
Calibrates thresholds and validates framework using experimental results.

**Run this after experiments complete:**
```bash
python calibrate_and_validate.py
```

---

## 📈 Expected Results

### Correlation Analysis:
```
Predicted Transferability vs Zero-Shot Performance:
  Pearson r = 0.XX (p = 0.XXX)
  ✓ Statistically significant!
```

### Calibrated Thresholds:
Based on your 7 pairs, the script will determine:
- **HIGH**: Pairs where zero-shot transfer works (>= X% of from-scratch)
- **MODERATE**: Pairs where fine-tuning helps significantly
- **LOW**: Pairs needing to train from scratch

### Framework Accuracy:
```
Pair   Predicted           Actual              Correct?
1      transfer_as_is      transfer_as_is      ✓
2      fine_tune_light     fine_tune_moderate  ✓
3      fine_tune_moderate  fine_tune_heavy     ✓
...

FRAMEWORK ACCURACY: 85.7% (6/7)
```

**Target:** >= 70% accuracy

---

## 🎯 Your Week 3-4 Tasks

### Week 3: ✅ Core Framework (Already Done!)
- [x] `framework.py` - Main framework class
- [x] `decision_engine.py` - Recommendation engine
- [x] `metrics.py` - Transferability metrics

### Week 4: Calibration & Validation (TO DO)

#### Task 1: Run Calibration (30 minutes)
```bash
python calibrate_and_validate.py
```

#### Task 2: Analyze Results (1-2 hours)
- Review correlation plots
- Check if predictions match reality
- Identify any outliers

#### Task 3: Tune if Needed (2-3 hours)
If accuracy < 70%:
- Adjust metric weights in `decision_engine.py`
- Consider different threshold calculation methods
- Re-run calibration

#### Task 4: Document Findings (2-3 hours)
Create `calibration_report.md`:
- What thresholds were determined?
- Why these thresholds make sense
- Framework accuracy results
- Insights from 7-pair analysis

#### Task 5: Create Demo (1-2 hours)
Update `demo_framework.py` to show:
- How to use framework on new domain pair
- Example outputs for each transferability level
- Interpretation guide

---

## 💡 Key Insights to Look For

### Question 1: Does Transferability Score Predict Performance?
**Look at:** Correlation between `transferability_score` and `zero_shot_performance`

**Expected:** Positive correlation (r > 0.5)
- HIGH transferability → Good zero-shot performance
- LOW transferability → Poor zero-shot performance

### Question 2: Do Low Transferability Pairs Benefit More from Fine-tuning?
**Look at:** Correlation between `transferability_score` and `improvement_from_finetune`

**Expected:** Negative correlation (r < -0.3)
- LOW transferability → Large improvement from fine-tuning
- HIGH transferability → Small improvement (already good)

### Question 3: What's the Optimal Threshold?
**Look at:** Where do pairs naturally cluster?

**Method:**
- Sort pairs by transferability score
- Find natural breakpoints based on actual best strategy
- Set thresholds at these breakpoints

### Question 4: Can We Trust the Framework?
**Look at:** Framework accuracy (% correct predictions)

**Target:** >= 70% accuracy
- 7/7 = 100% (Perfect!)
- 6/7 = 85.7% (Excellent)
- 5/7 = 71.4% (Good)
- 4/7 = 57.1% (Needs work)

---

## 🔍 Troubleshooting

### Issue: Accuracy < 70%

**Solution 1:** Adjust thresholds
```python
# In decision_engine.py, try different values
high_threshold = 0.88  # Stricter
moderate_threshold = 0.75
```

**Solution 2:** Change metric weights
```python
# In decision_engine.py
metric_weights = {
    'mmd': 0.25,           # Increase if distribution matters more
    'js_divergence': 0.20,
    'correlation_stability': 0.30,  # Increase if relationships matter
    'ks_statistic': 0.15,
    'wasserstein': 0.10
}
```

**Solution 3:** Use quantile-based thresholds
```python
# In calibrate_and_validate.py, try quantile method
high_threshold = scores.quantile(0.70)   # Top 30%
moderate_threshold = scores.quantile(0.40)  # Middle 30%
```

---

### Issue: Correlation is Weak (r < 0.3)

**Possible Reasons:**
1. **Data quality issues** - Check RFM data
2. **Metric weights wrong** - Some metrics may be noisy
3. **Domain pairs too different** - Framework may not generalize well

**Solutions:**
- Focus on top 2-3 most predictive metrics
- Use non-linear threshold function
- Add domain-specific adjustments

---

### Issue: Script Crashes

**Check:**
```bash
# Make sure results file exists
dir src\week3\results\ALL_EXPERIMENTS_RESULTS.csv

# Check it has all columns
python -c "import pandas as pd; df = pd.read_csv('src/week3/results/ALL_EXPERIMENTS_RESULTS.csv'); print(df.columns.tolist())"

# Verify 35 rows (7 pairs × 5 tests)
python -c "import pandas as pd; df = pd.read_csv('src/week3/results/ALL_EXPERIMENTS_RESULTS.csv'); print(f'Rows: {len(df)}')"
```

---

## 📝 Deliverables Checklist

### Code:
- [ ] `calibrate_and_validate.py` runs successfully
- [ ] Generates correlation plots
- [ ] Produces validation report
- [ ] Calculates optimal thresholds

### Documentation:
- [ ] `calibration_report.md` explaining findings
- [ ] Updated README with new thresholds
- [ ] Example usage in `demo_framework.py`

### Results:
- [ ] `framework_validation.csv` - Predictions vs actual
- [ ] `calibration_correlation.png` - Visualization
- [ ] `calibration_validation_report.txt` - Full report
- [ ] Framework accuracy >= 70%

---

## 🎓 What You're Learning

### Technical Skills:
- Statistical calibration techniques
- Threshold optimization
- Model validation
- Correlation analysis

### Research Skills:
- Experimental validation
- Hypothesis testing
- Results interpretation
- Technical writing

### Software Engineering:
- Building decision systems
- Framework design
- Testing and validation
- Documentation

---

## 🚀 Next Steps After Week 4

Once calibration is complete:

1. **Update `decision_engine.py`** with optimal thresholds
2. **Create user guide** for framework
3. **Prepare demo** for presentation
4. **Write methodology** for technical report

---

## 📞 Questions?

Common questions:

**Q: Do I need to modify existing files?**
A: Probably! Update thresholds in `decision_engine.py` based on calibration results.

**Q: What if accuracy is low?**
A: Try different threshold calculation methods or adjust metric weights.

**Q: How long should this take?**
A: Week 4 tasks = 8-10 hours total (calibration + analysis + documentation)

**Q: Can I test framework before experiments finish?**
A: Yes! Use the existing transferability scores from Week 1 as a baseline.

---

**Ready to calibrate? Run:**
```bash
python src\week3\calibrate_and_validate.py
```

Good luck! 🎯
