# 📋 MEMBER 3 QUICK REFERENCE - Week 3 & 4

## ✅ WHAT YOU HAVE (Week 3 - DONE)
```
✓ framework.py          - Main framework class
✓ decision_engine.py    - Strategy recommender  
✓ metrics.py            - Transferability metrics
```

## 🎯 WHAT TO DO (Week 4 - TODO)

### 1️⃣ Wait for experiments
```bash
# Check if done
dir src\week3\results\ALL_EXPERIMENTS_RESULTS.csv
```

### 2️⃣ Run calibration
```bash
python src\week3\calibrate_and_validate.py
```

### 3️⃣ Check accuracy
```
Target: >= 70% (5+ correct out of 7 pairs)
```

### 4️⃣ Update thresholds (if needed)
Edit `decision_engine.py` line ~88 with new values

### 5️⃣ Write report
Create `calibration_report.md` with findings

---

## 📊 WHAT CHANGED: 4 → 7 PAIRS

| Aspect | Before (4 pairs) | After (7 pairs) |
|--------|-----------------|-----------------|
| Data points | 4 | 7 |
| LOW pairs | 1 | 3 |
| Coverage | Limited | Better spectrum |
| Thresholds | Old values | **NEED RECALIBRATION** |

---

## 🎯 SUCCESS CRITERIA

| Metric | Target | Interpretation |
|--------|--------|----------------|
| **Accuracy** | >= 70% | Framework works! |
| **Correlation** | r > 0.5 | Predictions meaningful |
| **Thresholds** | Data-driven | Not arbitrary |

---

## 🚀 FILES YOU'LL CREATE

```
✓ calibration_validation_report.txt  (Auto-generated)
✓ framework_validation.csv           (Auto-generated)
✓ calibration_correlation.png        (Auto-generated)
□ calibration_report.md               (You write this)
```

---

## ⏱️ TIME: ~5-8 hours total

- Calibration: 5 min
- Analysis: 1-2 hrs  
- Tuning: 2-3 hrs (if needed)
- Docs: 2-3 hrs

---

## 💡 KEY INSIGHT

**Your framework predicts which transfer strategy works best:**
- HIGH transferability → Transfer as-is ✨
- MODERATE → Fine-tune with some data 🔧
- LOW → Train from scratch 🏗️

**If >= 70% accurate, you've built something useful!**

---

## 📞 NEED HELP?

Read: `MEMBER3_GUIDE.md` (detailed walkthrough)
Read: `MEMBER3_SUMMARY.md` (complete overview)
Read: This file (quick reference)

---

**NEXT:** Run `python src\week3\calibrate_and_validate.py` 🚀
