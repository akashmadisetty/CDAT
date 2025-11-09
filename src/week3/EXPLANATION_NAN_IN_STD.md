# Why Some Categories Show "NaN" for Standard Deviation

## Quick Answer

**NaN appears when there's only 1 pair in that category**

Standard deviation measures variation/spread between multiple values. With only 1 value, there's nothing to compare to, so Pandas returns `NaN`.

---

## Distribution of 7 Pairs Across Categories

```
HIGH          : 1 pair  (Pair 1)
LOW           : 3 pairs (Pairs 2, 5, 6)
MODERATE      : 1 pair  (Pair 3)
MODERATE-HIGH : 2 pairs (Pairs 4, 7)
```

---

## Your Results Explained

```
              zero_shot         best_finetune         from_scratch
                   mean     std          mean     std         mean     std
category
HIGH             0.3694     NaN        0.5937     NaN       0.5885     NaN  ← Only 1 pair!
LOW              0.2686  0.0658        0.5981  0.0085       0.5914  0.0173  ← 3 pairs ✓
MODERATE         0.2933     NaN        0.5484     NaN       0.5392     NaN  ← Only 1 pair!
MODERATE-HIGH    0.3913  0.0602        0.5622  0.0333       0.5666  0.0346  ← 2 pairs ✓
```

### Why This Happens:

**HIGH Category (NaN std):**
- Only Pair 1: Cleaning & Household → Foodgrains
- Mean = 0.3694 (just that one value)
- Std = NaN (can't calculate variation from 1 value)

**MODERATE Category (NaN std):**
- Only Pair 3: Beauty & Hygiene → Snacks
- Mean = 0.2933 (just that one value)
- Std = NaN (can't calculate variation from 1 value)

**MODERATE-HIGH Category (Has std):**
- Pair 4: Gourmet → Beauty (0.4338)
- Pair 7: Beverages → Gourmet (0.3487)
- Mean = 0.3913 (average of 0.4338 and 0.3487)
- Std = 0.0602 (measures difference between these two values)

**LOW Category (Has std):**
- Pair 2: Snacks → Kitchen (0.1926)
- Pair 5: Eggs/Meat → Baby Care (0.3094)
- Pair 6: Baby Care → Bakery (0.3038)
- Mean = 0.2686 (average of all three)
- Std = 0.0658 (measures variation between these values)

---

## Is This a Problem?

**NO!** ✅ This is normal and expected.

### Why It's Fine:

1. **Statistical Reality:**
   - Std requires ≥2 values to measure variation
   - NaN correctly indicates "not enough data for std"

2. **Mean is Still Valid:**
   - Mean of 1 value = that value itself
   - Perfectly accurate representation

3. **Interpretation:**
   - HIGH: Mean shows Pair 1's performance
   - MODERATE: Mean shows Pair 3's performance
   - No averaging or smoothing needed

---

## What Does This Tell Us?

### Categories with Multiple Pairs (LOW, MODERATE-HIGH):

**LOW Category (3 pairs):**
```
Zero-shot mean: 0.2686 ± 0.0658
```
- The ±0.0658 shows there's **variation** in how LOW transferability pairs perform
- Some LOW pairs do slightly better than others
- But all still cluster around ~0.27 zero-shot performance

**MODERATE-HIGH Category (2 pairs):**
```
Zero-shot mean: 0.3913 ± 0.0602
```
- The ±0.0602 shows **consistency** between Pairs 4 and 7
- Both perform similarly despite being different domains
- Good validation that the category is meaningful

### Categories with Single Pairs (HIGH, MODERATE):

**HIGH Category (1 pair):**
```
Zero-shot mean: 0.3694 (NaN std)
```
- Just showing Pair 1's actual performance
- No variation to measure (nothing to compare to)
- The value IS the category representation

**MODERATE Category (1 pair):**
```
Zero-shot mean: 0.2933 (NaN std)
```
- Just showing Pair 3's actual performance
- No variation to measure
- The value IS the category representation

---

## Should We Be Concerned?

### For Analysis: **NO** ✅

The statistics are correct and meaningful:
- Categories with multiple pairs show variation (std)
- Categories with single pairs show exact values (NaN std)
- Both are valid representations

### For Reporting:

When presenting results, you can:

**Option 1: Keep as-is**
```
HIGH: 0.3694 (n=1)
```

**Option 2: Show count instead of std**
```
HIGH: 0.3694 (1 pair)
LOW:  0.2686 ± 0.0658 (3 pairs)
```

**Option 3: Remove std column for single-pair categories**
```
Category      | Pairs | Zero-Shot Mean | Std (if n>1)
HIGH          | 1     | 0.3694        | -
LOW           | 3     | 0.2686        | 0.0658
MODERATE      | 1     | 0.2933        | -
MODERATE-HIGH | 2     | 0.3913        | 0.0602
```

---

## Key Takeaway

**NaN for std is CORRECT, not an error!**

- Mathematically valid (can't compute std from 1 value)
- Statistically appropriate (Pandas convention)
- Analytically meaningful (shows single-pair categories)

The analysis is sound. Your framework successfully handles categories with varying numbers of pairs! ✅
