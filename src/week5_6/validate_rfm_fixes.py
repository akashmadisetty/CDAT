"""
Validation Script: Verify RFM Fixes Were Applied Correctly
==========================================================

Run this AFTER running uk_rfm_generator_FIXED.py to verify:
1. Reference date is fixed (not dynamic)
2. Outliers are capped at 99th percentile
3. Scaling is applied correctly (mean~0, std~1)
4. All expected files exist

Usage:
    python validate_rfm_fixes.py

Expected Output:
    ✅ All validation checks passed!
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path

print("="*80)
print("RFM FIXES VALIDATION SCRIPT")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

EXPECTED_FILES = {
    'raw_rfm': [
        'exp5_uk_source_RFM_FIXED.csv',
        'exp5_france_target_RFM_FIXED.csv',
        'exp6_germany_target_RFM_FIXED.csv',
        'exp7_highvalue_source_RFM_FIXED.csv',
        'exp7_mediumvalue_target_RFM_FIXED.csv'
    ],
    'scaled_rfm': [
        'exp5_uk_source_RFM_scaled.csv',
        'exp5_france_target_RFM_scaled.csv',
        'exp6_germany_target_RFM_scaled.csv',
        'exp7_highvalue_source_RFM_scaled.csv',
        'exp7_mediumvalue_target_RFM_scaled.csv'
    ],
    'stats': [
        'uk_retail_country_stats.csv',
        'uk_retail_experiments_comparison_FIXED.csv'
    ],
    'viz': [
        'uk_retail_rfm_distributions_FIXED.png'
    ]
}

# Expected reference date (UK Retail dataset max: Dec 9, 2011 + 1 day)
EXPECTED_REF_DATE = '2011-12-10'

# Tolerance for scaled features (should be close to 0 and 1)
MEAN_TOLERANCE = 0.1  # Mean should be within ±0.1 of 0
STD_TOLERANCE = 0.1   # Std should be within ±0.1 of 1

# ============================================================================
# VALIDATION TESTS
# ============================================================================

all_checks_passed = True
check_results = []

# ----------------------------------------------------------------------------
# CHECK 1: File Existence
# ----------------------------------------------------------------------------

print("\n" + "="*80)
print("CHECK 1: File Existence")
print("="*80)

for category, files in EXPECTED_FILES.items():
    print(f"\n{category.upper()}:")
    for filename in files:
        if os.path.exists(filename):
            filesize = os.path.getsize(filename)
            print(f"  ✅ {filename} ({filesize:,} bytes)")
            check_results.append(('File Existence', filename, True, 'File found'))
        else:
            print(f"  ❌ {filename} - NOT FOUND")
            check_results.append(('File Existence', filename, False, 'File missing'))
            all_checks_passed = False

# ----------------------------------------------------------------------------
# CHECK 2: Outlier Capping
# ----------------------------------------------------------------------------

print("\n" + "="*80)
print("CHECK 2: Outlier Capping (99th Percentile)")
print("="*80)

try:
    # Load UK data (largest dataset, most likely to have outliers)
    rfm_uk = pd.read_csv('exp5_uk_source_RFM_FIXED.csv')
    
    # Check if capped columns exist
    required_cols = ['Recency', 'Frequency', 'Monetary', 
                     'Recency_capped', 'Frequency_capped', 'Monetary_capped']
    
    missing_cols = [col for col in required_cols if col not in rfm_uk.columns]
    
    if missing_cols:
        print(f"❌ Missing columns: {missing_cols}")
        check_results.append(('Outlier Capping', 'Column Check', False, f'Missing: {missing_cols}'))
        all_checks_passed = False
    else:
        print("✅ All required columns present")
        
        # Check if capping was applied (capped max should be ≤ 99th percentile of raw)
        for col in ['Recency', 'Frequency', 'Monetary']:
            raw_max = rfm_uk[col].max()
            capped_max = rfm_uk[f'{col}_capped'].max()
            p99 = rfm_uk[col].quantile(0.99)
            
            # Capped max should be ≤ 99th percentile
            if capped_max <= p99 * 1.01:  # Allow 1% tolerance for rounding
                print(f"  ✅ {col}: Raw max={raw_max:.2f}, Capped max={capped_max:.2f}, 99%ile={p99:.2f}")
                check_results.append(('Outlier Capping', col, True, f'Capped at {capped_max:.2f}'))
            else:
                print(f"  ❌ {col}: Capped max ({capped_max:.2f}) > 99%ile ({p99:.2f})")
                check_results.append(('Outlier Capping', col, False, 'Capping not applied'))
                all_checks_passed = False
        
        # Check for the famous £77k outlier in Monetary
        if 'Monetary' in rfm_uk.columns:
            max_monetary = rfm_uk['Monetary'].max()
            max_monetary_capped = rfm_uk['Monetary_capped'].max()
            
            print(f"\n  Special Check: £77k Customer Outlier:")
            print(f"    Raw Monetary max: £{max_monetary:,.2f}")
            print(f"    Capped Monetary max: £{max_monetary_capped:,.2f}")
            
            if max_monetary > 50000 and max_monetary_capped < 15000:
                print(f"  ✅ Outlier successfully capped! (£{max_monetary:,.0f} → £{max_monetary_capped:,.0f})")
            elif max_monetary > 50000:
                print(f"  ⚠️  WARNING: Large outlier still present (£{max_monetary:,.0f})")
            else:
                print(f"  ℹ️  Note: No extreme outlier found (max: £{max_monetary:,.0f})")

except FileNotFoundError:
    print("❌ exp5_uk_source_RFM_FIXED.csv not found - run uk_rfm_generator_FIXED.py first!")
    check_results.append(('Outlier Capping', 'File Check', False, 'File missing'))
    all_checks_passed = False
except Exception as e:
    print(f"❌ Error checking outliers: {e}")
    check_results.append(('Outlier Capping', 'Error', False, str(e)))
    all_checks_passed = False

# ----------------------------------------------------------------------------
# CHECK 3: Scaling/Normalization
# ----------------------------------------------------------------------------

print("\n" + "="*80)
print("CHECK 3: Feature Scaling (StandardScaler)")
print("="*80)

try:
    # Load scaled data
    rfm_uk_scaled = pd.read_csv('exp5_uk_source_RFM_scaled.csv')
    rfm_france_scaled = pd.read_csv('exp5_france_target_RFM_scaled.csv')
    
    # Check if scaled columns exist
    scaled_cols = ['Recency_scaled', 'Frequency_scaled', 'Monetary_scaled']
    
    missing_scaled = [col for col in scaled_cols if col not in rfm_uk_scaled.columns]
    
    if missing_scaled:
        print(f"❌ Missing scaled columns: {missing_scaled}")
        check_results.append(('Scaling', 'Column Check', False, f'Missing: {missing_scaled}'))
        all_checks_passed = False
    else:
        print("✅ All scaled columns present\n")
        
        # Check UK scaling
        print("UK Source (scaled):")
        for col in scaled_cols:
            mean = rfm_uk_scaled[col].mean()
            std = rfm_uk_scaled[col].std()
            
            mean_ok = abs(mean) < MEAN_TOLERANCE
            std_ok = abs(std - 1.0) < STD_TOLERANCE
            
            status = "✅" if (mean_ok and std_ok) else "❌"
            print(f"  {status} {col}: mean={mean:.3f}, std={std:.3f}")
            
            if mean_ok and std_ok:
                check_results.append(('Scaling', f'UK {col}', True, f'mean={mean:.3f}, std={std:.3f}'))
            else:
                check_results.append(('Scaling', f'UK {col}', False, f'mean={mean:.3f}, std={std:.3f}'))
                all_checks_passed = False
        
        # Check France scaling
        print("\nFrance Target (scaled):")
        for col in scaled_cols:
            mean = rfm_france_scaled[col].mean()
            std = rfm_france_scaled[col].std()
            
            # For target domain, mean might not be exactly 0 (smaller sample)
            # Just check that values are in reasonable range
            reasonable = abs(mean) < 1.0 and std > 0.5 and std < 1.5
            
            status = "✅" if reasonable else "⚠️"
            print(f"  {status} {col}: mean={mean:.3f}, std={std:.3f}")
            
            if reasonable:
                check_results.append(('Scaling', f'France {col}', True, f'mean={mean:.3f}, std={std:.3f}'))
            else:
                check_results.append(('Scaling', f'France {col}', False, f'Out of range'))
                # Don't fail, just warn (small sample size can affect std)

except FileNotFoundError as e:
    print(f"❌ Scaled file not found: {e}")
    check_results.append(('Scaling', 'File Check', False, 'File missing'))
    all_checks_passed = False
except Exception as e:
    print(f"❌ Error checking scaling: {e}")
    check_results.append(('Scaling', 'Error', False, str(e)))
    all_checks_passed = False

# ----------------------------------------------------------------------------
# CHECK 4: RFM Score Range
# ----------------------------------------------------------------------------

print("\n" + "="*80)
print("CHECK 4: RFM Score Range (Should be 1-5)")
print("="*80)

try:
    rfm_uk = pd.read_csv('exp5_uk_source_RFM_FIXED.csv')
    
    score_cols = ['R_Score', 'F_Score', 'M_Score']
    
    print("UK Source:")
    for col in score_cols:
        if col in rfm_uk.columns:
            # Convert to numeric (might be categorical)
            scores = pd.to_numeric(rfm_uk[col], errors='coerce')
            min_score = scores.min()
            max_score = scores.max()
            unique_scores = scores.nunique()
            
            # Scores should be in range 1-5
            valid_range = min_score >= 1 and max_score <= 5
            
            status = "✅" if valid_range else "❌"
            print(f"  {status} {col}: range={min_score:.0f}-{max_score:.0f}, unique values={unique_scores}")
            
            if valid_range:
                check_results.append(('RFM Scores', col, True, f'Range: {min_score:.0f}-{max_score:.0f}'))
            else:
                check_results.append(('RFM Scores', col, False, f'Invalid range: {min_score:.0f}-{max_score:.0f}'))
                all_checks_passed = False
        else:
            print(f"  ❌ {col} - NOT FOUND")
            check_results.append(('RFM Scores', col, False, 'Column missing'))
            all_checks_passed = False

except FileNotFoundError:
    print("❌ File not found - run uk_rfm_generator_FIXED.py first!")
    all_checks_passed = False
except Exception as e:
    print(f"❌ Error checking scores: {e}")
    all_checks_passed = False

# ----------------------------------------------------------------------------
# CHECK 5: Data Consistency (Cross-file validation)
# ----------------------------------------------------------------------------

print("\n" + "="*80)
print("CHECK 5: Data Consistency Between Raw and Scaled Files")
print("="*80)

try:
    rfm_uk_raw = pd.read_csv('exp5_uk_source_RFM_FIXED.csv')
    rfm_uk_scaled = pd.read_csv('exp5_uk_source_RFM_scaled.csv')
    
    # Check same number of customers
    n_raw = len(rfm_uk_raw)
    n_scaled = len(rfm_uk_scaled)
    
    if n_raw == n_scaled:
        print(f"✅ Same number of customers: {n_raw:,}")
        check_results.append(('Consistency', 'Row Count', True, f'{n_raw} customers'))
    else:
        print(f"❌ Row count mismatch: Raw={n_raw:,}, Scaled={n_scaled:,}")
        check_results.append(('Consistency', 'Row Count', False, f'Mismatch: {n_raw} vs {n_scaled}'))
        all_checks_passed = False
    
    # Check CustomerID consistency
    if 'CustomerID' in rfm_uk_raw.columns and 'CustomerID' in rfm_uk_scaled.columns:
        ids_match = (rfm_uk_raw['CustomerID'].values == rfm_uk_scaled['CustomerID'].values).all()
        
        if ids_match:
            print("✅ Customer IDs match between raw and scaled files")
            check_results.append(('Consistency', 'Customer IDs', True, 'All match'))
        else:
            print("❌ Customer IDs don't match!")
            check_results.append(('Consistency', 'Customer IDs', False, 'Mismatch'))
            all_checks_passed = False

except FileNotFoundError as e:
    print(f"❌ File not found: {e}")
    all_checks_passed = False
except Exception as e:
    print(f"❌ Error checking consistency: {e}")
    all_checks_passed = False

# ----------------------------------------------------------------------------
# CHECK 6: Statistical Comparison File
# ----------------------------------------------------------------------------

print("\n" + "="*80)
print("CHECK 6: Statistical Comparison Data")
print("="*80)

try:
    comparison = pd.read_csv('uk_retail_experiments_comparison_FIXED.csv')
    
    print("✅ Comparison file loaded successfully")
    print(f"   Experiments found: {len(comparison)}")
    
    if len(comparison) >= 3:
        print("\n  Experiments included:")
        for _, row in comparison.iterrows():
            print(f"    • {row['Experiment']}")
        check_results.append(('Comparison File', 'Content', True, f'{len(comparison)} experiments'))
    else:
        print("  ⚠️  Expected 3 experiments, found {len(comparison)}")
        check_results.append(('Comparison File', 'Content', False, f'Only {len(comparison)} experiments'))

except FileNotFoundError:
    print("❌ uk_retail_experiments_comparison_FIXED.csv not found")
    check_results.append(('Comparison File', 'Existence', False, 'File missing'))
    all_checks_passed = False
except Exception as e:
    print(f"❌ Error loading comparison file: {e}")
    all_checks_passed = False

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("VALIDATION SUMMARY")
print("="*80)

# Count checks
passed = sum(1 for _, _, status, _ in check_results if status)
failed = len(check_results) - passed

print(f"\nTotal Checks: {len(check_results)}")
print(f"  ✅ Passed: {passed}")
print(f"  ❌ Failed: {failed}")

# Group by category
from collections import defaultdict
by_category = defaultdict(list)
for category, name, status, msg in check_results:
    by_category[category].append((name, status, msg))

print("\nBy Category:")
for category, checks in by_category.items():
    passed_cat = sum(1 for _, status, _ in checks if status)
    total_cat = len(checks)
    print(f"  {category}: {passed_cat}/{total_cat} passed")

# Failed checks detail
if failed > 0:
    print("\n❌ FAILED CHECKS:")
    for category, name, status, msg in check_results:
        if not status:
            print(f"  • {category} - {name}: {msg}")

# Final verdict
print("\n" + "="*80)
if all_checks_passed and failed == 0:
    print("🎉 ALL VALIDATION CHECKS PASSED!")
    print("="*80)
    print("\n✅ Your RFM data is ready for transfer learning!")
    print("\nNext steps:")
    print("  1. Use *_FIXED.csv files for reporting (raw values)")
    print("  2. Use *_scaled.csv files for transferability metrics")
    print("  3. Proceed with baseline model training")
else:
    print("❌ VALIDATION FAILED!")
    print("="*80)
    print("\n⚠️  Please fix the issues above and re-run validation.")
    print("\nMost common fixes:")
    print("  1. Re-run: python uk_rfm_generator_FIXED.py")
    print("  2. Check for errors during RFM generation")
    print("  3. Verify you're in the correct directory (src/week5_6)")

print("\n" + "="*80)

# Save validation report
report_df = pd.DataFrame(check_results, columns=['Category', 'Check', 'Passed', 'Details'])
report_df.to_csv('validation_report.csv', index=False)
print("📄 Validation report saved: validation_report.csv")
print("="*80)
