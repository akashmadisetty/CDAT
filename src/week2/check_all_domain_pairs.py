# Verification Script: Check All Domain Pairs for Customer Overlap
# Run this AFTER generating RFM data to verify correctness

import os
import pandas as pd

print("="*80)
print("DOMAIN PAIR VERIFICATION SCRIPT")
print("Checking for customer overlap across all domain pairs")
print("="*80)

# Configuration
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

DOMAIN_PAIRS = [1, 2, 3, 4]

all_validation_passed = True
all_customers_global = set()

for pair_id in DOMAIN_PAIRS:
    print(f"\n{'='*80}")
    print(f"CHECKING PAIR {pair_id}")
    print(f"{'='*80}")
    
    # File paths
    source_file = os.path.join(OUTPUT_DIR, f'domain_pair{pair_id}_source_RFM.csv')
    target_file = os.path.join(OUTPUT_DIR, f'domain_pair{pair_id}_target_RFM.csv')
    
    # Check if files exist
    if not os.path.exists(source_file):
        print(f"❌ ERROR: {source_file} not found!")
        all_validation_passed = False
        continue
    
    if not os.path.exists(target_file):
        print(f"❌ ERROR: {target_file} not found!")
        all_validation_passed = False
        continue
    
    # Load data
    try:
        source_rfm = pd.read_csv(source_file)
        target_rfm = pd.read_csv(target_file)
    except Exception as e:
        print(f"❌ ERROR loading files: {e}")
        all_validation_passed = False
        continue
    
    # Get customer IDs
    source_customers = set(source_rfm['customer_id'])
    target_customers = set(target_rfm['customer_id'])
    
    print(f"\nSource customers: {len(source_customers):,}")
    print(f"  Range: {min(source_customers)} to {max(source_customers)}")
    
    print(f"\nTarget customers: {len(target_customers):,}")
    print(f"  Range: {min(target_customers)} to {max(target_customers)}")
    
    # Check for overlap within pair
    overlap_within_pair = source_customers & target_customers
    
    print(f"\nWithin-pair overlap: {len(overlap_within_pair):,} customers")
    
    if len(overlap_within_pair) > 0:
        print(f"❌ FAIL: Source and target overlap!")
        print(f"   Overlapping IDs (first 10): {list(overlap_within_pair)[:10]}")
        all_validation_passed = False
    else:
        print(f"✅ OK: Source and target are disjoint")
    
    # Check for duplicates within each domain
    source_dups = len(source_rfm) - len(source_customers)
    target_dups = len(target_rfm) - len(target_customers)
    
    if source_dups > 0:
        print(f"❌ FAIL: {source_dups} duplicate customer IDs in source!")
        all_validation_passed = False
    else:
        print(f"✅ OK: No duplicates in source")
    
    if target_dups > 0:
        print(f"❌ FAIL: {target_dups} duplicate customer IDs in target!")
        all_validation_passed = False
    else:
        print(f"✅ OK: No duplicates in target")
    
    # Check for overlap with OTHER pairs
    overlap_with_other_pairs = (source_customers | target_customers) & all_customers_global
    
    if len(overlap_with_other_pairs) > 0:
        print(f"❌ FAIL: {len(overlap_with_other_pairs)} customers overlap with other pairs!")
        print(f"   Overlapping IDs (first 10): {list(overlap_with_other_pairs)[:10]}")
        all_validation_passed = False
    else:
        print(f"✅ OK: No overlap with other pairs")
    
    # Add to global set
    all_customers_global.update(source_customers)
    all_customers_global.update(target_customers)
    
    # Check RFM data quality
    print(f"\nData Quality:")
    
    # Check for missing values
    source_nulls = source_rfm[['Recency', 'Frequency', 'Monetary']].isnull().sum().sum()
    target_nulls = target_rfm[['Recency', 'Frequency', 'Monetary']].isnull().sum().sum()
    
    if source_nulls > 0 or target_nulls > 0:
        print(f"⚠️  WARNING: Missing values detected (Source: {source_nulls}, Target: {target_nulls})")
    else:
        print(f"✅ OK: No missing values in RFM features")
    
    # Check RFM ranges
    print(f"\nRFM Statistics:")
    print(f"  Source Recency: {source_rfm['Recency'].min():.0f} - {source_rfm['Recency'].max():.0f} days")
    print(f"  Source Frequency: {source_rfm['Frequency'].min():.0f} - {source_rfm['Frequency'].max():.0f} purchases")
    print(f"  Source Monetary: ₹{source_rfm['Monetary'].min():.2f} - ₹{source_rfm['Monetary'].max():.2f}")
    
    print(f"\n  Target Recency: {target_rfm['Recency'].min():.0f} - {target_rfm['Recency'].max():.0f} days")
    print(f"  Target Frequency: {target_rfm['Frequency'].min():.0f} - {target_rfm['Frequency'].max():.0f} purchases")
    print(f"  Target Monetary: ₹{target_rfm['Monetary'].min():.2f} - ₹{target_rfm['Monetary'].max():.2f}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print(f"\n{'='*80}")
print("FINAL VERIFICATION SUMMARY")
print(f"{'='*80}")

print(f"\nTotal unique customers across all pairs: {len(all_customers_global):,}")
print(f"Expected: {4 * (1500 + 1200):,}")

expected_total = 4 * (1500 + 1200)

if len(all_customers_global) == expected_total:
    print("✅ Customer count matches expectation!")
else:
    print(f"⚠️  Customer count mismatch! (Expected: {expected_total:,}, Got: {len(all_customers_global):,})")
    all_validation_passed = False

print(f"\n{'='*80}")
if all_validation_passed:
    print("✅ ALL VALIDATION CHECKS PASSED!")
    print("="*80)
    print("\nYour RFM data is ready to use:")
    print("  - All customer sets are disjoint")
    print("  - No duplicates detected")
    print("  - Data quality is good")
    print("\nYou can now proceed with:")
    print("  1. Training baseline models (Member 2)")
    print("  2. Calculating transferability metrics (Member 3)")
    print("  3. Week 2 experiments")
else:
    print("❌ VALIDATION FAILED!")
    print("="*80)
    print("\n⚠️  Issues detected - please review errors above")
    print("\nYou need to regenerate the RFM data using:")
    print("  python generate_rfm_all_pairs_FIXED.py")

print(f"\n{'='*80}")
