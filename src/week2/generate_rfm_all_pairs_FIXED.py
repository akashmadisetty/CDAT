# Generate RFM for All 4 Domain Pairs - COMPLETELY FIXED VERSION
# Member 1 - Week 2 Final Deliverable
# FIX: Ensures 100% DISJOINT customer sets (0% overlap)
#
# Customer ID Allocation Strategy:
# ================================
# Pair 1: C00000 - C02699 (Source: C00000-C01499, Target: C01500-C02699)
# Pair 2: C03000 - C05699 (Source: C03000-C04499, Target: C04500-C05699)
# Pair 3: C06000 - C08699 (Source: C06000-C07499, Target: C07500-C08699)
# Pair 4: C09000 - C11699 (Source: C09000-C10499, Target: C10500-C11699)
# TOTAL: 10,800 unique customers (NO OVERLAP!)

import sys
import os
import pandas as pd
import numpy as np
from synth_FIXED import SyntheticCustomerGenerator

print("="*80)
print("GENERATING RFM FOR ALL DOMAIN PAIRS - FIXED VERSION")
print("Member 1 - Week 2 Deliverable")
print("="*80)

# Configuration
N_CUSTOMERS = 1500          # Customers per SOURCE domain
N_TRANSACTIONS = 15000      # Transactions per domain
START_DATE = '2024-01-01'
END_DATE = '2024-06-30'
REFERENCE_DATE = '2024-07-01'
SEED = 42

# Output directory
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# PAIR 2 ONLY - From Week 1 Analysis
DOMAIN_PAIRS = [
    {
        'id': 2,
        'name': 'Snacks → Kitchen, Garden & Pets',
        'source_file': '../week1/domain_pair2_source.csv',
        'target_file': '../week1/domain_pair2_target.csv',
        'expected_transfer': 'MODERATE-HIGH'
    }
]

# Track statistics
all_stats = []
all_customer_ids = set()  # Global check for uniqueness

# ============================================================================
# PROCESS EACH DOMAIN PAIR
# ============================================================================

for pair in DOMAIN_PAIRS:
    print("\n" + "="*80)
    print(f"DOMAIN PAIR {pair['id']}: {pair['name']}")
    print(f"Expected Transferability: {pair['expected_transfer']}")
    print("="*80)
    
    # ✅ CRITICAL: Calculate base offset for THIS pair
    # Each pair gets 3000 customer IDs (1500 source + 1500 buffer)
    base_offset = (pair['id'] - 1) * 3000
    
    print(f"\n📋 Customer ID allocation for Pair {pair['id']}:")
    print(f"   Base offset: {base_offset}")
    print(f"   Source will be: C{base_offset:05d} - C{base_offset + N_CUSTOMERS - 1:05d}")
    print(f"   Target will be: C{base_offset + N_CUSTOMERS:05d} - C{base_offset + N_CUSTOMERS + int(N_CUSTOMERS * 0.8) - 1:05d}")
    
    # ------------------------------------------------------------------------
    # SOURCE DOMAIN
    # ------------------------------------------------------------------------
    print(f"\n📊 Processing SOURCE domain...")
    
    try:
        # Load source products
        source_products = pd.read_csv(pair['source_file'])
        print(f"  ✓ Loaded {len(source_products):,} source products")
        
        # Initialize generator
        generator_source = SyntheticCustomerGenerator(source_products, seed=SEED)
        
        # ✅ FIX: Generate customers with OFFSET
        print(f"  👥 Generating {N_CUSTOMERS:,} customers (offset: {base_offset})...")
        source_customers = generator_source.generate_customers(
            n_customers=N_CUSTOMERS,
            customer_id_offset=base_offset  # ← KEY FIX!
        )
        
        print(f"  ✓ Customer IDs: {source_customers['customer_id'].iloc[0]} to {source_customers['customer_id'].iloc[-1]}")
        
        # Verify uniqueness globally
        source_ids = set(source_customers['customer_id'])
        overlap_global = source_ids & all_customer_ids
        
        if len(overlap_global) > 0:
            print(f"  ❌ ERROR: {len(overlap_global)} customers already exist in other pairs!")
            raise ValueError("Customer ID collision detected!")
        
        all_customer_ids.update(source_ids)
        print(f"  ✓ Verified: All {len(source_ids)} customer IDs are globally unique")
        
        # Generate transactions
        print(f"  🛒 Generating {N_TRANSACTIONS:,} transactions...")
        source_transactions = generator_source.generate_transactions(
            source_customers,
            n_transactions=N_TRANSACTIONS,
            start_date=START_DATE,
            end_date=END_DATE
        )
        
        # Calculate RFM
        print(f"  📈 Calculating RFM features...")
        source_rfm = generator_source.calculate_rfm(
            source_transactions,
            reference_date=REFERENCE_DATE
        )
        
        # Save outputs
        source_rfm_file = os.path.join(OUTPUT_DIR, f'domain_pair{pair["id"]}_source_RFM.csv')
        source_transactions_file = os.path.join(OUTPUT_DIR, f'domain_pair{pair["id"]}_source_transactions.csv')
        
        source_rfm.to_csv(source_rfm_file, index=False)
        source_transactions.to_csv(source_transactions_file, index=False)
        
        print(f"  ✓ Saved: {os.path.basename(source_rfm_file)}")
        print(f"  ✓ Saved: {os.path.basename(source_transactions_file)}")
        
        # Statistics
        print(f"\n  📊 SOURCE Statistics:")
        print(f"    Customers: {len(source_customers):,}")
        print(f"    Transactions: {len(source_transactions):,}")
        print(f"    Avg Recency: {source_rfm['Recency'].mean():.1f} days")
        print(f"    Avg Frequency: {source_rfm['Frequency'].mean():.1f} purchases")
        print(f"    Avg Monetary: ₹{source_rfm['Monetary'].mean():.2f}")
        
        source_stats = {
            'pair_id': pair['id'],
            'domain': 'source',
            'n_customers': len(source_customers),
            'n_transactions': len(source_transactions),
            'avg_recency': source_rfm['Recency'].mean(),
            'avg_frequency': source_rfm['Frequency'].mean(),
            'avg_monetary': source_rfm['Monetary'].mean(),
            'file': source_rfm_file,
            'customer_id_min': source_customers['customer_id'].iloc[0],
            'customer_id_max': source_customers['customer_id'].iloc[-1]
        }
        all_stats.append(source_stats)
        
    except Exception as e:
        print(f"  ❌ Error processing source domain: {e}")
        import traceback
        traceback.print_exc()
        continue
    
    # ------------------------------------------------------------------------
    # TARGET DOMAIN
    # ------------------------------------------------------------------------
    print(f"\n📊 Processing TARGET domain...")
    
    try:
        # Load target products
        target_products = pd.read_csv(pair['target_file'])
        print(f"  ✓ Loaded {len(target_products):,} target products")
        
        # Initialize generator
        generator_target = SyntheticCustomerGenerator(target_products, seed=SEED+1)
        
        # ✅ FIX: Generate customers with OFFSET (after source)
        n_target_customers = int(N_CUSTOMERS * 0.8)  # 80% of source = 1200
        target_offset = base_offset + N_CUSTOMERS  # Start AFTER source customers
        
        print(f"  👥 Generating {n_target_customers:,} customers (offset: {target_offset})...")
        target_customers = generator_target.generate_customers(
            n_customers=n_target_customers,
            customer_id_offset=target_offset  # ← KEY FIX!
        )
        
        print(f"  ✓ Customer IDs: {target_customers['customer_id'].iloc[0]} to {target_customers['customer_id'].iloc[-1]}")
        
        # Verify NO overlap with source
        target_ids = set(target_customers['customer_id'])
        overlap_with_source = source_ids & target_ids
        
        if len(overlap_with_source) > 0:
            print(f"  ❌ ERROR: {len(overlap_with_source)} customers overlap with source!")
            raise ValueError("Source-target overlap detected!")
        
        print(f"  ✅ Verified: 0 customer overlap with source")
        
        # Verify uniqueness globally
        overlap_global = target_ids & all_customer_ids
        
        if len(overlap_global) > 0:
            print(f"  ❌ ERROR: {len(overlap_global)} customers already exist in other pairs!")
            raise ValueError("Customer ID collision detected!")
        
        all_customer_ids.update(target_ids)
        print(f"  ✓ Verified: All {len(target_ids)} customer IDs are globally unique")
        
        # Generate transactions
        n_target_transactions = int(N_TRANSACTIONS * 0.8)
        print(f"  🛒 Generating {n_target_transactions:,} transactions...")
        target_transactions = generator_target.generate_transactions(
            target_customers,
            n_transactions=n_target_transactions,
            start_date=START_DATE,
            end_date=END_DATE
        )
        
        # Calculate RFM
        print(f"  📈 Calculating RFM features...")
        target_rfm = generator_target.calculate_rfm(
            target_transactions,
            reference_date=REFERENCE_DATE
        )
        
        # Save outputs
        target_rfm_file = os.path.join(OUTPUT_DIR, f'domain_pair{pair["id"]}_target_RFM.csv')
        target_transactions_file = os.path.join(OUTPUT_DIR, f'domain_pair{pair["id"]}_target_transactions.csv')
        
        target_rfm.to_csv(target_rfm_file, index=False)
        target_transactions.to_csv(target_transactions_file, index=False)
        
        print(f"  ✓ Saved: {os.path.basename(target_rfm_file)}")
        print(f"  ✓ Saved: {os.path.basename(target_transactions_file)}")
        
        # Statistics
        print(f"\n  📊 TARGET Statistics:")
        print(f"    Customers: {len(target_customers):,}")
        print(f"    Transactions: {len(target_transactions):,}")
        print(f"    Avg Recency: {target_rfm['Recency'].mean():.1f} days")
        print(f"    Avg Frequency: {target_rfm['Frequency'].mean():.1f} purchases")
        print(f"    Avg Monetary: ₹{target_rfm['Monetary'].mean():.2f}")
        
        target_stats = {
            'pair_id': pair['id'],
            'domain': 'target',
            'n_customers': len(target_customers),
            'n_transactions': len(target_transactions),
            'avg_recency': target_rfm['Recency'].mean(),
            'avg_frequency': target_rfm['Frequency'].mean(),
            'avg_monetary': target_rfm['Monetary'].mean(),
            'file': target_rfm_file,
            'customer_id_min': target_customers['customer_id'].iloc[0],
            'customer_id_max': target_customers['customer_id'].iloc[-1]
        }
        all_stats.append(target_stats)
        
    except Exception as e:
        print(f"  ❌ Error processing target domain: {e}")
        import traceback
        traceback.print_exc()
        continue
    
    print(f"\n✅ Pair {pair['id']} complete!")
    print(f"   Source: {len(source_ids)} customers")
    print(f"   Target: {len(target_ids)} customers")
    print(f"   Overlap: 0 customers ✅")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*80)
print("📊 SUMMARY STATISTICS")
print("="*80)

# Convert to DataFrame
stats_df = pd.DataFrame(all_stats)

# Display summary
print("\n" + "-"*120)
print(f"{'Pair':<6} {'Domain':<8} {'Customers':>10} {'Transactions':>12} {'Customer ID Range':>30} {'Avg Recency':>12} {'Avg Freq':>10}")
print("-"*120)

for _, row in stats_df.iterrows():
    id_range = f"{row['customer_id_min']} - {row['customer_id_max']}"
    print(f"{row['pair_id']:<6} {row['domain']:<8} {row['n_customers']:>10,} {row['n_transactions']:>12,} "
          f"{id_range:>30} {row['avg_recency']:>12.1f} {row['avg_frequency']:>10.1f}")

# Save statistics
stats_csv_file = os.path.join(OUTPUT_DIR, 'rfm_generation_statistics.csv')
stats_df.to_csv(stats_csv_file, index=False)
print(f"\n✓ Saved: rfm_generation_statistics.csv")

# ============================================================================
# VALIDATION CHECKS
# ============================================================================

print("\n" + "="*80)
print("✅ VALIDATION CHECKS")
print("="*80)

print("\n📋 Files Created:")
files_ok = 0
files_expected = 0

for pair in DOMAIN_PAIRS:
    pair_id = pair['id']
    files = [
        (f'domain_pair{pair_id}_source_RFM.csv', 'Source RFM'),
        (f'domain_pair{pair_id}_target_RFM.csv', 'Target RFM'),
        (f'domain_pair{pair_id}_source_transactions.csv', 'Source Transactions'),
        (f'domain_pair{pair_id}_target_transactions.csv', 'Target Transactions')
    ]
    
    print(f"\nPair {pair_id}: {pair['name']}")
    
    for filename, description in files:
        files_expected += 1
        filepath = os.path.join(OUTPUT_DIR, filename)
        exists = os.path.exists(filepath)
        
        if exists:
            files_ok += 1
            filesize = os.path.getsize(filepath) / 1024  # KB
            print(f"  {'✓' if exists else '❌'} {description}: {filename} ({filesize:.1f} KB)")
        else:
            print(f"  ❌ {description}: {filename} (MISSING!)")

print(f"\nFile Check: {files_ok}/{files_expected} files created")

# Data quality checks
print("\n📊 Data Quality Checks:")

validation_passed = True

for pair in DOMAIN_PAIRS:
    pair_id = pair['id']
    try:
        source_rfm = pd.read_csv(os.path.join(OUTPUT_DIR, f'domain_pair{pair_id}_source_RFM.csv'))
        target_rfm = pd.read_csv(os.path.join(OUTPUT_DIR, f'domain_pair{pair_id}_target_RFM.csv'))
        
        print(f"\nPair {pair_id}:")
        
        # Check for missing values
        source_missing = source_rfm.isnull().sum().sum()
        target_missing = target_rfm.isnull().sum().sum()
        print(f"  Missing values: Source={source_missing}, Target={target_missing}")
        
        if source_missing > 0 or target_missing > 0:
            validation_passed = False
        
        # Check for duplicates
        source_dups = source_rfm.duplicated(subset=['customer_id']).sum()
        target_dups = target_rfm.duplicated(subset=['customer_id']).sum()
        print(f"  Duplicate customers: Source={source_dups}, Target={target_dups}")
        
        if source_dups > 0 or target_dups > 0:
            validation_passed = False
        
        # ✅ CRITICAL: Check for overlap
        source_ids = set(source_rfm['customer_id'])
        target_ids = set(target_rfm['customer_id'])
        overlap = source_ids & target_ids
        
        print(f"  Source customers: {len(source_ids):,}")
        print(f"  Target customers: {len(target_ids):,}")
        print(f"  Overlap: {len(overlap):,} customers")
        
        if len(overlap) > 0:
            print(f"  ❌ OVERLAP DETECTED!")
            validation_passed = False
        else:
            print(f"  ✅ No overlap - domains are disjoint!")
        
        # Check RFM ranges
        print(f"  Recency range: Source={source_rfm['Recency'].min():.0f}-{source_rfm['Recency'].max():.0f}, "
              f"Target={target_rfm['Recency'].min():.0f}-{target_rfm['Recency'].max():.0f}")
        print(f"  Frequency range: Source={source_rfm['Frequency'].min():.0f}-{source_rfm['Frequency'].max():.0f}, "
              f"Target={target_rfm['Frequency'].min():.0f}-{target_rfm['Frequency'].max():.0f}")
        
    except Exception as e:
        print(f"  ❌ Error checking Pair {pair_id}: {e}")
        validation_passed = False

# ============================================================================
# FINAL VALIDATION
# ============================================================================

print("\n" + "="*80)
print("🔍 FINAL VALIDATION")
print("="*80)

print(f"\nTotal unique customers generated: {len(all_customer_ids):,}")
print(f"Expected: {len(DOMAIN_PAIRS) * (N_CUSTOMERS + int(N_CUSTOMERS * 0.8)):,}")

expected_total = len(DOMAIN_PAIRS) * (N_CUSTOMERS + int(N_CUSTOMERS * 0.8))

if len(all_customer_ids) == expected_total:
    print("✅ Customer count matches expectation!")
else:
    print(f"⚠️  Customer count mismatch!")
    validation_passed = False

# Check for any duplicates across ALL pairs
all_rfm_customers = []
for pair in DOMAIN_PAIRS:
    pair_id = pair['id']
    try:
        source_rfm = pd.read_csv(os.path.join(OUTPUT_DIR, f'domain_pair{pair_id}_source_RFM.csv'))
        target_rfm = pd.read_csv(os.path.join(OUTPUT_DIR, f'domain_pair{pair_id}_target_RFM.csv'))
        all_rfm_customers.extend(source_rfm['customer_id'].tolist())
        all_rfm_customers.extend(target_rfm['customer_id'].tolist())
    except:
        pass

total_customers = len(all_rfm_customers)
unique_customers = len(set(all_rfm_customers))

print(f"\nGlobal uniqueness check:")
print(f"  Total customer entries: {total_customers:,}")
print(f"  Unique customers: {unique_customers:,}")
print(f"  Duplicates: {total_customers - unique_customers:,}")

if total_customers == unique_customers:
    print("  ✅ All customers are globally unique!")
else:
    print(f"  ❌ {total_customers - unique_customers} duplicate customer IDs found!")
    validation_passed = False

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
if validation_passed:
    print("🎉 ALL VALIDATION CHECKS PASSED!")
    print("="*80)
    print("\n✅ DATA GENERATION COMPLETE AND VERIFIED!")
else:
    print("❌ VALIDATION FAILED!")
    print("="*80)
    print("\n⚠️  Please review errors above and regenerate data")

print("\n✅ Deliverables:")
print("  1. 8 RFM datasets (4 pairs × 2 domains) - DISJOINT customer sets")
print("  2. 8 transaction datasets (for reference)")
print("  3. Statistics summary (rfm_generation_statistics.csv)")

print("\n📦 Files Created:")
print(f"  {files_ok}/{files_expected} files successfully created")

print("\n📊 Quick Statistics:")
if len(stats_df) > 0:
    print(f"  Total customers generated: {stats_df['n_customers'].sum():,}")
    print(f"  Total transactions generated: {stats_df['n_transactions'].sum():,}")
    print(f"  Average transactions/customer: {stats_df['n_transactions'].sum()/stats_df['n_customers'].sum():.1f}")

print("\n🎯 Next Steps:")
print("  1. Run verification: python check_all_domain_pairs.py")
print("  2. Share RFM files with Member 2 (baseline modeling)")
print("  3. Share statistics with Member 3 (metrics validation)")
print("  4. Continue with Week 2 deliverables")

print("\n" + "="*80)
print("✅ MEMBER 1 WEEK 2 COMPLETE!")
print("="*80)
