# Generate RFM for All 4 Domain Pairs
# Member 1 - Week 2 Final Deliverable
# This script processes all domain pairs and creates RFM datasets

import sys
import os
import pandas as pd
import numpy as np
from synth import SyntheticCustomerGenerator

print("="*80)
print("GENERATING RFM FOR ALL DOMAIN PAIRS")
print("Member 1 - Week 2 Deliverable")
print("="*80)

# Configuration
N_CUSTOMERS = 1500          # Customers per domain
N_TRANSACTIONS = 15000      # Transactions per domain
START_DATE = '2024-01-01'
END_DATE = '2024-06-30'
REFERENCE_DATE = '2024-07-01'
SEED = 42

# Output directory (relative path - will save in src/week2)
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Domain pairs from Week 1
DOMAIN_PAIRS = [
    {
        'id': 1,
        'name': 'Cleaning & Household → Foodgrains, Oil & Masala',
        'source_file': '../week1/domain_pair1_source_FINAL.csv',
        'target_file': '../week1/domain_pair1_target_FINAL.csv',
        'expected_transfer': 'HIGH'
    },
    {
        'id': 2,
        'name': 'Snacks & Branded Foods → Fruits & Vegetables',
        'source_file': '../week1/domain_pair2_source_FINAL.csv',
        'target_file': '../week1/domain_pair2_target_FINAL.csv',
        'expected_transfer': 'MODERATE'
    },
    {
        'id': 3,
        'name': 'Premium → Budget',
        'source_file': '../week1/domain_pair3_source_FINAL.csv',
        'target_file': '../week1/domain_pair3_target_FINAL.csv',
        'expected_transfer': 'LOW'
    },
    {
        'id': 4,
        'name': 'Popular Brands → Niche Brands',
        'source_file': '../week1/domain_pair4_source_FINAL.csv',
        'target_file': '../week1/domain_pair4_target_FINAL.csv',
        'expected_transfer': 'LOW-MODERATE'
    }
]

# Track statistics
all_stats = []

# ============================================================================
# PROCESS EACH DOMAIN PAIR
# ============================================================================

for pair in DOMAIN_PAIRS:
    print("\n" + "="*80)
    print(f"DOMAIN PAIR {pair['id']}: {pair['name']}")
    print(f"Expected Transferability: {pair['expected_transfer']}")
    print("="*80)
    
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
        
        # Generate customers
        print(f"  👥 Generating {N_CUSTOMERS:,} customers...")
        source_customers = generator_source.generate_customers(n_customers=N_CUSTOMERS)
        
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
        
        print(f"  ✓ Saved: {source_rfm_file}")
        print(f"  ✓ Saved: {source_transactions_file}")
        
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
            'file': source_rfm_file
        }
        all_stats.append(source_stats)
        
    except Exception as e:
        print(f"  ❌ Error processing source domain: {e}")
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
        
        # Generate customers (slightly fewer for target)
        n_target_customers = int(N_CUSTOMERS * 0.8)  # 80% of source
        print(f"  👥 Generating {n_target_customers:,} customers...")
        target_customers = generator_target.generate_customers(n_customers=n_target_customers)
        
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
        
        print(f"  ✓ Saved: {target_rfm_file}")
        print(f"  ✓ Saved: {target_transactions_file}")
        
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
            'file': target_rfm_file
        }
        all_stats.append(target_stats)
        
    except Exception as e:
        print(f"  ❌ Error processing target domain: {e}")
        continue
    
    print(f"\n✅ Pair {pair['id']} complete!")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*80)
print("📊 SUMMARY STATISTICS")
print("="*80)

# Convert to DataFrame
stats_df = pd.DataFrame(all_stats)

# Display summary
print("\n" + "-"*80)
print(f"{'Pair':<6} {'Domain':<8} {'Customers':>10} {'Transactions':>12} {'Avg Recency':>12} {'Avg Freq':>10} {'Avg Monetary':>14}")
print("-"*80)

for _, row in stats_df.iterrows():
    print(f"{row['pair_id']:<6} {row['domain']:<8} {row['n_customers']:>10,} {row['n_transactions']:>12,} "
          f"{row['avg_recency']:>12.1f} {row['avg_frequency']:>10.1f} ₹{row['avg_monetary']:>12.2f}")

# Save statistics
stats_csv_file = os.path.join(OUTPUT_DIR, 'rfm_generation_statistics.csv')
stats_df.to_csv(stats_csv_file, index=False)
print("\n✓ Saved: rfm_generation_statistics.csv")

# ============================================================================
# VALIDATION CHECKS
# ============================================================================

print("\n" + "="*80)
print("✅ VALIDATION CHECKS")
print("="*80)

print("\n📋 Files Created:")
for pair in DOMAIN_PAIRS:
    pair_id = pair['id']
    source_file = os.path.join(OUTPUT_DIR, f'domain_pair{pair_id}_source_RFM.csv')
    target_file = os.path.join(OUTPUT_DIR, f'domain_pair{pair_id}_target_RFM.csv')
    
    source_exists = os.path.exists(source_file)
    target_exists = os.path.exists(target_file)
    
    print(f"\nPair {pair_id}: {pair['name']}")
    print(f"  Source RFM: {'✓' if source_exists else '❌'} {source_file}")
    print(f"  Target RFM: {'✓' if target_exists else '❌'} {target_file}")

# Data quality checks
print("\n📊 Data Quality Checks:")

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
        
        # Check RFM ranges
        print(f"  Recency range: Source={source_rfm['Recency'].min():.0f}-{source_rfm['Recency'].max():.0f}, "
              f"Target={target_rfm['Recency'].min():.0f}-{target_rfm['Recency'].max():.0f}")
        print(f"  Frequency range: Source={source_rfm['Frequency'].min():.0f}-{source_rfm['Frequency'].max():.0f}, "
              f"Target={target_rfm['Frequency'].min():.0f}-{target_rfm['Frequency'].max():.0f}")
        
        # Check for duplicates
        source_dups = source_rfm.duplicated(subset=['customer_id']).sum()
        target_dups = target_rfm.duplicated(subset=['customer_id']).sum()
        print(f"  Duplicate customers: Source={source_dups}, Target={target_dups}")
        
        if source_missing == 0 and target_missing == 0 and source_dups == 0 and target_dups == 0:
            print(f"  ✅ Quality check passed!")
        else:
            print(f"  ⚠️  Quality issues detected")
            
    except Exception as e:
        print(f"  ❌ Error checking Pair {pair_id}: {e}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("🎉 RFM GENERATION COMPLETE!")
print("="*80)

print("\n✅ Deliverables:")
print("  1. 8 RFM datasets (4 pairs × 2 domains)")
print("  2. 8 transaction datasets (for reference)")
print("  3. Statistics summary (rfm_generation_statistics.csv)")

print("\n📦 Files Created:")
total_files = 0
for pair in DOMAIN_PAIRS:
    pair_id = pair['id']
    files = [
        os.path.join(OUTPUT_DIR, f'domain_pair{pair_id}_source_RFM.csv'),
        os.path.join(OUTPUT_DIR, f'domain_pair{pair_id}_target_RFM.csv'),
        os.path.join(OUTPUT_DIR, f'domain_pair{pair_id}_source_transactions.csv'),
        os.path.join(OUTPUT_DIR, f'domain_pair{pair_id}_target_transactions.csv')
    ]
    for f in files:
        if os.path.exists(f):
            total_files += 1
            print(f"  ✓ {f}")

print(f"\nTotal files: {total_files} (expected: 16)")

print("\n🎯 Next Steps:")
print("  1. Share RFM files with Member 2 (baseline modeling)")
print("  2. Share statistics with Member 3 (metrics validation)")
print("  3. Create documentation of synthetic data generation process")
print("  4. Prepare for Week 2 team sync")

print("\n📊 Quick Statistics:")
print(f"  Total customers generated: {stats_df['n_customers'].sum():,}")
print(f"  Total transactions generated: {stats_df['n_transactions'].sum():,}")
print(f"  Average RFM quality: {stats_df['n_transactions'].mean()/stats_df['n_customers'].mean():.1f} transactions/customer")

print("\n" + "="*80)
print("✅ MEMBER 1 WEEK 2 COMPLETE!")
print("="*80)
print("\n🎉 Great work! Ready for experiments in Week 3!")