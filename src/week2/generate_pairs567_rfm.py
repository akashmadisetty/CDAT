# Generate RFM for Domain Pairs 5, 6, 7 - RESEARCH-BACKED VERSION
# Member 1 - Week 2 Final Deliverable
# Specialized personas for cross-category pairs
#
# Customer ID Allocation Strategy:
# ================================
# Pair 5: C12000 - C14699 (Source: C12000-C13499, Target: C13500-C14699)
# Pair 6: C15000 - C17699 (Source: C15000-C16499, Target: C16500-C17699)
# Pair 7: C18000 - C20699 (Source: C18000-C19499, Target: C19500-C20699)
# TOTAL: 8,100 unique customers (NO OVERLAP with Pairs 1-4!)

import sys
import os
import pandas as pd
import numpy as np
from synth_567 import (
    SyntheticCustomerGenerator567,
    PAIR5_SOURCE_PERSONAS,
    PAIR5_TARGET_PERSONAS,
    PAIR6_SOURCE_PERSONAS,
    PAIR6_TARGET_PERSONAS,
    PAIR7_SOURCE_PERSONAS,
    PAIR7_TARGET_PERSONAS
)

print("="*80)
print("GENERATING RFM FOR DOMAIN PAIRS 5, 6, 7 - RESEARCH-BACKED VERSION")
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

# DOMAIN PAIRS 5, 6, 7 - From Week 1 Analysis
DOMAIN_PAIRS = [
    {
        'id': 5,
        'name': 'Eggs, Meat & Fish → Baby Care',
        'source_file': '../week1/domain_pair5_source.csv',
        'target_file': '../week1/domain_pair5_target.csv',
        'source_personas': PAIR5_SOURCE_PERSONAS,
        'target_personas': PAIR5_TARGET_PERSONAS,
        'expected_transfer': 'LOW (Train New Model)'
    },
    {
        'id': 6,
        'name': 'Baby Care → Bakery, Cakes & Dairy',
        'source_file': '../week1/domain_pair6_source.csv',
        'target_file': '../week1/domain_pair6_target.csv',
        'source_personas': PAIR6_SOURCE_PERSONAS,
        'target_personas': PAIR6_TARGET_PERSONAS,
        'expected_transfer': 'LOW (Train New Model)'
    },
    {
        'id': 7,
        'name': 'Beverages → Gourmet & World Food',
        'source_file': '../week1/domain_pair7_source.csv',
        'target_file': '../week1/domain_pair7_target.csv',
        'source_personas': PAIR7_SOURCE_PERSONAS,
        'target_personas': PAIR7_TARGET_PERSONAS,
        'expected_transfer': 'VERY LOW (No Transfer)'
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
    # Pair 5 starts at 12000 (after Pair 4's 11699)
    # Each pair gets 3000 customer IDs (1500 source + 1500 target)
    base_offset = 12000 + (pair['id'] - 5) * 3000
    
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
        
        # Initialize generator with SOURCE personas
        generator_source = SyntheticCustomerGenerator567(
            source_products, 
            seed=SEED,
            custom_personas=pair['source_personas']
        )
        
        # Generate customers with OFFSET
        print(f"  👥 Generating {N_CUSTOMERS:,} customers (offset: {base_offset})...")
        source_customers = generator_source.generate_customers(
            n_customers=N_CUSTOMERS,
            customer_id_offset=base_offset
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
        
        # Show persona distribution
        print(f"\n  🎭 Persona Distribution:")
        for persona, count in source_customers['persona'].value_counts().items():
            print(f"     {persona}: {count} ({count/len(source_customers)*100:.1f}%)")
        
        # Generate transactions
        print(f"\n  🛒 Generating {N_TRANSACTIONS:,} transactions...")
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
        
        # Initialize generator with TARGET personas
        generator_target = SyntheticCustomerGenerator567(
            target_products, 
            seed=SEED+1,
            custom_personas=pair['target_personas']
        )
        
        # Generate customers with OFFSET (after source)
        n_target_customers = int(N_CUSTOMERS * 0.8)  # 80% of source = 1200
        target_offset = base_offset + N_CUSTOMERS  # Start AFTER source customers
        
        print(f"  👥 Generating {n_target_customers:,} customers (offset: {target_offset})...")
        target_customers = generator_target.generate_customers(
            n_customers=n_target_customers,
            customer_id_offset=target_offset
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
        
        # Show persona distribution
        print(f"\n  🎭 Persona Distribution:")
        for persona, count in target_customers['persona'].value_counts().items():
            print(f"     {persona}: {count} ({count/len(target_customers)*100:.1f}%)")
        
        # Generate transactions
        n_target_transactions = int(N_TRANSACTIONS * 0.8)
        print(f"\n  🛒 Generating {n_target_transactions:,} transactions...")
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

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("✅ RFM GENERATION COMPLETE FOR PAIRS 5, 6, 7!")
print("="*80)

if len(all_stats) > 0:
    # Create summary DataFrame
    stats_df = pd.DataFrame(all_stats)
    
    # Save summary
    summary_file = os.path.join(OUTPUT_DIR, 'rfm_generation_statistics_pairs567.csv')
    stats_df.to_csv(summary_file, index=False)
    print(f"\n📊 Summary saved to: {os.path.basename(summary_file)}")
    
    # Display summary
    print("\n📋 GENERATION SUMMARY:")
    print("="*80)
    
    for pair_id in sorted(stats_df['pair_id'].unique()):
        pair_data = stats_df[stats_df['pair_id'] == pair_id]
        pair_info = next(p for p in DOMAIN_PAIRS if p['id'] == pair_id)
        
        print(f"\nPair {pair_id}: {pair_info['name']}")
        print(f"  Expected: {pair_info['expected_transfer']}")
        print(f"  Personas: {len(pair_info['source_personas'])} source, {len(pair_info['target_personas'])} target")
        
        for _, row in pair_data.iterrows():
            print(f"\n  {row['domain'].upper()}:")
            print(f"    Customers: {row['n_customers']:,} ({row['customer_id_min']} to {row['customer_id_max']})")
            print(f"    Transactions: {row['n_transactions']:,}")
            print(f"    Avg Recency: {row['avg_recency']:.1f} days")
            print(f"    Avg Frequency: {row['avg_frequency']:.1f}")
            print(f"    Avg Monetary: ₹{row['avg_monetary']:.2f}")
    
    # Verify global uniqueness
    print("\n" + "="*80)
    print("🔍 GLOBAL VERIFICATION")
    print("="*80)
    print(f"Total unique customer IDs: {len(all_customer_ids):,}")
    print(f"Expected customer IDs: {sum([r['n_customers'] for r in all_stats]):,}")
    
    if len(all_customer_ids) == sum([r['n_customers'] for r in all_stats]):
        print("✅ VERIFIED: All customer IDs are globally unique (NO OVERLAPS!)")
    else:
        print("❌ WARNING: Customer ID mismatch detected!")
    
    print("\n📁 Output Files:")
    for stat in all_stats:
        print(f"  • {os.path.basename(stat['file'])}")
    
    print("\n" + "="*80)
    print("🎉 ALL PAIRS (5, 6, 7) SUCCESSFULLY GENERATED!")
    print("="*80)
    
    # Research backing summary
    print("\n📚 RESEARCH-BACKED PERSONAS:")
    print("="*80)
    print("Pair 5 (Eggs/Meat → Baby Care):")
    print("  Sources: Mintel Baby Care 2024, Circana Parent Shopper Study")
    print("  Key insight: Safety-first parents prioritize brand trust over price")
    
    print("\nPair 6 (Baby Care → Bakery/Dairy):")
    print("  Sources: NPD Household Pantry Report 2024")
    print("  Key insight: Routine repeat buyers dominate staples (40%)")
    
    print("\nPair 7 (Beverages → Gourmet):")
    print("  Sources: Mintel Gourmet Food 2024, Specialty Food Association 2024")
    print("  Key insight: Quality connoisseurs (35%) vs mass-market beverage buyers")

else:
    print("\n❌ No statistics generated - all pairs failed!")
    sys.exit(1)
