"""
QUICK START GUIDE - RFM Generation Fix
=======================================

Run these commands in order:
"""

# 1. Navigate to week2 directory
print("Step 1: Navigate to week2 directory")
print("cd d:\\Akash\\B.Tech\\5th Sem\\ADA\\CDAT\\src\\week2")
print()

# 2. Generate RFM data for all 4 pairs
print("Step 2: Generate RFM data (15-20 minutes)")
print("python generate_rfm_all_pairs_FIXED.py")
print()

# 3. Verify success
print("Step 3: Verify data quality (2 minutes)")
print("python check_all_domain_pairs.py")
print()

# 4. Expected output
print("="*80)
print("EXPECTED OUTPUT FROM VERIFICATION:")
print("="*80)
print("""
CHECKING PAIR 1
================================================================================
Source customers: 1,500
  Range: C00000 to C01499
Target customers: 1,200
  Range: C01500 to C02699
Within-pair overlap: 0 customers
✅ OK: Source and target are disjoint
✅ OK: No duplicates in source
✅ OK: No duplicates in target
✅ OK: No overlap with other pairs

[... similar for pairs 2, 3, 4 ...]

FINAL VERIFICATION SUMMARY
================================================================================
Total unique customers across all pairs: 10,800
Expected: 10,800
✅ Customer count matches expectation!

================================================================================
✅ ALL VALIDATION CHECKS PASSED!
================================================================================
""")

print("="*80)
print("FILES YOU'LL GET:")
print("="*80)
files = [
    "domain_pair1_source_RFM.csv (1,500 customers)",
    "domain_pair1_target_RFM.csv (1,200 customers)",
    "domain_pair1_source_transactions.csv",
    "domain_pair1_target_transactions.csv",
    "",
    "domain_pair2_source_RFM.csv (1,500 customers)",
    "domain_pair2_target_RFM.csv (1,200 customers)",
    "domain_pair2_source_transactions.csv",
    "domain_pair2_target_transactions.csv",
    "",
    "domain_pair3_source_RFM.csv (1,500 customers)",
    "domain_pair3_target_RFM.csv (1,200 customers)",
    "domain_pair3_source_transactions.csv",
    "domain_pair3_target_transactions.csv",
    "",
    "domain_pair4_source_RFM.csv (1,500 customers)",
    "domain_pair4_target_RFM.csv (1,200 customers)",
    "domain_pair4_source_transactions.csv",
    "domain_pair4_target_transactions.csv",
    "",
    "rfm_generation_statistics.csv",
]

for f in files:
    if f:
        print(f"  ✓ {f}")
    else:
        print()

print()
print("="*80)
print("TROUBLESHOOTING:")
print("="*80)
print("""
If you get an error:

1. "No module named 'synth_FIXED'":
   - Make sure synth_FIXED.py is in src/week2/
   - Run from the week2 directory

2. "File not found: domain_pair*_source_FINAL.csv":
   - Check that Week 1 files exist in src/week1/
   - Update file paths in generate_rfm_all_pairs_FIXED.py if needed

3. "Validation failed":
   - Delete all domain_pair*_RFM.csv files
   - Re-run generate_rfm_all_pairs_FIXED.py
   - Re-run check_all_domain_pairs.py

For detailed explanation, see: FIX_DOCUMENTATION.md
""")

print("="*80)
print("READY TO START!")
print("="*80)
print()
print("Total time needed: ~30 minutes")
print("  - Generation: 15-20 min")
print("  - Verification: 2 min")
print("  - Re-training models (if needed): 10 min")
print()
print("Good luck! 🚀")
