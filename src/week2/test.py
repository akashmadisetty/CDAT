"""
Quick diagnostic to check ALL 4 domain pairs for data leakage
Run this to see which pairs need regeneration
"""

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp

DOMAINS = {
    1: {'name': 'Cleaning & Household → Foodgrains', 'transferability': 'HIGH (0.903)'},
    2: {'name': 'Snacks → Fruits & Vegetables', 'transferability': 'MODERATE (0.548)'},
    3: {'name': 'Premium → Budget', 'transferability': 'LOW (0.715)'},
    4: {'name': 'Popular → Niche Brands', 'transferability': 'LOW-MOD (0.874)'}
}

print("="*80)
print("🔍 CHECKING ALL DOMAIN PAIRS FOR DATA LEAKAGE")
print("="*80)

results = []
critical_issues = []

for pair_id, info in DOMAINS.items():
    print(f"\n{'='*80}")
    print(f"📦 DOMAIN PAIR {pair_id}: {info['name']}")
    print(f"   Week 1 Prediction: {info['transferability']}")
    print('='*80)
    
    try:
        # Load files
        source = pd.read_csv(f'domain_pair{pair_id}_source_RFM.csv')
        target = pd.read_csv(f'domain_pair{pair_id}_target_RFM.csv')
        
        print(f"\n✅ Files loaded")
        print(f"   Source: {len(source)} customers")
        print(f"   Target: {len(target)} customers")
        
        # Check 1: Customer overlap
        source_ids = set(source['customer_id'])
        target_ids = set(target['customer_id'])
        overlap = source_ids & target_ids
        
        overlap_pct = (len(overlap) / len(source_ids)) * 100 if len(source_ids) > 0 else 0
        
        if len(overlap) > 0:
            print(f"\n❌ CRITICAL: {len(overlap)} customers overlap ({overlap_pct:.1f}% of source)")
            critical_issues.append(f"Pair {pair_id}: {len(overlap)} overlapping customers")
        else:
            print(f"\n✅ No customer overlap")
        
        # Check 2: Missing values
        source_missing = source[['Recency', 'Frequency', 'Monetary']].isnull().sum().sum()
        target_missing = target[['Recency', 'Frequency', 'Monetary']].isnull().sum().sum()
        
        if source_missing > 0 or target_missing > 0:
            print(f"\n⚠️  Missing values: Source={source_missing}, Target={target_missing}")
            critical_issues.append(f"Pair {pair_id}: Missing values detected")
        else:
            print(f"✅ No missing values")
        
        # Check 3: Distribution similarity (KS test)
        ks_r = ks_2samp(source['Recency'], target['Recency'])
        ks_f = ks_2samp(source['Frequency'], target['Frequency'])
        ks_m = ks_2samp(source['Monetary'], target['Monetary'])
        
        avg_p_value = np.mean([ks_r.pvalue, ks_f.pvalue, ks_m.pvalue])
        
        print(f"\n📊 Distribution Similarity (KS Test):")
        print(f"   Recency:   p={ks_r.pvalue:.4f} {'⚠️  TOO SIMILAR' if ks_r.pvalue > 0.05 else '✅ Different'}")
        print(f"   Frequency: p={ks_f.pvalue:.4f} {'⚠️  TOO SIMILAR' if ks_f.pvalue > 0.05 else '✅ Different'}")
        print(f"   Monetary:  p={ks_m.pvalue:.4f} {'⚠️  TOO SIMILAR' if ks_m.pvalue > 0.05 else '✅ Different'}")
        
        # Check 4: MMD calculation
        from sklearn.preprocessing import StandardScaler
        
        X_source = source[['Recency', 'Frequency', 'Monetary']].values
        X_target = target[['Recency', 'Frequency', 'Monetary']].values
        
        scaler = StandardScaler()
        X_source_scaled = scaler.fit_transform(X_source)
        X_target_scaled = scaler.transform(X_target)
        
        # Simple MMD
        def compute_mmd(X, Y):
            XX = np.dot(X, X.T)
            YY = np.dot(Y, Y.T)
            XY = np.dot(X, Y.T)
            return np.mean(XX) + np.mean(YY) - 2 * np.mean(XY)
        
        mmd = compute_mmd(X_source_scaled, X_target_scaled)
        
        print(f"\n📊 MMD Score: {mmd:.4f}")
        if mmd < 0.1:
            print(f"   ⚠️  EXTREMELY LOW - Source and target are too similar!")
            if len(overlap) > 0:
                print(f"   → Likely caused by customer overlap")
        elif mmd < 0.3:
            print(f"   ✅ LOW - Good transfer expected")
        elif mmd < 0.6:
            print(f"   ⚠️  MODERATE - Transfer needs fine-tuning")
        else:
            print(f"   ❌ HIGH - Poor transfer expected")
        
        # Store results
        results.append({
            'pair': pair_id,
            'name': info['name'],
            'source_n': len(source),
            'target_n': len(target),
            'overlap': len(overlap),
            'overlap_pct': overlap_pct,
            'missing': source_missing + target_missing,
            'mmd': mmd,
            'avg_ks_pvalue': avg_p_value,
            'status': '❌ CRITICAL' if len(overlap) > 0 else ('⚠️  WARNING' if avg_p_value > 0.1 else '✅ OK')
        })
        
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: File not found - {e}")
        results.append({
            'pair': pair_id,
            'name': info['name'],
            'status': '❌ FILE NOT FOUND'
        })
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        results.append({
            'pair': pair_id,
            'name': info['name'],
            'status': f'❌ ERROR: {str(e)}'
        })

# Summary report
print("\n" + "="*80)
print("📊 SUMMARY REPORT")
print("="*80)

df_results = pd.DataFrame(results)

# Display table
print("\n" + df_results.to_string(index=False))

# Count issues
pairs_with_overlap = sum(1 for r in results if r.get('overlap', 0) > 0)
pairs_ok = sum(1 for r in results if r.get('status', '') == '✅ OK')

print("\n" + "="*80)
print("🎯 FINAL VERDICT")
print("="*80)

if len(critical_issues) > 0:
    print(f"\n❌ REGENERATION NEEDED FOR {pairs_with_overlap} PAIR(S)")
    print("\n🚨 Critical Issues Found:")
    for issue in critical_issues:
        print(f"   • {issue}")
    
    print("\n📋 Action Items:")
    print("   1. Contact Member 1 (data generator)")
    print("   2. Share this diagnostic report")
    print("   3. Fix data generation to ensure DISJOINT customer sets")
    print("   4. Verify fix with: set(source['customer_id']) & set(target['customer_id']) == set()")
    print("   5. Re-generate ALL affected pairs")
    print("   6. Re-run training after data is fixed")
    
    print(f"\n⏱️  Estimated fix time: 2-3 hours")
    print(f"   • Member 1 fixes data: ~1-2 hours")
    print(f"   • You re-train models: ~10 minutes")
    
else:
    print(f"\n✅ ALL PAIRS ARE VALID!")
    print(f"   • {pairs_ok} pairs have proper disjoint customer sets")
    print(f"   • No data leakage detected")
    print(f"   • Safe to continue with Week 2 deliverables")
    
    print("\n💡 Note on MMD scores:")
    for r in results:
        if 'mmd' in r:
            print(f"   • Pair {r['pair']}: MMD={r['mmd']:.3f} → ", end="")
            if r['mmd'] < 0.3:
                print("Good transfer expected ✅")
            elif r['mmd'] < 0.6:
                print("Moderate transfer ⚠️")
            else:
                print("Poor transfer (expected for this pair) ❌")

print("\n" + "="*80)

# Save detailed report
df_results.to_csv('data_integrity_report.csv', index=False)
print("\n✅ Detailed report saved: data_integrity_report.csv")
print("="*80)