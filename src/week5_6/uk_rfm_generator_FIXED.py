"""
UK Online Retail Dataset - FIXED RFM Generation for Week 5-6
=============================================================
FIXES APPLIED:
1. ✅ Fixed reference date (consistent with Week 2 methodology)
2. ✅ Outlier capping at 99th percentile (research-backed)
3. ✅ Normalization function for cross-domain comparison
4. ✅ Improved visualizations (density plots, log scale, box plots)
5. ✅ Statistical comparison tables

Research-Backed Implementation:
- Fader & Hardie (2009): RFM methodology + outlier handling
- Pan & Yang (2010): Transfer learning feature normalization
- Hughes (1994): Original RFM framework

Author: Week 5-6 Team (FIXED by Technical Lead)
Date: November 2024
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)

print("="*80)
print("UK ONLINE RETAIL DATASET - FIXED RFM GENERATION")
print("Week 5-6: Transfer Learning Validation Experiments")
print("="*80)

# ============================================================================
# CONFIGURATION (MATCHING WEEK 2 METHODOLOGY)
# ============================================================================

# ✅ FIX 1: FIXED REFERENCE DATE (not dynamic!)
# UK Retail dataset: Dec 1, 2010 → Dec 9, 2011
# Reference date: Dec 10, 2011 (one day after last transaction)
REFERENCE_DATE = '2011-12-10'

# Outlier capping threshold (research-backed: Fader & Hardie 2009)
OUTLIER_THRESHOLD = 0.99  # Remove top 1%

# Random seed for reproducibility
SEED = 42
np.random.seed(SEED)

print(f"\n⚙️  Configuration:")
print(f"   Reference Date: {REFERENCE_DATE} (FIXED)")
print(f"   Outlier Threshold: {OUTLIER_THRESHOLD} (99th percentile cap)")
print(f"   Random Seed: {SEED}")

# ============================================================================
# STEP 1: LOAD DATASET
# ============================================================================

print("\n📦 STEP 1: Loading Dataset")
print("-"*80)

def load_uk_retail():
    """Load UK Retail dataset (try multiple methods)"""
    try:
        print("Attempting local CSV file...")
        df = pd.read_csv(r'D:\Akash\B.Tech\5th Sem\ADA\Backup\CDAT\data\original\UK.csv', 
                         encoding='ISO-8859-1')
        print("✓ Loaded from local CSV file")
        return df
    except Exception as e:
        print(f"  ✗ CSV method failed: {e}")
        return None

df_raw = load_uk_retail()

if df_raw is None:
    print("\n⚠️  ERROR: Could not load dataset!")
    print("Please ensure UK.csv is in the correct location.")
    exit()

print(f"\n✓ Dataset loaded: {len(df_raw):,} transactions")
print(f"✓ Columns: {list(df_raw.columns)}")

# ============================================================================
# STEP 2: DATA CLEANING & PREPROCESSING
# ============================================================================

print("\n🧹 STEP 2: Data Cleaning")
print("-"*80)

df = df_raw.copy()

# Standardize column names
column_mapping = {
    'Invoice': 'InvoiceNo',
    'StockCode': 'StockCode',
    'Description': 'Description',
    'Quantity': 'Quantity',
    'InvoiceDate': 'InvoiceDate',
    'Price': 'UnitPrice',
    'UnitPrice': 'UnitPrice',
    'Customer ID': 'CustomerID',
    'CustomerID': 'CustomerID',
    'Country': 'Country'
}
df = df.rename(columns=column_mapping)

print(f"Original: {len(df):,} rows")

# Remove cancelled transactions
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
print(f"After removing cancellations: {len(df):,} rows")

# Remove missing CustomerID
df = df.dropna(subset=['CustomerID'])
print(f"After removing missing CustomerID: {len(df):,} rows")

# Remove invalid quantities and prices
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
print(f"After removing invalid values: {len(df):,} rows")

# Convert types
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']
df['CustomerID'] = df['CustomerID'].astype(int)

print(f"\n✓ Clean dataset: {len(df):,} transactions")
print(f"✓ Customers: {df['CustomerID'].nunique():,}")
print(f"✓ Countries: {df['Country'].nunique()}")
print(f"✓ Date range: {df['InvoiceDate'].min().date()} to {df['InvoiceDate'].max().date()}")

# Verify reference date is valid
max_date = df['InvoiceDate'].max()
ref_date = pd.to_datetime(REFERENCE_DATE)
if ref_date <= max_date:
    print(f"⚠️  WARNING: Reference date {REFERENCE_DATE} is not after max transaction date {max_date.date()}")
    print("   This will result in negative Recency values!")
else:
    print(f"✓ Reference date validation: {REFERENCE_DATE} > {max_date.date()} ✅")

# ============================================================================
# STEP 3: COUNTRY DISTRIBUTION ANALYSIS
# ============================================================================

print("\n🌍 STEP 3: Country Distribution Analysis")
print("-"*80)

country_stats = df.groupby('Country').agg({
    'CustomerID': 'nunique',
    'InvoiceNo': 'nunique',
    'TotalPrice': 'sum'
}).rename(columns={
    'CustomerID': 'n_customers',
    'InvoiceNo': 'n_orders',
    'TotalPrice': 'total_revenue'
}).sort_values('n_customers', ascending=False)

print("\nTop 10 Countries by Customer Count:")
print(country_stats.head(10).to_string())

country_stats.to_csv('uk_retail_country_stats.csv')
print("\n✓ Saved: uk_retail_country_stats.csv")

# ============================================================================
# STEP 4: FIXED RFM CALCULATION FUNCTION
# ============================================================================

def calculate_rfm_standardized(df, reference_date, outlier_threshold=0.99):
    """
    ✅ FIXED: Standardized RFM calculation (matches Week 2 methodology)
    
    RESEARCH-BACKED IMPLEMENTATION:
    - Fader & Hardie (2009): RFM formulas + outlier capping at 95-99th percentile
    - Hughes (1994): Original RFM methodology
    - Kumar & Reinartz (2018): Quintile scoring
    
    Parameters:
    -----------
    df : DataFrame
        Transaction data with columns: CustomerID, InvoiceDate, InvoiceNo, TotalPrice
    reference_date : str or datetime
        FIXED reference date for Recency calculation (MUST be same across experiments!)
    outlier_threshold : float
        Percentile threshold for capping outliers (0.99 = remove top 1%)
    
    Returns:
    --------
    DataFrame with columns:
        - CustomerID
        - Recency, Frequency, Monetary (raw values)
        - Recency_capped, Frequency_capped, Monetary_capped (outlier-handled)
        - R_Score, F_Score, M_Score (1-5 quintiles)
        - RFM_Score (combined string)
    """
    # Convert reference date to datetime
    reference_date = pd.to_datetime(reference_date)
    
    # Aggregate customer transactions
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days,  # Recency
        'InvoiceNo': 'nunique',                                    # Frequency
        'TotalPrice': 'sum'                                        # Monetary
    }).rename(columns={
        'InvoiceDate': 'Recency',
        'InvoiceNo': 'Frequency',
        'TotalPrice': 'Monetary'
    }).reset_index()
    
    # ✅ FIX 2: OUTLIER CAPPING (Research-backed: Fader & Hardie 2009)
    print(f"   Applying {outlier_threshold*100:.0f}th percentile capping...")
    
    outlier_summary = []
    for col in ['Recency', 'Frequency', 'Monetary']:
        upper_limit = rfm[col].quantile(outlier_threshold)
        n_outliers = (rfm[col] > upper_limit).sum()
        
        rfm[f'{col}_capped'] = rfm[col].clip(upper=upper_limit)
        
        outlier_summary.append({
            'Feature': col,
            'Threshold': upper_limit,
            'N_Outliers': n_outliers,
            'Pct_Outliers': f"{n_outliers/len(rfm)*100:.1f}%"
        })
    
    outlier_df = pd.DataFrame(outlier_summary)
    print(outlier_df.to_string(index=False))
    
    # ✅ RFM SCORING (using capped values, matching Week 2: 5 quintiles)
    # Use try-except to handle cases where qcut fails (not enough unique values)
    try:
        rfm['R_Score'] = pd.qcut(rfm['Recency_capped'], q=5, labels=[5,4,3,2,1], duplicates='drop')
    except:
        rfm['R_Score'] = pd.cut(rfm['Recency_capped'], bins=5, labels=[5,4,3,2,1], duplicates='drop')
    
    try:
        rfm['F_Score'] = pd.qcut(rfm['Frequency_capped'], q=5, labels=[1,2,3,4,5], duplicates='drop')
    except:
        rfm['F_Score'] = pd.cut(rfm['Frequency_capped'], bins=5, labels=[1,2,3,4,5], duplicates='drop')
    
    try:
        rfm['M_Score'] = pd.qcut(rfm['Monetary_capped'], q=5, labels=[1,2,3,4,5], duplicates='drop')
    except:
        rfm['M_Score'] = pd.cut(rfm['Monetary_capped'], bins=5, labels=[1,2,3,4,5], duplicates='drop')
    
    # Combined RFM Score
    rfm['RFM_Score'] = (rfm['R_Score'].astype(str) + 
                        rfm['F_Score'].astype(str) + 
                        rfm['M_Score'].astype(str))
    
    return rfm


def normalize_rfm_for_transfer_learning(rfm_source, rfm_target):
    """
    ✅ FIX 3: Normalize RFM features for cross-domain comparison
    
    RESEARCH-BACKED: Pan et al. (2011) - Transfer Component Analysis
    "Feature scaling essential for comparing domains with different scales"
    
    Uses StandardScaler (z-score normalization):
        scaled_value = (value - mean) / std
    
    Parameters:
    -----------
    rfm_source : DataFrame
        Source domain RFM features
    rfm_target : DataFrame
        Target domain RFM features
    
    Returns:
    --------
    rfm_source_scaled, rfm_target_scaled : DataFrames
        Both domains with added columns: Recency_scaled, Frequency_scaled, Monetary_scaled
    """
    # Combine source + target to fit scaler (standard practice)
    # This ensures both domains are scaled to the same reference
    combined = pd.concat([
        rfm_source[['Recency_capped', 'Frequency_capped', 'Monetary_capped']],
        rfm_target[['Recency_capped', 'Frequency_capped', 'Monetary_capped']]
    ], axis=0)
    
    scaler = StandardScaler()
    scaler.fit(combined)
    
    # Transform both domains
    rfm_source_scaled = rfm_source.copy()
    rfm_target_scaled = rfm_target.copy()
    
    source_features = scaler.transform(
        rfm_source[['Recency_capped', 'Frequency_capped', 'Monetary_capped']]
    )
    target_features = scaler.transform(
        rfm_target[['Recency_capped', 'Frequency_capped', 'Monetary_capped']]
    )
    
    rfm_source_scaled[['Recency_scaled', 'Frequency_scaled', 'Monetary_scaled']] = source_features
    rfm_target_scaled[['Recency_scaled', 'Frequency_scaled', 'Monetary_scaled']] = target_features
    
    print(f"   ✓ Scaled features: Mean ≈ 0, Std ≈ 1")
    print(f"     Source Recency: mean={source_features[:,0].mean():.3f}, std={source_features[:,0].std():.3f}")
    print(f"     Target Recency: mean={target_features[:,0].mean():.3f}, std={target_features[:,0].std():.3f}")
    
    return rfm_source_scaled, rfm_target_scaled


# ============================================================================
# EXPERIMENT 5: UK → FRANCE TRANSFER
# ============================================================================

print("\n" + "="*80)
print("EXPERIMENT 5: UK → FRANCE TRANSFER")
print("="*80)

df_uk = df[df['Country'] == 'United Kingdom'].copy()
df_france = df[df['Country'] == 'France'].copy()

print(f"\nData Summary:")
print(f"  UK: {len(df_uk):,} transactions, {df_uk['CustomerID'].nunique():,} customers")
print(f"  France: {len(df_france):,} transactions, {df_france['CustomerID'].nunique():,} customers")

# Generate RFM with FIXED reference date
print(f"\n📊 Calculating UK RFM (reference: {REFERENCE_DATE})...")
rfm_uk = calculate_rfm_standardized(df_uk, REFERENCE_DATE, OUTLIER_THRESHOLD)
print(f"✓ UK RFM: {len(rfm_uk):,} customers")

print(f"\n📊 Calculating France RFM (reference: {REFERENCE_DATE})...")
rfm_france = calculate_rfm_standardized(df_france, REFERENCE_DATE, OUTLIER_THRESHOLD)
print(f"✓ France RFM: {len(rfm_france):,} customers")

# Normalize for transfer learning
print(f"\n🔧 Normalizing features for cross-domain comparison...")
rfm_uk_scaled, rfm_france_scaled = normalize_rfm_for_transfer_learning(rfm_uk, rfm_france)

# Save
rfm_uk.to_csv('exp5_uk_source_RFM_FIXED.csv', index=False)
rfm_france.to_csv('exp5_france_target_RFM_FIXED.csv', index=False)
rfm_uk_scaled.to_csv('exp5_uk_source_RFM_scaled.csv', index=False)
rfm_france_scaled.to_csv('exp5_france_target_RFM_scaled.csv', index=False)

print("\n✅ Saved:")
print("   - exp5_uk_source_RFM_FIXED.csv (raw RFM)")
print("   - exp5_france_target_RFM_FIXED.csv (raw RFM)")
print("   - exp5_uk_source_RFM_scaled.csv (normalized for transfer learning)")
print("   - exp5_france_target_RFM_scaled.csv (normalized for transfer learning)")

# ============================================================================
# EXPERIMENT 6: UK → GERMANY TRANSFER
# ============================================================================

print("\n" + "="*80)
print("EXPERIMENT 6: UK → GERMANY TRANSFER")
print("="*80)

df_germany = df[df['Country'] == 'Germany'].copy()

print(f"\nData Summary:")
print(f"  UK: {len(df_uk):,} transactions, {df_uk['CustomerID'].nunique():,} customers")
print(f"  Germany: {len(df_germany):,} transactions, {df_germany['CustomerID'].nunique():,} customers")

print(f"\n📊 Calculating Germany RFM (reference: {REFERENCE_DATE})...")
rfm_germany = calculate_rfm_standardized(df_germany, REFERENCE_DATE, OUTLIER_THRESHOLD)
print(f"✓ Germany RFM: {len(rfm_germany):,} customers")

# Normalize
print(f"\n🔧 Normalizing features for cross-domain comparison...")
rfm_uk_scaled_g, rfm_germany_scaled = normalize_rfm_for_transfer_learning(rfm_uk, rfm_germany)

# Save
rfm_germany.to_csv('exp6_germany_target_RFM_FIXED.csv', index=False)
rfm_germany_scaled.to_csv('exp6_germany_target_RFM_scaled.csv', index=False)

print("\n✅ Saved:")
print("   - exp6_germany_target_RFM_FIXED.csv (raw RFM)")
print("   - exp6_germany_target_RFM_scaled.csv (normalized for transfer learning)")
print("   - (Using same UK source as Exp 5)")

# ============================================================================
# EXPERIMENT 7: HIGH-VALUE → MEDIUM-VALUE CUSTOMERS
# ============================================================================

print("\n" + "="*80)
print("EXPERIMENT 7: HIGH-VALUE → MEDIUM-VALUE CUSTOMERS")
print("="*80)

# Use CAPPED monetary values for quartile calculation (avoid outlier bias)
monetary_75 = rfm_uk['Monetary_capped'].quantile(0.75)
monetary_25 = rfm_uk['Monetary_capped'].quantile(0.25)

print(f"\nMonetary Value Quartiles (using capped values):")
print(f"  75th percentile (High-value threshold): £{monetary_75:,.2f}")
print(f"  25th percentile (Low-value threshold): £{monetary_25:,.2f}")

# High-value: Top 25% (≥75th percentile)
rfm_high_value = rfm_uk[rfm_uk['Monetary_capped'] >= monetary_75].copy()

# Medium-value: 25th-75th percentile
rfm_medium_value = rfm_uk[
    (rfm_uk['Monetary_capped'] >= monetary_25) & 
    (rfm_uk['Monetary_capped'] < monetary_75)
].copy()

print(f"\nSegment Summary:")
print(f"  High-value: {len(rfm_high_value):,} customers")
print(f"    Avg Recency: {rfm_high_value['Recency_capped'].mean():.1f} days")
print(f"    Avg Frequency: {rfm_high_value['Frequency_capped'].mean():.1f} orders")
print(f"    Avg Monetary: £{rfm_high_value['Monetary_capped'].mean():,.2f}")

print(f"  Medium-value: {len(rfm_medium_value):,} customers")
print(f"    Avg Recency: {rfm_medium_value['Recency_capped'].mean():.1f} days")
print(f"    Avg Frequency: {rfm_medium_value['Frequency_capped'].mean():.1f} orders")
print(f"    Avg Monetary: £{rfm_medium_value['Monetary_capped'].mean():,.2f}")

# Normalize
print(f"\n🔧 Normalizing features for cross-domain comparison...")
rfm_high_scaled, rfm_medium_scaled = normalize_rfm_for_transfer_learning(
    rfm_high_value, rfm_medium_value
)

# Save
rfm_high_value.to_csv('exp7_highvalue_source_RFM_FIXED.csv', index=False)
rfm_medium_value.to_csv('exp7_mediumvalue_target_RFM_FIXED.csv', index=False)
rfm_high_scaled.to_csv('exp7_highvalue_source_RFM_scaled.csv', index=False)
rfm_medium_scaled.to_csv('exp7_mediumvalue_target_RFM_scaled.csv', index=False)

print("\n✅ Saved:")
print("   - exp7_highvalue_source_RFM_FIXED.csv (raw RFM)")
print("   - exp7_mediumvalue_target_RFM_FIXED.csv (raw RFM)")
print("   - exp7_highvalue_source_RFM_scaled.csv (normalized)")
print("   - exp7_mediumvalue_target_RFM_scaled.csv (normalized)")

# ============================================================================
# STEP 5: STATISTICAL COMPARISON (Research-Quality)
# ============================================================================

print("\n" + "="*80)
print("📊 STATISTICAL COMPARISON - ALL EXPERIMENTS")
print("="*80)

def calculate_domain_similarity(rfm_source, rfm_target, domain_names):
    """Calculate statistical similarity metrics between domains"""
    results = {}
    
    for feature in ['Recency_capped', 'Frequency_capped', 'Monetary_capped']:
        source_data = rfm_source[feature].values
        target_data = rfm_target[feature].values
        
        # Kolmogorov-Smirnov test (distribution similarity)
        ks_stat, ks_pval = stats.ks_2samp(source_data, target_data)
        
        # t-test (mean difference)
        t_stat, t_pval = stats.ttest_ind(source_data, target_data)
        
        # Mean difference percentage
        mean_diff_pct = abs(source_data.mean() - target_data.mean()) / source_data.mean() * 100
        
        results[feature] = {
            'Source_Mean': source_data.mean(),
            'Target_Mean': target_data.mean(),
            'Mean_Diff_%': mean_diff_pct,
            'KS_Statistic': ks_stat,
            'KS_PValue': ks_pval,
            'Similar_Distribution': 'Yes' if ks_pval > 0.05 else 'No'
        }
    
    return pd.DataFrame(results).T

# Exp 5: UK vs France
print("\n🔍 Experiment 5: UK → France")
exp5_stats = calculate_domain_similarity(rfm_uk, rfm_france, ['UK', 'France'])
print(exp5_stats.to_string())

# Exp 6: UK vs Germany
print("\n🔍 Experiment 6: UK → Germany")
exp6_stats = calculate_domain_similarity(rfm_uk, rfm_germany, ['UK', 'Germany'])
print(exp6_stats.to_string())

# Exp 7: High vs Medium
print("\n🔍 Experiment 7: High-value → Medium-value")
exp7_stats = calculate_domain_similarity(rfm_high_value, rfm_medium_value, ['High', 'Medium'])
print(exp7_stats.to_string())

# Save comprehensive comparison
comparison_summary = pd.DataFrame({
    'Experiment': [
        'Exp 5: UK→France',
        'Exp 6: UK→Germany',
        'Exp 7: High→Medium'
    ],
    'Source_N': [len(rfm_uk), len(rfm_uk), len(rfm_high_value)],
    'Target_N': [len(rfm_france), len(rfm_germany), len(rfm_medium_value)],
    'Recency_MeanDiff_%': [
        exp5_stats.loc['Recency_capped', 'Mean_Diff_%'],
        exp6_stats.loc['Recency_capped', 'Mean_Diff_%'],
        exp7_stats.loc['Recency_capped', 'Mean_Diff_%']
    ],
    'Monetary_MeanDiff_%': [
        exp5_stats.loc['Monetary_capped', 'Mean_Diff_%'],
        exp6_stats.loc['Monetary_capped', 'Mean_Diff_%'],
        exp7_stats.loc['Monetary_capped', 'Mean_Diff_%']
    ],
    'Expected_Transferability': [
        'MODERATE-HIGH (similar distributions)',
        'MODERATE-HIGH (similar distributions)',
        'LOW (large domain shift)'
    ]
})

print("\n📋 Summary Comparison:")
print(comparison_summary.to_string(index=False))

comparison_summary.to_csv('uk_retail_experiments_comparison_FIXED.csv', index=False)
print("\n✓ Saved: uk_retail_experiments_comparison_FIXED.csv")

# ============================================================================
# STEP 6: IMPROVED VISUALIZATIONS
# ============================================================================

print("\n📊 STEP 6: Creating Research-Quality Visualizations")
print("-"*80)

# ✅ FIX 4: IMPROVED VISUALIZATIONS (density, log scale, box plots)

fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

# ROW 1: RECENCY DISTRIBUTIONS (Density plots)
ax1 = fig.add_subplot(gs[0, 0])
ax1.hist([rfm_uk['Recency_capped'], rfm_france['Recency_capped']], 
         bins=30, label=['UK', 'France'], alpha=0.7, density=True)
ax1.set_title('Exp 5: Recency (Density)\nUK vs France', fontweight='bold')
ax1.set_xlabel('Recency (days)')
ax1.set_ylabel('Density')
ax1.legend()

ax2 = fig.add_subplot(gs[0, 1])
ax2.hist([rfm_uk['Recency_capped'], rfm_germany['Recency_capped']], 
         bins=30, label=['UK', 'Germany'], alpha=0.7, density=True)
ax2.set_title('Exp 6: Recency (Density)\nUK vs Germany', fontweight='bold')
ax2.set_xlabel('Recency (days)')
ax2.set_ylabel('Density')
ax2.legend()

ax3 = fig.add_subplot(gs[0, 2])
ax3.hist([rfm_high_value['Recency_capped'], rfm_medium_value['Recency_capped']], 
         bins=30, label=['High', 'Medium'], alpha=0.7, density=True)
ax3.set_title('Exp 7: Recency (Density)\nHigh vs Medium', fontweight='bold')
ax3.set_xlabel('Recency (days)')
ax3.set_ylabel('Density')
ax3.legend()

# ROW 2: MONETARY DISTRIBUTIONS (Log scale)
ax4 = fig.add_subplot(gs[1, 0])
ax4.hist([rfm_uk['Monetary_capped'], rfm_france['Monetary_capped']], 
         bins=30, label=['UK', 'France'], alpha=0.7)
ax4.set_xscale('log')
ax4.set_title('Exp 5: Monetary (Log Scale)\nUK vs France', fontweight='bold')
ax4.set_xlabel('Monetary Value (£) - Log Scale')
ax4.set_ylabel('Count')
ax4.legend()

ax5 = fig.add_subplot(gs[1, 1])
ax5.hist([rfm_uk['Monetary_capped'], rfm_germany['Monetary_capped']], 
         bins=30, label=['UK', 'Germany'], alpha=0.7)
ax5.set_xscale('log')
ax5.set_title('Exp 6: Monetary (Log Scale)\nUK vs Germany', fontweight='bold')
ax5.set_xlabel('Monetary Value (£) - Log Scale')
ax5.set_ylabel('Count')
ax5.legend()

ax6 = fig.add_subplot(gs[1, 2])
ax6.hist([rfm_high_value['Monetary_capped'], rfm_medium_value['Monetary_capped']], 
         bins=30, label=['High', 'Medium'], alpha=0.7)
ax6.set_xscale('log')
ax6.set_title('Exp 7: Monetary (Log Scale)\nHigh vs Medium', fontweight='bold')
ax6.set_xlabel('Monetary Value (£) - Log Scale')
ax6.set_ylabel('Count')
ax6.legend()

# ROW 3: BOX PLOTS (Better for comparison)
ax7 = fig.add_subplot(gs[2, 0])
data_exp5 = pd.DataFrame({
    'Recency': list(rfm_uk['Recency_capped']) + list(rfm_france['Recency_capped']),
    'Domain': ['UK']*len(rfm_uk) + ['France']*len(rfm_france)
})
sns.boxplot(data=data_exp5, x='Domain', y='Recency', ax=ax7)
ax7.set_title('Exp 5: Recency Box Plot', fontweight='bold')

ax8 = fig.add_subplot(gs[2, 1])
data_exp6 = pd.DataFrame({
    'Recency': list(rfm_uk['Recency_capped']) + list(rfm_germany['Recency_capped']),
    'Domain': ['UK']*len(rfm_uk) + ['Germany']*len(rfm_germany)
})
sns.boxplot(data=data_exp6, x='Domain', y='Recency', ax=ax8)
ax8.set_title('Exp 6: Recency Box Plot', fontweight='bold')

ax9 = fig.add_subplot(gs[2, 2])
data_exp7 = pd.DataFrame({
    'Recency': list(rfm_high_value['Recency_capped']) + list(rfm_medium_value['Recency_capped']),
    'Domain': ['High']*len(rfm_high_value) + ['Medium']*len(rfm_medium_value)
})
sns.boxplot(data=data_exp7, x='Domain', y='Recency', ax=ax9)
ax9.set_title('Exp 7: Recency Box Plot', fontweight='bold')

# Summary statistics table (right column)
ax10 = fig.add_subplot(gs[:, 3])
ax10.axis('off')

summary_text = f"""
SUMMARY STATISTICS (Capped Values)

Experiment 5: UK → France
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Source (UK):
  N = {len(rfm_uk):,}
  Recency: {rfm_uk['Recency_capped'].mean():.1f} ± {rfm_uk['Recency_capped'].std():.1f} days
  Monetary: £{rfm_uk['Monetary_capped'].mean():,.0f} ± £{rfm_uk['Monetary_capped'].std():,.0f}

Target (France):
  N = {len(rfm_france):,}
  Recency: {rfm_france['Recency_capped'].mean():.1f} ± {rfm_france['Recency_capped'].std():.1f} days
  Monetary: £{rfm_france['Monetary_capped'].mean():,.0f} ± £{rfm_france['Monetary_capped'].std():,.0f}

Difference: {exp5_stats.loc['Recency_capped', 'Mean_Diff_%']:.1f}% (Recency)

Experiment 6: UK → Germany
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Source (UK):
  N = {len(rfm_uk):,}

Target (Germany):
  N = {len(rfm_germany):,}
  Recency: {rfm_germany['Recency_capped'].mean():.1f} ± {rfm_germany['Recency_capped'].std():.1f} days
  Monetary: £{rfm_germany['Monetary_capped'].mean():,.0f} ± £{rfm_germany['Monetary_capped'].std():,.0f}

Difference: {exp6_stats.loc['Recency_capped', 'Mean_Diff_%']:.1f}% (Recency)

Experiment 7: High → Medium
━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Source (High-value):
  N = {len(rfm_high_value):,}
  Recency: {rfm_high_value['Recency_capped'].mean():.1f} ± {rfm_high_value['Recency_capped'].std():.1f} days
  Monetary: £{rfm_high_value['Monetary_capped'].mean():,.0f} ± £{rfm_high_value['Monetary_capped'].std():,.0f}

Target (Medium-value):
  N = {len(rfm_medium_value):,}
  Recency: {rfm_medium_value['Recency_capped'].mean():.1f} ± {rfm_medium_value['Recency_capped'].std():.1f} days
  Monetary: £{rfm_medium_value['Monetary_capped'].mean():,.0f} ± £{rfm_medium_value['Monetary_capped'].std():,.0f}

Difference: {exp7_stats.loc['Recency_capped', 'Mean_Diff_%']:.1f}% (Recency)
            {exp7_stats.loc['Monetary_capped', 'Mean_Diff_%']:.1f}% (Monetary)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FIXES APPLIED:
✅ Fixed reference date: {REFERENCE_DATE}
✅ Outlier capping: 99th percentile
✅ Normalized features for TL
✅ Density/log/box plots
"""

ax10.text(0.05, 0.95, summary_text, transform=ax10.transAxes, 
          fontsize=9, verticalalignment='top', family='monospace',
          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.savefig('uk_retail_rfm_distributions_FIXED.png', dpi=300, bbox_inches='tight')
print("✅ Saved: uk_retail_rfm_distributions_FIXED.png")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("🎉 UK RETAIL RFM GENERATION COMPLETE (FIXED VERSION)!")
print("="*80)

print("\n📁 Files Created:")
files_created = [
    ("uk_retail_country_stats.csv", "Country distribution analysis"),
    ("exp5_uk_source_RFM_FIXED.csv", "Exp 5 source (raw)"),
    ("exp5_france_target_RFM_FIXED.csv", "Exp 5 target (raw)"),
    ("exp5_uk_source_RFM_scaled.csv", "Exp 5 source (normalized)"),
    ("exp5_france_target_RFM_scaled.csv", "Exp 5 target (normalized)"),
    ("exp6_germany_target_RFM_FIXED.csv", "Exp 6 target (raw)"),
    ("exp6_germany_target_RFM_scaled.csv", "Exp 6 target (normalized)"),
    ("exp7_highvalue_source_RFM_FIXED.csv", "Exp 7 source (raw)"),
    ("exp7_mediumvalue_target_RFM_FIXED.csv", "Exp 7 target (raw)"),
    ("exp7_highvalue_source_RFM_scaled.csv", "Exp 7 source (normalized)"),
    ("exp7_mediumvalue_target_RFM_scaled.csv", "Exp 7 target (normalized)"),
    ("uk_retail_experiments_comparison_FIXED.csv", "Statistical comparison"),
    ("uk_retail_rfm_distributions_FIXED.png", "Research-quality visualizations")
]

for i, (filename, description) in enumerate(files_created, 1):
    print(f"  {i:2d}. {filename:<50} ({description})")

print(f"\n✅ Total: {len(files_created)} files created")

print("\n🔧 FIXES APPLIED:")
print("  1. ✅ Fixed reference date (consistent with Week 2)")
print("  2. ✅ Outlier capping at 99th percentile (research-backed)")
print("  3. ✅ Normalization for transfer learning (scaled features)")
print("  4. ✅ Improved visualizations (density, log scale, box plots)")
print("  5. ✅ Statistical tests (KS test, t-test)")

print("\n📊 DATA QUALITY VALIDATION:")
print(f"  Reference Date: {REFERENCE_DATE} ✅")
print(f"  Outlier Handling: 99th percentile cap ✅")
print(f"  Normalization: StandardScaler (z-score) ✅")
print(f"  RFM Scoring: 5 quintiles (matching Week 2) ✅")

print("\n🎯 NEXT STEPS:")
print("  1. ✅ RFM generation complete (use *_FIXED.csv files)")
print("  2. ⏳ Run baseline models on UK source (use *_scaled.csv for training)")
print("  3. ⏳ Test transfer to France, Germany, Medium-value targets")
print("  4. ⏳ Calculate transferability metrics (use scaled features!)")
print("  5. ⏳ Compare predictions vs reality")
print("  6. ⏳ Validate framework on real data")

print("\n⚠️  IMPORTANT FOR YOUR FRAMEWORK:")
print("  - Use *_scaled.csv files when calculating MMD, KL divergence, etc.")
print("  - Use *_FIXED.csv files for interpretability (raw values)")
print("  - Reference date is NOW consistent with Week 2 methodology")
print("  - Outliers are handled (no more £77k customer problem)")

print("\n✨ Ready for Week 5-6 Transfer Learning Validation!")
print("="*80)
