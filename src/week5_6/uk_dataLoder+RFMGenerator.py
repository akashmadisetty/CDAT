"""
UK Online Retail Dataset - Week 5-6 Implementation
===================================================
Loads UK Online Retail dataset, generates RFM features, and prepares
domain pairs for transfer learning validation experiments.

Experiments:
- Exp 5: UK → France transfer
- Exp 6: UK → Germany transfer  
- Exp 7: High-value → Medium-value customers
"""

import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 6)

print("="*80)
print("UK ONLINE RETAIL DATASET - RFM GENERATION FOR TRANSFER LEARNING")
print("="*80)

# ============================================================================
# STEP 1: LOAD DATASET
# ============================================================================

print("\n📦 STEP 1: Loading Dataset")
print("-"*80)

# Try multiple methods to load data
def load_uk_retail():
    """Try different methods to load UK Retail dataset"""
    
    
    # Method 2: Try local file (Excel)
    # try:
    #     print("Attempting local Excel file...")
    #     df = pd.read_excel(r'D:\ADA_Project_COPY\newdev\CDAT\data\original\UK.csv')
    #     print("✓ Loaded from local Excel file")
    #     return df
    # except Exception as e:
    #     print(f"  ✗ Excel method failed: {e}")
    
    # Method 3: Try local CSV
    try:
        print("Attempting local CSV file...")
        df = pd.read_csv(r'D:\ADA_Project_COPY\newdev\CDAT\data\original\UK.csv', encoding='ISO-8859-1')
        print("✓ Loaded from local CSV file")
        return df
    except Exception as e:
        print(f"  ✗ CSV method failed: {e}")
    

    print("="*80)
    return None

df_raw = load_uk_retail()

if df_raw is None:
    print("\n⚠️  Please download the dataset and re-run this script.")
    exit()

print(f"\n✓ Dataset loaded: {len(df_raw):,} transactions")
print(f"✓ Columns: {list(df_raw.columns)}")

# ============================================================================
# STEP 2: DATA CLEANING & PREPROCESSING
# ============================================================================

print("\n🧹 STEP 2: Data Cleaning")
print("-"*80)

df = df_raw.copy()

# Standardize column names (different versions have different names)
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

# Remove cancelled transactions (InvoiceNo starts with 'C')
df = df[~df['InvoiceNo'].astype(str).str.startswith('C')]
print(f"After removing cancellations: {len(df):,} rows")

# Remove missing CustomerID
df = df.dropna(subset=['CustomerID'])
print(f"After removing missing CustomerID: {len(df):,} rows")

# Remove invalid quantities and prices
df = df[(df['Quantity'] > 0) & (df['UnitPrice'] > 0)]
print(f"After removing invalid values: {len(df):,} rows")

# Convert InvoiceDate to datetime
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])

# Calculate TotalPrice
df['TotalPrice'] = df['Quantity'] * df['UnitPrice']

# Convert CustomerID to integer
df['CustomerID'] = df['CustomerID'].astype(int)

print(f"\n✓ Clean dataset: {len(df):,} transactions")
print(f"✓ Customers: {df['CustomerID'].nunique():,}")
print(f"✓ Countries: {df['Country'].nunique()}")
print(f"✓ Date range: {df['InvoiceDate'].min()} to {df['InvoiceDate'].max()}")

# ============================================================================
# STEP 3: EXPLORE COUNTRIES FOR EXPERIMENTS 5 & 6
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
print(country_stats.head(10))

# Save country stats
country_stats.to_csv('uk_retail_country_stats.csv')
print("\n✓ Saved: uk_retail_country_stats.csv")

# ============================================================================
# STEP 4: GENERATE RFM FEATURES - FUNCTION
# ============================================================================

def calculate_rfm(df, reference_date=None):
    """
    Calculate RFM features for customers
    
    Returns: DataFrame with CustomerID, Recency, Frequency, Monetary
    """
    if reference_date is None:
        reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    
    rfm = df.groupby('CustomerID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days,  # Recency
        'InvoiceNo': 'nunique',  # Frequency
        'TotalPrice': 'sum'  # Monetary
    }).rename(columns={
        'InvoiceDate': 'Recency',
        'InvoiceNo': 'Frequency',
        'TotalPrice': 'Monetary'
    })
    
    rfm = rfm.reset_index()
    
    # Add RFM scores (1-4 quartiles) - FIXED VERSION
    # Use dynamic labels based on actual number of bins created
    try:
        rfm['R_Score'] = pd.qcut(rfm['Recency'], q=4, labels=False, duplicates='drop') + 1
        # Reverse for Recency (lower recency = higher score)
        rfm['R_Score'] = 5 - rfm['R_Score']
    except:
        # Fallback if qcut fails
        rfm['R_Score'] = pd.cut(rfm['Recency'], bins=4, labels=False, duplicates='drop') + 1
        rfm['R_Score'] = 5 - rfm['R_Score']
    
    try:
        rfm['F_Score'] = pd.qcut(rfm['Frequency'], q=4, labels=False, duplicates='drop') + 1
    except:
        rfm['F_Score'] = pd.cut(rfm['Frequency'], bins=4, labels=False, duplicates='drop') + 1
    
    try:
        rfm['M_Score'] = pd.qcut(rfm['Monetary'], q=4, labels=False, duplicates='drop') + 1
    except:
        rfm['M_Score'] = pd.cut(rfm['Monetary'], bins=4, labels=False, duplicates='drop') + 1
    
    # Combined RFM Score
    rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
    
    return rfm

# ============================================================================
# EXPERIMENT 5: UK → FRANCE TRANSFER
# ============================================================================

print("\n" + "="*80)
print("EXPERIMENT 5: UK → FRANCE TRANSFER")
print("="*80)

# Filter UK and France data
df_uk = df[df['Country'] == 'United Kingdom'].copy()
df_france = df[df['Country'] == 'France'].copy()

print(f"\nUK: {len(df_uk):,} transactions, {df_uk['CustomerID'].nunique():,} customers")
print(f"France: {len(df_france):,} transactions, {df_france['CustomerID'].nunique():,} customers")

# Generate RFM for UK (source)
rfm_uk = calculate_rfm(df_uk)
print(f"\n✓ UK RFM: {len(rfm_uk):,} customers")
print(rfm_uk.head())

# Generate RFM for France (target)
rfm_france = calculate_rfm(df_france)
print(f"\n✓ France RFM: {len(rfm_france):,} customers")
print(rfm_france.head())

# Save
rfm_uk.to_csv('exp5_uk_source_RFM.csv', index=False)
rfm_france.to_csv('exp5_france_target_RFM.csv', index=False)
print("\n✅ Saved: exp5_uk_source_RFM.csv")
print("✅ Saved: exp5_france_target_RFM.csv")

# ============================================================================
# EXPERIMENT 6: UK → GERMANY TRANSFER
# ============================================================================

print("\n" + "="*80)
print("EXPERIMENT 6: UK → GERMANY TRANSFER")
print("="*80)

df_germany = df[df['Country'] == 'Germany'].copy()

print(f"\nUK: {len(df_uk):,} transactions, {df_uk['CustomerID'].nunique():,} customers")
print(f"Germany: {len(df_germany):,} transactions, {df_germany['CustomerID'].nunique():,} customers")

# Generate RFM for Germany (target)
rfm_germany = calculate_rfm(df_germany)
print(f"\n✓ Germany RFM: {len(rfm_germany):,} customers")
print(rfm_germany.head())

# Save (UK already saved above)
rfm_germany.to_csv('exp6_germany_target_RFM.csv', index=False)
print("\n✅ Saved: exp6_germany_target_RFM.csv")
print("   (Using same UK source as Exp 5)")

# ============================================================================
# EXPERIMENT 7: HIGH-VALUE → MEDIUM-VALUE CUSTOMERS
# ============================================================================

print("\n" + "="*80)
print("EXPERIMENT 7: HIGH-VALUE → MEDIUM-VALUE CUSTOMERS")
print("="*80)

# Calculate RFM for all UK customers
rfm_all_uk = calculate_rfm(df_uk)

# Define value segments based on Monetary
monetary_75 = rfm_all_uk['Monetary'].quantile(0.75)
monetary_50 = rfm_all_uk['Monetary'].quantile(0.50)
monetary_25 = rfm_all_uk['Monetary'].quantile(0.25)

print(f"\nMonetary Value Quartiles:")
print(f"  75th percentile (High-value threshold): £{monetary_75:.2f}")
print(f"  50th percentile (Median): £{monetary_50:.2f}")
print(f"  25th percentile (Low-value threshold): £{monetary_25:.2f}")

# High-value: Top 25% (≥75th percentile)
rfm_high_value = rfm_all_uk[rfm_all_uk['Monetary'] >= monetary_75].copy()

# Medium-value: 25th-75th percentile
rfm_medium_value = rfm_all_uk[
    (rfm_all_uk['Monetary'] >= monetary_25) & 
    (rfm_all_uk['Monetary'] < monetary_75)
].copy()

print(f"\nHigh-value: {len(rfm_high_value):,} customers (£{rfm_high_value['Monetary'].mean():.2f} avg)")
print(f"Medium-value: {len(rfm_medium_value):,} customers (£{rfm_medium_value['Monetary'].mean():.2f} avg)")

# Save
rfm_high_value.to_csv('exp7_highvalue_source_RFM.csv', index=False)
rfm_medium_value.to_csv('exp7_mediumvalue_target_RFM.csv', index=False)
print("\n✅ Saved: exp7_highvalue_source_RFM.csv")
print("✅ Saved: exp7_mediumvalue_target_RFM.csv")

# ============================================================================
# STEP 5: COMPARATIVE STATISTICS
# ============================================================================

print("\n" + "="*80)
print("COMPARATIVE STATISTICS - ALL EXPERIMENTS")
print("="*80)

comparison = pd.DataFrame({
    'Experiment': ['Exp 5: UK→France', 'Exp 6: UK→Germany', 'Exp 7: High→Medium'],
    'Source_N': [len(rfm_uk), len(rfm_uk), len(rfm_high_value)],
    'Target_N': [len(rfm_france), len(rfm_germany), len(rfm_medium_value)],
    'Source_Avg_Recency': [
        rfm_uk['Recency'].mean(),
        rfm_uk['Recency'].mean(),
        rfm_high_value['Recency'].mean()
    ],
    'Target_Avg_Recency': [
        rfm_france['Recency'].mean(),
        rfm_germany['Recency'].mean(),
        rfm_medium_value['Recency'].mean()
    ],
    'Source_Avg_Monetary': [
        rfm_uk['Monetary'].mean(),
        rfm_uk['Monetary'].mean(),
        rfm_high_value['Monetary'].mean()
    ],
    'Target_Avg_Monetary': [
        rfm_france['Monetary'].mean(),
        rfm_germany['Monetary'].mean(),
        rfm_medium_value['Monetary'].mean()
    ]
})

print("\n", comparison.to_string(index=False))
comparison.to_csv('uk_retail_experiments_comparison.csv', index=False)
print("\n✅ Saved: uk_retail_experiments_comparison.csv")

# ============================================================================
# STEP 6: VISUALIZATIONS
# ============================================================================

print("\n📊 STEP 6: Creating Visualizations")
print("-"*80)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Exp 5: UK vs France
axes[0,0].hist([rfm_uk['Recency'], rfm_france['Recency']], bins=30, label=['UK', 'France'], alpha=0.7)
axes[0,0].set_title('Exp 5: Recency Distribution\nUK vs France')
axes[0,0].set_xlabel('Recency (days)')
axes[0,0].legend()

axes[1,0].hist([rfm_uk['Monetary'], rfm_france['Monetary']], bins=30, label=['UK', 'France'], alpha=0.7)
axes[1,0].set_title('Exp 5: Monetary Distribution\nUK vs France')
axes[1,0].set_xlabel('Monetary Value (£)')
axes[1,0].legend()

# Exp 6: UK vs Germany
axes[0,1].hist([rfm_uk['Recency'], rfm_germany['Recency']], bins=30, label=['UK', 'Germany'], alpha=0.7)
axes[0,1].set_title('Exp 6: Recency Distribution\nUK vs Germany')
axes[0,1].set_xlabel('Recency (days)')
axes[0,1].legend()

axes[1,1].hist([rfm_uk['Monetary'], rfm_germany['Monetary']], bins=30, label=['UK', 'Germany'], alpha=0.7)
axes[1,1].set_title('Exp 6: Monetary Distribution\nUK vs Germany')
axes[1,1].set_xlabel('Monetary Value (£)')
axes[1,1].legend()

# Exp 7: High vs Medium value
axes[0,2].hist([rfm_high_value['Frequency'], rfm_medium_value['Frequency']], bins=30, label=['High', 'Medium'], alpha=0.7)
axes[0,2].set_title('Exp 7: Frequency Distribution\nHigh vs Medium Value')
axes[0,2].set_xlabel('Frequency (orders)')
axes[0,2].legend()

axes[1,2].hist([rfm_high_value['Monetary'], rfm_medium_value['Monetary']], bins=30, label=['High', 'Medium'], alpha=0.7)
axes[1,2].set_title('Exp 7: Monetary Distribution\nHigh vs Medium Value')
axes[1,2].set_xlabel('Monetary Value (£)')
axes[1,2].legend()

plt.tight_layout()
plt.savefig('uk_retail_rfm_distributions.png', dpi=300, bbox_inches='tight')
print("✅ Saved: uk_retail_rfm_distributions.png")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("🎉 UK RETAIL RFM GENERATION COMPLETE!")
print("="*80)

print("\n📁 Files Created:")
files = [
    "uk_retail_country_stats.csv",
    "exp5_uk_source_RFM.csv",
    "exp5_france_target_RFM.csv",
    "exp6_germany_target_RFM.csv",
    "exp7_highvalue_source_RFM.csv",
    "exp7_mediumvalue_target_RFM.csv",
    "uk_retail_experiments_comparison.csv",
    "uk_retail_rfm_distributions.png"
]

for i, f in enumerate(files, 1):
    print(f"  {i}. {f}")

print("\n🎯 Next Steps:")
print("  1. Review the RFM distributions visualization")
print("  2. Run your baseline models on UK source (Exp 5 & 6)")
print("  3. Run your baseline model on High-value source (Exp 7)")
print("  4. Test transfer to France, Germany, and Medium-value targets")
print("  5. Calculate transferability metrics using your framework")
print("  6. Compare: Do predictions match reality?")

print("\n✨ Ready for Week 5-6 experiments!")
print("="*80)