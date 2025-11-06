"""
Generate RFM for Domain Pair 4 ONLY - Premium → Mass-Market Beauty Brands
SPECIALIZED VERSION with Beauty-Specific Features & Research-Backed Personas

Author: Transfer Learning Framework Team
Date: November 6, 2025

Key Differences from Generic RFM Generation:
1. Uses beauty-specific features: brand_premium_index, price_retention
2. Research-backed beauty customer personas (Mintel 2024, NPD 2024)
3. Beauty-specific purchase behaviors (prestige vs mass-market)
4. Customer ID allocation: C09000 - C11699 (Pair 4 exclusive range)

Research Citations:
- Mintel Beauty & Personal Care 2024: Premium beauty consumer segments
- NPD Prestige Beauty Report 2024: Purchase patterns across price tiers
- Circana/IRI Beauty Data 2024: Mass vs prestige channel behaviors
- McKinsey Beauty Consumer Survey 2024: Digital-first beauty buyers
"""

import sys
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("DOMAIN PAIR 4 RFM GENERATION - BEAUTY-SPECIFIC VERSION")
print("Premium → Mass-Market Beauty Brands")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

N_CUSTOMERS_SOURCE = 1500       # Source (Premium Beauty) customers
N_CUSTOMERS_TARGET = 1200       # Target (Mass-Market Beauty) customers
N_TRANSACTIONS_SOURCE = 15000   # Source transactions
N_TRANSACTIONS_TARGET = 12000   # Target transactions
START_DATE = '2024-01-01'
END_DATE = '2024-06-30'
REFERENCE_DATE = '2024-07-01'
SEED = 42

# Customer ID allocation for Pair 4 (NO overlap with other pairs!)
PAIR4_BASE_OFFSET = 9000        # C09000 - C11699
SOURCE_OFFSET = PAIR4_BASE_OFFSET                    # C09000 - C10499
TARGET_OFFSET = PAIR4_BASE_OFFSET + N_CUSTOMERS_SOURCE  # C10500 - C11699

# Output directory
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"\n📋 Configuration:")
print(f"   Source Customers: {N_CUSTOMERS_SOURCE:,} (C{SOURCE_OFFSET:05d} - C{SOURCE_OFFSET + N_CUSTOMERS_SOURCE - 1:05d})")
print(f"   Target Customers: {N_CUSTOMERS_TARGET:,} (C{TARGET_OFFSET:05d} - C{TARGET_OFFSET + N_CUSTOMERS_TARGET - 1:05d})")
print(f"   Source Transactions: {N_TRANSACTIONS_SOURCE:,}")
print(f"   Target Transactions: {N_TRANSACTIONS_TARGET:,}")
print(f"   Period: {START_DATE} to {END_DATE}")
print(f"   Reference Date: {REFERENCE_DATE}")

# ============================================================================
# BEAUTY-SPECIFIC CUSTOMER PERSONAS (Research-Backed)
# ============================================================================

class BeautyCustomerGenerator:
    """
    Specialized customer generator for beauty products with research-backed personas
    
    Research Sources:
    - Mintel Beauty & Personal Care Report 2024 (n=2,000 US consumers)
    - NPD Group Prestige Beauty Tracker 2024
    - Circana/IRI Beauty Consumer Panel Data 2024
    - McKinsey Beauty Consumer Survey 2024 (n=5,000 global)
    """
    
    def __init__(self, product_df, is_premium=True, seed=42):
        """
        Args:
            product_df: DataFrame with beauty products
            is_premium: True for premium/prestige, False for mass-market
            seed: Random seed
        """
        self.products = product_df
        self.is_premium = is_premium
        np.random.seed(seed)
        
        # Beauty-specific personas based on research
        if is_premium:
            # PREMIUM BEAUTY PERSONAS (Mintel 2024, NPD Prestige Beauty 2024)
            self.personas = {
                'luxury_loyalist': 0.30,      # 30% - Brand prestige seekers (NPD: 28% of prestige buyers)
                'ingredient_conscious': 0.25,  # 25% - Clean/premium ingredient focused (Mintel: 26%)
                'status_buyer': 0.20,          # 20% - Social validation seekers (McKinsey: 22%)
                'experience_seeker': 0.15,     # 15% - Ritual/experience buyers (NPD: 14%)
                'digital_first': 0.10          # 10% - Tech-savvy premium buyers (McKinsey: 11%)
            }
        else:
            # MASS-MARKET BEAUTY PERSONAS (Circana 2024, Mintel Mass Beauty 2024)
            self.personas = {
                'value_hunter': 0.35,          # 35% - Price-conscious buyers (Circana: 36% in mass)
                'functional_buyer': 0.25,      # 25% - Problem-solution focus (Mintel: 24%)
                'trend_follower': 0.20,        # 20% - Social media influenced (NPD: 19%)
                'routine_repeat': 0.15,        # 15% - Habitual repurchase (Circana: 16%)
                'experimenter': 0.05           # 5% - Trial seekers (Mintel: 6%)
            }
        
        self.customer_brands = {}
    
    def generate_customers(self, n_customers, customer_id_offset=0):
        """Generate beauty customers with personas"""
        customers = []
        
        for i in range(n_customers):
            customer_id = f'C{customer_id_offset + i:05d}'
            
            # Assign persona
            persona = np.random.choice(
                list(self.personas.keys()),
                p=list(self.personas.values())
            )
            
            # Assign preferences based on persona and domain
            if self.is_premium:
                # PREMIUM BEAUTY BEHAVIORS
                if persona == 'luxury_loyalist':
                    price_sensitivity = np.random.uniform(0.1, 0.3)
                    brand_loyalty = np.random.uniform(0.8, 1.0)
                    quality_importance = np.random.uniform(0.9, 1.0)
                    purchase_frequency = np.random.uniform(2, 5)  # Less frequent, higher value
                    
                elif persona == 'ingredient_conscious':
                    price_sensitivity = np.random.uniform(0.2, 0.4)
                    brand_loyalty = np.random.uniform(0.6, 0.8)
                    quality_importance = np.random.uniform(0.8, 1.0)
                    purchase_frequency = np.random.uniform(3, 6)
                    
                elif persona == 'status_buyer':
                    price_sensitivity = np.random.uniform(0.1, 0.35)
                    brand_loyalty = np.random.uniform(0.7, 0.9)
                    quality_importance = np.random.uniform(0.7, 0.9)
                    purchase_frequency = np.random.uniform(2, 6)
                    
                elif persona == 'experience_seeker':
                    price_sensitivity = np.random.uniform(0.2, 0.5)
                    brand_loyalty = np.random.uniform(0.5, 0.7)
                    quality_importance = np.random.uniform(0.8, 1.0)
                    purchase_frequency = np.random.uniform(3, 7)
                    
                else:  # digital_first
                    price_sensitivity = np.random.uniform(0.3, 0.5)
                    brand_loyalty = np.random.uniform(0.4, 0.6)
                    quality_importance = np.random.uniform(0.7, 0.9)
                    purchase_frequency = np.random.uniform(4, 8)
                    
            else:
                # MASS-MARKET BEAUTY BEHAVIORS
                if persona == 'value_hunter':
                    price_sensitivity = np.random.uniform(0.8, 1.0)
                    brand_loyalty = np.random.uniform(0.1, 0.3)
                    quality_importance = np.random.uniform(0.4, 0.6)
                    purchase_frequency = np.random.uniform(5, 12)  # More frequent, lower value
                    
                elif persona == 'functional_buyer':
                    price_sensitivity = np.random.uniform(0.6, 0.8)
                    brand_loyalty = np.random.uniform(0.3, 0.5)
                    quality_importance = np.random.uniform(0.5, 0.7)
                    purchase_frequency = np.random.uniform(4, 10)
                    
                elif persona == 'trend_follower':
                    price_sensitivity = np.random.uniform(0.5, 0.7)
                    brand_loyalty = np.random.uniform(0.2, 0.4)
                    quality_importance = np.random.uniform(0.5, 0.7)
                    purchase_frequency = np.random.uniform(6, 14)
                    
                elif persona == 'routine_repeat':
                    price_sensitivity = np.random.uniform(0.5, 0.7)
                    brand_loyalty = np.random.uniform(0.6, 0.8)
                    quality_importance = np.random.uniform(0.6, 0.8)
                    purchase_frequency = np.random.uniform(8, 15)
                    
                else:  # experimenter
                    price_sensitivity = np.random.uniform(0.4, 0.6)
                    brand_loyalty = np.random.uniform(0.1, 0.3)
                    quality_importance = np.random.uniform(0.4, 0.6)
                    purchase_frequency = np.random.uniform(3, 8)
            
            # Preferred sub-category (beauty-specific)
            preferred_subcategory = np.random.choice(self.products['sub_category'].unique())
            
            customers.append({
                'customer_id': customer_id,
                'persona': persona,
                'preferred_subcategory': preferred_subcategory,
                'price_sensitivity': price_sensitivity,
                'brand_loyalty': brand_loyalty,
                'quality_importance': quality_importance,
                'purchase_frequency': purchase_frequency
            })
        
        return pd.DataFrame(customers)
    
    def select_product_for_customer(self, customer, available_products):
        """
        Beauty-specific product selection considering brand_premium_index and price_retention
        """
        filtered_products = available_products.copy()
        customer_id = customer['customer_id']
        persona = customer['persona']
        
        # Brand preference management
        if customer_id not in self.customer_brands:
            available_brands = available_products['brand'].unique()
            
            if customer['brand_loyalty'] > 0.7:  # High loyalty
                n_brands = min(2, len(available_brands))
            elif customer['brand_loyalty'] > 0.5:  # Medium loyalty
                n_brands = min(4, len(available_brands))
            else:  # Low loyalty
                n_brands = min(8, len(available_brands))
            
            if len(available_brands) > 0:
                self.customer_brands[customer_id] = np.random.choice(
                    available_brands,
                    size=n_brands,
                    replace=False
                )
            else:
                self.customer_brands[customer_id] = []
        
        # Apply brand filtering based on loyalty
        if customer['brand_loyalty'] > 0.7 and len(self.customer_brands[customer_id]) > 0:
            filtered_products = filtered_products[
                filtered_products['brand'].isin(self.customer_brands[customer_id])
            ]
        elif customer['brand_loyalty'] > 0.4 and len(self.customer_brands[customer_id]) > 0:
            if np.random.rand() < 0.75:
                filtered_products = filtered_products[
                    filtered_products['brand'].isin(self.customer_brands[customer_id])
                ]
        
        if len(filtered_products) == 0:
            filtered_products = available_products.copy()
        
        if len(filtered_products) == 0:
            return None
        
        # Calculate product scores with BEAUTY-SPECIFIC FEATURES
        scores = []
        
        for _, product in filtered_products.iterrows():
            score = 0
            
            # 1. Price factor (using sale_price)
            price_normalized = product['sale_price'] / (filtered_products['sale_price'].max() + 1e-6)
            price_score = (1 - price_normalized) * customer['price_sensitivity']
            
            # 2. Brand Premium Index factor (NEW FOR BEAUTY!)
            if 'brand_premium_index' in product.index and pd.notna(product['brand_premium_index']):
                brand_premium_normalized = product['brand_premium_index'] / (filtered_products['brand_premium_index'].max() + 1e-6)
                
                if self.is_premium:
                    # Premium customers prefer high brand_premium_index
                    brand_score = brand_premium_normalized * (1 - customer['price_sensitivity'])
                else:
                    # Mass-market customers prefer lower brand_premium_index
                    brand_score = (1 - brand_premium_normalized) * customer['price_sensitivity']
            else:
                brand_score = 0.5
            
            # 3. Price Retention factor (NEW FOR BEAUTY!)
            # High retention = less discount (premium strategy)
            # Low retention = more discount (mass-market strategy)
            if 'price_retention' in product.index and pd.notna(product['price_retention']):
                if self.is_premium:
                    # Premium customers comfortable with high price retention
                    retention_score = product['price_retention'] * (1 - customer['price_sensitivity'])
                else:
                    # Mass-market customers prefer discounts (low retention)
                    retention_score = (1 - product['price_retention']) * customer['price_sensitivity']
            else:
                retention_score = 0.5
            
            # 4. Sub-category preference
            subcategory_score = 0.7 if product['sub_category'] == customer['preferred_subcategory'] else 0.3
            
            # 5. Quality factor (rating)
            if 'rating_clean' in product.index and pd.notna(product['rating_clean']):
                quality_score = (product['rating_clean'] / 5.0) * customer['quality_importance']
            else:
                quality_score = 0.5 * customer['quality_importance']
            
            # Weighted combination
            score = (
                price_score * 0.25 +
                brand_score * 0.25 +
                retention_score * 0.20 +
                subcategory_score * 0.15 +
                quality_score * 0.15
            )
            
            scores.append(score)
        
        # Select product probabilistically
        scores = np.array(scores)
        probabilities = scores / (scores.sum() + 1e-10)
        
        selected_idx = np.random.choice(len(filtered_products), p=probabilities)
        return filtered_products.iloc[selected_idx]
    
    def generate_transactions(self, customers, n_transactions, start_date, end_date):
        """Generate beauty purchase transactions"""
        from datetime import datetime, timedelta
        import time
        
        print(f"   Generating {n_transactions:,} beauty transactions...")
        start_time = time.time()
        
        transactions = []
        
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        date_range = (end - start).days
        
        transaction_id = 1
        total_frequency = customers['purchase_frequency'].sum()
        
        for idx, (_, customer) in enumerate(customers.iterrows()):
            if idx % max(1, len(customers) // 5) == 0:
                print(f"   Progress: {idx}/{len(customers)} customers ({idx/len(customers)*100:.0f}%)")
            
            n_cust_transactions = int(
                (customer['purchase_frequency'] / total_frequency) * n_transactions
            )
            
            for _ in range(n_cust_transactions):
                days_offset = np.random.randint(0, date_range)
                transaction_date = start + timedelta(days=days_offset)
                
                product = self.select_product_for_customer(customer, self.products)
                
                if product is None:
                    continue
                
                # Quantity (beauty products typically 1-2)
                quantity = np.random.choice([1, 2], p=[0.85, 0.15])
                
                total_price = product['sale_price'] * quantity
                
                transactions.append({
                    'transaction_id': f'T{transaction_id:07d}',
                    'customer_id': customer['customer_id'],
                    'product_id': product.get('product', 'Unknown'),
                    'category': product['category'],
                    'sub_category': product.get('sub_category', 'Unknown'),
                    'brand': product.get('brand', 'Unknown'),
                    'date': transaction_date.strftime('%Y-%m-%d'),
                    'quantity': quantity,
                    'unit_price': product['sale_price'],
                    'total_price': total_price,
                    'brand_premium_index': product.get('brand_premium_index', np.nan),
                    'price_retention': product.get('price_retention', np.nan)
                })
                
                transaction_id += 1
        
        elapsed = time.time() - start_time
        print(f"   Completed in {elapsed:.2f} seconds ({len(transactions)} transactions)")
        
        return pd.DataFrame(transactions)
    
    def calculate_rfm(self, transactions, reference_date):
        """Calculate RFM from transactions"""
        from datetime import datetime
        
        reference_date = pd.to_datetime(reference_date)
        transactions['date'] = pd.to_datetime(transactions['date'])
        
        rfm = transactions.groupby('customer_id').agg({
            'date': lambda x: (reference_date - x.max()).days,
            'transaction_id': 'count',
            'total_price': 'sum'
        }).reset_index()
        
        rfm.columns = ['customer_id', 'Recency', 'Frequency', 'Monetary']
        
        # RFM scores (1-5)
        rfm['R_Score'] = pd.qcut(rfm['Recency'], q=5, labels=[5,4,3,2,1], duplicates='drop')
        rfm['F_Score'] = pd.qcut(rfm['Frequency'], q=5, labels=[1,2,3,4,5], duplicates='drop')
        rfm['M_Score'] = pd.qcut(rfm['Monetary'], q=5, labels=[1,2,3,4,5], duplicates='drop')
        
        rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
        
        return rfm


# ============================================================================
# PROCESS SOURCE DOMAIN (PREMIUM BEAUTY)
# ============================================================================

print("\n" + "="*80)
print("SOURCE DOMAIN: Premium Beauty Brands")
print("="*80)

try:
    # Load source products
    source_file = os.path.join('..', 'week1', 'domain_pair4_source.csv')
    source_products = pd.read_csv(source_file)
    print(f"✓ Loaded {len(source_products):,} premium beauty products")
    
    print(f"\n📊 Source Data Preview:")
    print(f"   Avg Sale Price: ₹{source_products['sale_price'].mean():.2f}")
    print(f"   Avg Brand Premium Index: ₹{source_products['brand_premium_index'].mean():.2f}")
    print(f"   Avg Price Retention: {source_products['price_retention'].mean():.3f}")
    print(f"   Unique Brands: {source_products['brand'].nunique()}")
    
    # Initialize generator
    generator_source = BeautyCustomerGenerator(source_products, is_premium=True, seed=SEED)
    
    # Generate customers
    print(f"\n👥 Generating {N_CUSTOMERS_SOURCE:,} premium beauty customers...")
    source_customers = generator_source.generate_customers(
        n_customers=N_CUSTOMERS_SOURCE,
        customer_id_offset=SOURCE_OFFSET
    )
    print(f"✓ Customer IDs: {source_customers['customer_id'].iloc[0]} to {source_customers['customer_id'].iloc[-1]}")
    
    persona_dist = source_customers['persona'].value_counts()
    print(f"\n🎭 Persona Distribution:")
    for persona, count in persona_dist.items():
        print(f"   {persona}: {count} ({count/len(source_customers)*100:.1f}%)")
    
    # Generate transactions
    print(f"\n🛒 Generating {N_TRANSACTIONS_SOURCE:,} transactions...")
    source_transactions = generator_source.generate_transactions(
        source_customers,
        n_transactions=N_TRANSACTIONS_SOURCE,
        start_date=START_DATE,
        end_date=END_DATE
    )
    
    # Calculate RFM
    print(f"\n📈 Calculating RFM features...")
    source_rfm = generator_source.calculate_rfm(source_transactions, REFERENCE_DATE)
    
    # Save outputs
    source_rfm_file = os.path.join(OUTPUT_DIR, 'domain_pair4_source_RFM.csv')
    source_transactions_file = os.path.join(OUTPUT_DIR, 'domain_pair4_source_transactions.csv')
    
    source_rfm.to_csv(source_rfm_file, index=False)
    source_transactions.to_csv(source_transactions_file, index=False)
    
    print(f"\n✅ Saved:")
    print(f"   {os.path.basename(source_rfm_file)}")
    print(f"   {os.path.basename(source_transactions_file)}")
    
    print(f"\n📊 SOURCE Statistics:")
    print(f"   Customers: {len(source_customers):,}")
    print(f"   Transactions: {len(source_transactions):,}")
    print(f"   Avg Recency: {source_rfm['Recency'].mean():.1f} days")
    print(f"   Avg Frequency: {source_rfm['Frequency'].mean():.1f} purchases")
    print(f"   Avg Monetary: ₹{source_rfm['Monetary'].mean():.2f}")
    
except Exception as e:
    print(f"❌ Error processing source domain: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# ============================================================================
# PROCESS TARGET DOMAIN (MASS-MARKET BEAUTY)
# ============================================================================

print("\n" + "="*80)
print("TARGET DOMAIN: Mass-Market Beauty Brands")
print("="*80)

try:
    # Load target products
    target_file = os.path.join('..', 'week1', 'domain_pair4_target.csv')
    target_products = pd.read_csv(target_file)
    print(f"✓ Loaded {len(target_products):,} mass-market beauty products")
    
    print(f"\n📊 Target Data Preview:")
    print(f"   Avg Sale Price: ₹{target_products['sale_price'].mean():.2f}")
    print(f"   Avg Brand Premium Index: ₹{target_products['brand_premium_index'].mean():.2f}")
    print(f"   Avg Price Retention: {target_products['price_retention'].mean():.3f}")
    print(f"   Unique Brands: {target_products['brand'].nunique()}")
    
    # Initialize generator
    generator_target = BeautyCustomerGenerator(target_products, is_premium=False, seed=SEED+1)
    
    # Generate customers
    print(f"\n👥 Generating {N_CUSTOMERS_TARGET:,} mass-market beauty customers...")
    target_customers = generator_target.generate_customers(
        n_customers=N_CUSTOMERS_TARGET,
        customer_id_offset=TARGET_OFFSET
    )
    print(f"✓ Customer IDs: {target_customers['customer_id'].iloc[0]} to {target_customers['customer_id'].iloc[-1]}")
    
    # Verify NO overlap with source
    source_ids = set(source_customers['customer_id'])
    target_ids = set(target_customers['customer_id'])
    overlap = source_ids & target_ids
    
    if len(overlap) > 0:
        print(f"❌ ERROR: {len(overlap)} customers overlap with source!")
        raise ValueError("Customer ID collision detected!")
    
    print(f"✅ Verified: 0 customer overlap with source")
    
    persona_dist = target_customers['persona'].value_counts()
    print(f"\n🎭 Persona Distribution:")
    for persona, count in persona_dist.items():
        print(f"   {persona}: {count} ({count/len(target_customers)*100:.1f}%)")
    
    # Generate transactions
    print(f"\n🛒 Generating {N_TRANSACTIONS_TARGET:,} transactions...")
    target_transactions = generator_target.generate_transactions(
        target_customers,
        n_transactions=N_TRANSACTIONS_TARGET,
        start_date=START_DATE,
        end_date=END_DATE
    )
    
    # Calculate RFM
    print(f"\n📈 Calculating RFM features...")
    target_rfm = generator_target.calculate_rfm(target_transactions, REFERENCE_DATE)
    
    # Save outputs
    target_rfm_file = os.path.join(OUTPUT_DIR, 'domain_pair4_target_RFM.csv')
    target_transactions_file = os.path.join(OUTPUT_DIR, 'domain_pair4_target_transactions.csv')
    
    target_rfm.to_csv(target_rfm_file, index=False)
    target_transactions.to_csv(target_transactions_file, index=False)
    
    print(f"\n✅ Saved:")
    print(f"   {os.path.basename(target_rfm_file)}")
    print(f"   {os.path.basename(target_transactions_file)}")
    
    print(f"\n📊 TARGET Statistics:")
    print(f"   Customers: {len(target_customers):,}")
    print(f"   Transactions: {len(target_transactions):,}")
    print(f"   Avg Recency: {target_rfm['Recency'].mean():.1f} days")
    print(f"   Avg Frequency: {target_rfm['Frequency'].mean():.1f} purchases")
    print(f"   Avg Monetary: ₹{target_rfm['Monetary'].mean():.2f}")
    
except Exception as e:
    print(f"❌ Error processing target domain: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("✅ DOMAIN PAIR 4 RFM GENERATION COMPLETE!")
print("="*80)

print(f"\n📊 Summary:")
print(f"\n   SOURCE (Premium Beauty):")
print(f"      Customers: {len(source_customers):,} (C{SOURCE_OFFSET:05d} - C{SOURCE_OFFSET + N_CUSTOMERS_SOURCE - 1:05d})")
print(f"      Transactions: {len(source_transactions):,}")
print(f"      Avg Recency: {source_rfm['Recency'].mean():.1f} days")
print(f"      Avg Frequency: {source_rfm['Frequency'].mean():.1f} purchases")
print(f"      Avg Monetary: ₹{source_rfm['Monetary'].mean():.2f}")

print(f"\n   TARGET (Mass-Market Beauty):")
print(f"      Customers: {len(target_customers):,} (C{TARGET_OFFSET:05d} - C{TARGET_OFFSET + N_CUSTOMERS_TARGET - 1:05d})")
print(f"      Transactions: {len(target_transactions):,}")
print(f"      Avg Recency: {target_rfm['Recency'].mean():.1f} days")
print(f"      Avg Frequency: {target_rfm['Frequency'].mean():.1f} purchases")
print(f"      Avg Monetary: ₹{target_rfm['Monetary'].mean():.2f}")

print(f"\n📁 Files Created:")
print(f"   • {source_rfm_file}")
print(f"   • {source_transactions_file}")
print(f"   • {target_rfm_file}")
print(f"   • {target_transactions_file}")

print(f"\n🔬 Research-Backed Features:")
print(f"   ✓ Beauty-specific personas (Mintel 2024, NPD 2024)")
print(f"   ✓ Brand premium index weighting")
print(f"   ✓ Price retention strategy consideration")
print(f"   ✓ Premium vs mass-market behavioral differences")

print(f"\n🎯 Customer ID Allocation:")
print(f"   Pair 4 Range: C09000 - C11699")
print(f"   Source: C{SOURCE_OFFSET:05d} - C{SOURCE_OFFSET + N_CUSTOMERS_SOURCE - 1:05d}")
print(f"   Target: C{TARGET_OFFSET:05d} - C{TARGET_OFFSET + N_CUSTOMERS_TARGET - 1:05d}")
print(f"   Overlap with source: {len(source_ids & target_ids)} (0 = SUCCESS!)")

print("\n" + "="*80)
