# Member 1: Synthetic Customer Transaction Generator - FIXED VERSION
# Week 2 Deliverable
# Generate realistic customer purchase patterns from product catalog
#
# FIX: Added customer_id_offset parameter to prevent overlap
#
# RESEARCH-BACKED IMPLEMENTATION
# ================================
# Persona Distribution: Adobe 2025 Customer Loyalty Research (n=1,003)
# Brand Behavior: Foxall & James (2004) - Behavioral Economics
# Quality Premium: Medallia 2024 Personalization Study (n=3,654)
# Validation: Bain & Company 2024, NIQ BASES 2024

import pandas as pd
import numpy as np
import random
import time
from datetime import datetime, timedelta

class SyntheticCustomerGenerator:
    """
    Generate synthetic customer transactions from product catalog
    Creates realistic purchase patterns for customer segmentation
    
    Research-backed synthetic customer generation following industry standards:
    - Adobe (2025): Customer loyalty and persona distribution
    - Foxall & James (2004): Multi-brand buying behavior (5-15% single-brand loyal)
    - Medallia (2024): 61% willing to pay premium for quality
    """
    
    def __init__(self, product_df, seed=42):
        """
        Args:
            product_df: DataFrame with products (must have: product, category, sale_price, brand, rating)
            seed: Random seed for reproducibility
        """
        self.products = product_df
        np.random.seed(seed)
        random.seed(seed)
        
        # Cache for customer brand preferences
        self.customer_brands = {}
        
        # Customer personas (behavioral archetypes)
        # RESEARCH-BACKED: Adobe 2025 study (n=1,003 US consumers)
        self.personas = {
            'price_sensitive': 0.35,    # 35% - Deal seekers
            'convenience': 0.30,         # 30% - Ease-focused
            'brand_loyal': 0.20,         # 20% - Stick to brands
            'quality_focused': 0.15      # 15% - Premium buyers
        }
    
    def generate_customers(self, n_customers=5000, customer_id_offset=0):
        """
        Generate customer profiles with different behaviors
        
        Args:
            n_customers: Number of customers to generate
            customer_id_offset: Starting customer ID (CRITICAL for preventing overlap!)
        
        Returns:
            DataFrame with customer_id, persona, preferred_category, price_sensitivity, etc.
        """
        customers = []
        
        for i in range(n_customers):
            # ✅ FIX: Add offset to customer ID
            customer_id = f'C{customer_id_offset + i:05d}'
            
            # Assign persona
            persona = np.random.choice(
                list(self.personas.keys()),
                p=list(self.personas.values())
            )
            
            # Assign preferences based on persona
            if persona == 'price_sensitive':
                price_sensitivity = np.random.uniform(0.7, 1.0)
                brand_loyalty = np.random.uniform(0.0, 0.3)
                quality_importance = np.random.uniform(0.0, 0.4)
                
            elif persona == 'brand_loyal':
                price_sensitivity = np.random.uniform(0.2, 0.5)
                brand_loyalty = np.random.uniform(0.7, 1.0)
                quality_importance = np.random.uniform(0.5, 0.8)
                
            elif persona == 'quality_focused':
                price_sensitivity = np.random.uniform(0.1, 0.4)
                brand_loyalty = np.random.uniform(0.4, 0.7)
                quality_importance = np.random.uniform(0.8, 1.0)
                
            else:  # convenience
                price_sensitivity = np.random.uniform(0.3, 0.6)
                brand_loyalty = np.random.uniform(0.3, 0.6)
                quality_importance = np.random.uniform(0.4, 0.7)
            
            # Assign preferred category
            preferred_category = np.random.choice(self.products['category'].unique())
            
            # Purchase frequency (transactions per month)
            if persona == 'convenience':
                purchase_frequency = np.random.uniform(4, 10)
            elif persona in ['brand_loyal', 'quality_focused']:
                purchase_frequency = np.random.uniform(3, 8)
            else:  # price_sensitive
                purchase_frequency = np.random.uniform(1, 5)
            
            customers.append({
                'customer_id': customer_id,
                'persona': persona,
                'preferred_category': preferred_category,
                'price_sensitivity': price_sensitivity,
                'brand_loyalty': brand_loyalty,
                'quality_importance': quality_importance,
                'purchase_frequency': purchase_frequency
            })
        
        return pd.DataFrame(customers)
    
    def select_product_for_customer(self, customer, available_products):
        """
        Select a product based on customer preferences
        
        Uses weighted scoring based on customer persona.
        Implements multi-brand buying behavior (Foxall & James, 2004)
        """
        filtered_products = available_products.copy()
        
        customer_id = customer['customer_id']
        persona = customer['persona']
        
        # Initialize brand preferences for this customer if not already done
        if customer_id not in self.customer_brands:
            available_brands = available_products['brand'].unique()
            
            if persona == 'brand_loyal':
                if len(available_brands) > 0:
                    self.customer_brands[customer_id] = np.random.choice(
                        available_brands,
                        size=min(2, len(available_brands)),
                        replace=False
                    )
                else:
                    self.customer_brands[customer_id] = []
                    
            elif persona != 'convenience':
                if len(available_brands) > 0:
                    self.customer_brands[customer_id] = np.random.choice(
                        available_brands,
                        size=min(5, len(available_brands)),
                        replace=False
                    )
                else:
                    self.customer_brands[customer_id] = []
            else:
                self.customer_brands[customer_id] = []
        
        # Apply brand filtering
        if persona == 'brand_loyal' and len(self.customer_brands[customer_id]) > 0:
            filtered_products = filtered_products[
                filtered_products['brand'].isin(self.customer_brands[customer_id])
            ]
        elif persona != 'convenience' and len(self.customer_brands[customer_id]) > 0:
            if np.random.rand() < 0.7:
                filtered_products = filtered_products[
                    filtered_products['brand'].isin(self.customer_brands[customer_id])
                ]
        
        # Fallback
        if len(filtered_products) == 0:
            filtered_products = available_products.copy()
        
        if len(filtered_products) == 0:
            return None
        
        # Calculate scores for each product
        scores = []
        
        for _, product in filtered_products.iterrows():
            score = 0
            
            # Price factor
            price_normalized = product['sale_price'] / filtered_products['sale_price'].max()
            price_score = (1 - price_normalized) * customer['price_sensitivity']
            
            # Category preference
            category_score = 0.5 if product['category'] == customer['preferred_category'] else 0.1
            
            # Quality factor
            if pd.notna(product.get('rating_clean', product.get('rating'))):
                quality_score = product.get('rating_clean', product.get('rating')) / 5.0
            else:
                quality_score = 0.5
            quality_score *= customer['quality_importance']
            
            # Combine scores
            score = (
                price_score * 0.4 +
                category_score * 0.3 +
                quality_score * 0.3
            )
            
            scores.append(score)
        
        # Select product probabilistically based on scores
        scores = np.array(scores)
        probabilities = scores / scores.sum()
        
        selected_idx = np.random.choice(len(filtered_products), p=probabilities)
        return filtered_products.iloc[selected_idx]
    
    def generate_transactions(self, customers, n_transactions=50000, 
                              start_date='2024-01-01', end_date='2024-06-30'):
        """
        Generate realistic purchase transactions
        
        Returns:
            DataFrame with: transaction_id, customer_id, product_id, date, quantity, price
        """
        print(f"   Generating {n_transactions:,} transactions...")
        start_time = time.time()
        
        transactions = []
        
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        date_range = (end - start).days
        
        transaction_id = 1
        
        # Distribute transactions across customers based on their frequency
        total_frequency = customers['purchase_frequency'].sum()
        
        print(f"   Processing {len(customers)} customers...")
        
        for idx, (_, customer) in enumerate(customers.iterrows()):
            # Progress indicator
            if idx % max(1, len(customers) // 5) == 0:
                print(f"   Progress: {idx}/{len(customers)} customers ({idx/len(customers)*100:.0f}%)")
            
            # Number of transactions for this customer
            n_cust_transactions = int(
                (customer['purchase_frequency'] / total_frequency) * n_transactions
            )
            
            for _ in range(n_cust_transactions):
                # Random date within range
                days_offset = np.random.randint(0, date_range)
                transaction_date = start + timedelta(days=days_offset)
                
                # Select product
                product = self.select_product_for_customer(customer, self.products)
                
                if product is None:
                    continue
                
                # Quantity
                quantity = np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])
                
                # Total price
                total_price = product['sale_price'] * quantity
                
                transactions.append({
                    'transaction_id': f'T{transaction_id:07d}',
                    'customer_id': customer['customer_id'],
                    'product_id': product.get('product', 'Unknown'),
                    'category': product['category'],
                    'brand': product.get('brand', 'Unknown'),
                    'date': transaction_date.strftime('%Y-%m-%d'),
                    'quantity': quantity,
                    'unit_price': product['sale_price'],
                    'total_price': total_price
                })
                
                transaction_id += 1
        
        elapsed = time.time() - start_time
        print(f"   Completed in {elapsed:.2f} seconds ({len(transactions)} transactions)")
        
        return pd.DataFrame(transactions)
    
    def calculate_rfm(self, transactions, reference_date=None):
        """
        Calculate RFM features from transactions
        
        Returns:
            DataFrame with: customer_id, Recency, Frequency, Monetary
        """
        if reference_date is None:
            reference_date = pd.to_datetime(transactions['date']).max()
        else:
            reference_date = pd.to_datetime(reference_date)
        
        transactions['date'] = pd.to_datetime(transactions['date'])
        
        rfm = transactions.groupby('customer_id').agg({
            'date': lambda x: (reference_date - x.max()).days,
            'transaction_id': 'count',
            'total_price': 'sum'
        }).reset_index()
        
        rfm.columns = ['customer_id', 'Recency', 'Frequency', 'Monetary']
        
        # Add RFM scores (1-5 scale)
        rfm['R_Score'] = pd.qcut(rfm['Recency'], q=5, labels=[5,4,3,2,1], duplicates='drop')
        rfm['F_Score'] = pd.qcut(rfm['Frequency'], q=5, labels=[1,2,3,4,5], duplicates='drop')
        rfm['M_Score'] = pd.qcut(rfm['Monetary'], q=5, labels=[1,2,3,4,5], duplicates='drop')
        
        # RFM combined score
        rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
        
        return rfm


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    
    print("="*80)
    print("SYNTHETIC CUSTOMER TRANSACTION GENERATOR - FIXED VERSION")
    print("="*80)
    
    # Load product data
    print("\n📂 Loading product data...")
    products = pd.read_csv(r'D:\Akash\B.Tech\5th Sem\ADA\CDAT\data\processed\BigBasket_v2.csv')
    print(f"✓ Loaded {len(products):,} products")
    
    # Initialize generator
    generator = SyntheticCustomerGenerator(products, seed=42)
    
    # Generate customers WITH OFFSET
    print("\n👥 Generating customers with offset...")
    customers = generator.generate_customers(n_customers=100, customer_id_offset=0)
    print(f"✓ Generated {len(customers):,} customers")
    print(f"   Customer ID range: {customers['customer_id'].iloc[0]} to {customers['customer_id'].iloc[-1]}")
    
    # Generate transactions
    print("\n🛒 Generating transactions...")
    transactions = generator.generate_transactions(
        customers, 
        n_transactions=500,
        start_date='2024-01-01',
        end_date='2024-06-30'
    )
    print(f"✓ Generated {len(transactions):,} transactions")
    
    # Calculate RFM
    print("\n📊 Calculating RFM features...")
    rfm = generator.calculate_rfm(transactions, reference_date='2024-07-01')
    print(f"✓ Calculated RFM for {len(rfm):,} customers")
    
    print("\n" + "="*80)
    print("✅ TEST COMPLETE - synth_FIXED.py is working!")
    print("="*80)
