# Member 1: Synthetic Customer Transaction Generator - PAIRS 5, 6, 7 VERSION
# Week 2 Deliverable - Domain Pairs 5, 6, 7
# Generate realistic customer purchase patterns with PAIR-SPECIFIC PERSONAS
#
# ENHANCEMENTS:
# - Custom persona distributions per pair
# - Fixed negative probability bug (clip scores to non-negative)
# - Parameterized behavioral ranges per persona
#
# RESEARCH-BACKED IMPLEMENTATION
# ================================
# Pair 5 (Eggs/Meat → Baby Care): Mintel Baby Care 2024, Circana Parent Shopper Study
# Pair 6 (Baby Care → Bakery/Dairy): NPD Household Pantry Report 2024
# Pair 7 (Beverages → Gourmet): Mintel Gourmet Food 2024, Specialty Food Association 2024
# Base personas: Adobe 2025, Foxall & James (2004), Medallia 2024

import pandas as pd
import numpy as np
import random
import time
from datetime import datetime, timedelta

class SyntheticCustomerGenerator567:
    """
    Generate synthetic customer transactions with PAIR-SPECIFIC PERSONAS
    
    Supports custom persona distributions and behavioral parameters per domain pair.
    Fixes negative probability bug in product selection.
    """
    
    def __init__(self, product_df, seed=42, custom_personas=None):
        """
        Args:
            product_df: DataFrame with products (must have: product, category, sale_price, brand, rating)
            seed: Random seed for reproducibility
            custom_personas: Dict of {persona_name: weight} OR None (use default)
        """
        self.products = product_df
        np.random.seed(seed)
        random.seed(seed)
        
        # Cache for customer brand preferences
        self.customer_brands = {}
        
        # Customer personas - can be overridden per pair
        if custom_personas is not None:
            self.personas = custom_personas
        else:
            # Default (generic grocery personas)
            self.personas = {
                'price_sensitive': 0.35,
                'convenience': 0.30,
                'brand_loyal': 0.20,
                'quality_focused': 0.15
            }
    
    def _get_persona_params(self, persona):
        """
        Get behavioral parameters for a given persona
        
        Returns: dict with price_sensitivity, brand_loyalty, quality_importance, purchase_frequency ranges
        """
        # PAIR 5: Eggs, Meat & Fish → Baby Care
        # Source (Protein buyers): price-conscious, freshness focus
        # Target (Baby care): safety-first, brand-loyal parents
        if persona == 'new_parent_guardian':
            return {
                'price_sensitivity': (0.2, 0.5),
                'brand_loyalty': (0.6, 0.9),
                'quality_importance': (0.8, 1.0),
                'purchase_frequency': (2, 6)
            }
        elif persona == 'safety_first':
            return {
                'price_sensitivity': (0.3, 0.6),
                'brand_loyalty': (0.5, 0.8),
                'quality_importance': (0.7, 0.95),
                'purchase_frequency': (1, 4)
            }
        
        # PAIR 6: Baby Care → Bakery, Cakes & Dairy
        # Target: Routine family replenishment, higher frequency
        elif persona == 'routine_repeat':
            return {
                'price_sensitivity': (0.3, 0.6),
                'brand_loyalty': (0.4, 0.7),
                'quality_importance': (0.4, 0.7),
                'purchase_frequency': (6, 14)
            }
        elif persona == 'occasion_shopper':
            return {
                'price_sensitivity': (0.3, 0.6),
                'brand_loyalty': (0.2, 0.5),
                'quality_importance': (0.5, 0.8),
                'purchase_frequency': (2, 6)
            }
        
        # PAIR 7: Beverages → Gourmet & World Food
        # Target: Quality-focused, niche explorers, lower frequency
        elif persona == 'quality_connoisseur':
            return {
                'price_sensitivity': (0.1, 0.3),
                'brand_loyalty': (0.5, 0.9),
                'quality_importance': (0.8, 1.0),
                'purchase_frequency': (1, 4)
            }
        elif persona == 'niche_explorer':
            return {
                'price_sensitivity': (0.3, 0.6),
                'brand_loyalty': (0.2, 0.5),
                'quality_importance': (0.6, 0.9),
                'purchase_frequency': (1, 3)
            }
        elif persona == 'gifting_shopper':
            return {
                'price_sensitivity': (0.2, 0.5),
                'brand_loyalty': (0.3, 0.6),
                'quality_importance': (0.7, 0.95),
                'purchase_frequency': (0.5, 2)
            }
        
        # Generic personas (fallback for source domains or reuse)
        elif persona == 'value_hunter':
            return {
                'price_sensitivity': (0.6, 1.0),
                'brand_loyalty': (0.1, 0.3),
                'quality_importance': (0.2, 0.5),
                'purchase_frequency': (1, 5)
            }
        elif persona == 'convenience_seeker':
            return {
                'price_sensitivity': (0.4, 0.7),
                'brand_loyalty': (0.2, 0.5),
                'quality_importance': (0.3, 0.6),
                'purchase_frequency': (3, 8)
            }
        elif persona == 'experimenter':
            return {
                'price_sensitivity': (0.3, 0.7),
                'brand_loyalty': (0.1, 0.4),
                'quality_importance': (0.4, 0.7),
                'purchase_frequency': (1, 4)
            }
        elif persona == 'trend_follower':
            return {
                'price_sensitivity': (0.3, 0.7),
                'brand_loyalty': (0.2, 0.5),
                'quality_importance': (0.4, 0.8),
                'purchase_frequency': (1, 3)
            }
        
        # Default fallbacks
        elif persona == 'price_sensitive':
            return {
                'price_sensitivity': (0.7, 1.0),
                'brand_loyalty': (0.0, 0.3),
                'quality_importance': (0.0, 0.4),
                'purchase_frequency': (1, 5)
            }
        elif persona == 'brand_loyal':
            return {
                'price_sensitivity': (0.2, 0.5),
                'brand_loyalty': (0.7, 1.0),
                'quality_importance': (0.5, 0.8),
                'purchase_frequency': (3, 8)
            }
        elif persona == 'quality_focused':
            return {
                'price_sensitivity': (0.1, 0.4),
                'brand_loyalty': (0.4, 0.7),
                'quality_importance': (0.8, 1.0),
                'purchase_frequency': (2, 6)
            }
        elif persona == 'convenience':
            return {
                'price_sensitivity': (0.3, 0.6),
                'brand_loyalty': (0.3, 0.6),
                'quality_importance': (0.4, 0.7),
                'purchase_frequency': (4, 10)
            }
        else:
            # Ultimate fallback
            return {
                'price_sensitivity': (0.4, 0.7),
                'brand_loyalty': (0.3, 0.6),
                'quality_importance': (0.4, 0.7),
                'purchase_frequency': (2, 6)
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
            customer_id = f'C{customer_id_offset + i:05d}'
            
            # Assign persona based on custom distribution
            persona = np.random.choice(
                list(self.personas.keys()),
                p=list(self.personas.values())
            )
            
            # Get behavioral parameters for this persona
            params = self._get_persona_params(persona)
            
            price_sensitivity = np.random.uniform(*params['price_sensitivity'])
            brand_loyalty = np.random.uniform(*params['brand_loyalty'])
            quality_importance = np.random.uniform(*params['quality_importance'])
            purchase_frequency = np.random.uniform(*params['purchase_frequency'])
            
            # Assign preferred category
            preferred_category = np.random.choice(self.products['category'].unique())
            
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
        
        ✅ FIX: Clips scores to non-negative and handles zero-sum case
        """
        filtered_products = available_products.copy()
        
        customer_id = customer['customer_id']
        persona = customer['persona']
        
        # Initialize brand preferences for this customer if not already done
        if customer_id not in self.customer_brands:
            available_brands = available_products['brand'].unique()
            
            # Brand-loyal personas prefer 2-3 brands
            if persona in ['brand_loyal', 'new_parent_guardian', 'safety_first', 'quality_connoisseur']:
                if len(available_brands) > 0:
                    self.customer_brands[customer_id] = np.random.choice(
                        available_brands,
                        size=min(2, len(available_brands)),
                        replace=False
                    )
                else:
                    self.customer_brands[customer_id] = []
            
            # Mid-loyalty personas prefer 3-5 brands
            elif persona not in ['convenience', 'convenience_seeker', 'value_hunter']:
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
        if persona in ['brand_loyal', 'new_parent_guardian', 'safety_first', 'quality_connoisseur']:
            if len(self.customer_brands[customer_id]) > 0:
                filtered_products = filtered_products[
                    filtered_products['brand'].isin(self.customer_brands[customer_id])
                ]
        elif len(self.customer_brands[customer_id]) > 0:
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
            
            # Price factor (inverse: lower price = higher score for price-sensitive)
            max_price = filtered_products['sale_price'].max()
            if max_price > 0:
                price_normalized = product['sale_price'] / max_price
                price_score = (1 - price_normalized) * customer['price_sensitivity']
            else:
                price_score = 0.5
            
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
        
        # ✅ FIX: Clip scores to non-negative and handle edge cases
        scores = np.array(scores)
        scores = np.clip(scores, 1e-8, None)  # Ensure all scores are positive
        
        score_sum = scores.sum()
        if score_sum <= 0:
            # Fallback to uniform selection if all scores are zero
            probabilities = np.ones(len(scores)) / len(scores)
        else:
            probabilities = scores / score_sum
        
        # Verify probabilities are valid
        if not np.all(probabilities >= 0):
            probabilities = np.ones(len(scores)) / len(scores)
        
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
# PERSONA DEFINITIONS - RESEARCH-BACKED
# ============================================================================

# Pair 5: Eggs, Meat & Fish → Baby Care
PAIR5_SOURCE_PERSONAS = {
    'value_hunter': 0.40,           # Protein buyers price-conscious
    'convenience_seeker': 0.30,     # Quick shop for fresh items
    'quality_focused': 0.20,        # Freshness-focused
    'experimenter': 0.10
}

PAIR5_TARGET_PERSONAS = {
    'new_parent_guardian': 0.30,    # Safety-first, brand-loyal parents
    'safety_first': 0.25,           # Quality > price for baby items
    'value_hunter': 0.20,           # Budget-conscious parents
    'convenience_seeker': 0.15,     # Time-starved parents
    'experimenter': 0.10            # Trial new baby products
}

# Pair 6: Baby Care → Bakery, Cakes & Dairy
PAIR6_SOURCE_PERSONAS = {
    'new_parent_guardian': 0.30,
    'safety_first': 0.25,
    'value_hunter': 0.25,
    'routine_repeat': 0.20
}

PAIR6_TARGET_PERSONAS = {
    'routine_repeat': 0.40,         # Family staples, high frequency
    'value_hunter': 0.30,           # Price-conscious for commodities
    'occasion_shopper': 0.15,       # Cakes/treats for events
    'experimenter': 0.10,
    'trend_follower': 0.05
}

# Pair 7: Beverages → Gourmet & World Food
PAIR7_SOURCE_PERSONAS = {
    'convenience': 0.35,            # Routine beverage purchases
    'value_hunter': 0.30,
    'brand_loyal': 0.20,
    'quality_focused': 0.15
}

PAIR7_TARGET_PERSONAS = {
    'quality_connoisseur': 0.35,    # Premium/specialty food seekers
    'niche_explorer': 0.25,         # Variety/exotic seekers
    'value_hunter': 0.25,           # Deal hunters in gourmet
    'gifting_shopper': 0.15         # Special occasions/gifts
}


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    
    print("="*80)
    print("SYNTHETIC CUSTOMER TRANSACTION GENERATOR - PAIRS 5, 6, 7 VERSION")
    print("="*80)
    
    # Test with Pair 5 target personas
    print("\n🧪 Testing with Pair 5 TARGET personas (Baby Care)...")
    
    # Load product data
    print("\n📂 Loading product data...")
    products = pd.read_csv(r'../week1/domain_pair5_target.csv')
    print(f"✓ Loaded {len(products):,} products")
    
    # Initialize generator with custom personas
    generator = SyntheticCustomerGenerator567(
        products, 
        seed=42, 
        custom_personas=PAIR5_TARGET_PERSONAS
    )
    
    # Generate customers WITH OFFSET
    print("\n👥 Generating customers with offset...")
    customers = generator.generate_customers(n_customers=100, customer_id_offset=12000)
    print(f"✓ Generated {len(customers):,} customers")
    print(f"   Customer ID range: {customers['customer_id'].iloc[0]} to {customers['customer_id'].iloc[-1]}")
    
    print(f"\n🎭 Persona distribution:")
    for persona, count in customers['persona'].value_counts().items():
        print(f"   {persona}: {count} ({count/len(customers)*100:.1f}%)")
    
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
    
    print("\n📈 RFM Summary:")
    print(f"   Avg Recency: {rfm['Recency'].mean():.1f} days")
    print(f"   Avg Frequency: {rfm['Frequency'].mean():.1f} purchases")
    print(f"   Avg Monetary: ₹{rfm['Monetary'].mean():.2f}")
    
    print("\n" + "="*80)
    print("✅ TEST COMPLETE - synth_567.py is working!")
    print("="*80)
