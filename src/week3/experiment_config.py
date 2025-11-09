"""
Experiment Configuration for Transfer Learning Framework
Defines all 7 domain pairs and experiment parameters
"""

# Domain Pair Definitions
DOMAIN_PAIRS = {
    1: {
        'name': 'Pair 1: Cleaning & Household → Foodgrains',
        'source': 'Cleaning & Household',
        'target': 'Foodgrains, Oil & Masala',
        'expected_transferability': 'HIGH',  # 0.9028 from analysis
        'transferability_score': 0.9028,
        'strategy': 'No Finetune',
        'description': 'High transferability - expect good zero-shot transfer'
    },
    2: {
        'name': 'Pair 2: Snacks → Kitchen, Garden & Pets',
        'source': 'Snacks & Branded Foods',
        'target': 'Kitchen, Garden & Pets',
        'expected_transferability': 'LOW',  # 0.7254 from analysis
        'transferability_score': 0.7254,
        'strategy': 'Partial Finetune',
        'description': 'Low transferability - expect significant improvement with finetuning'
    },
    3: {
        'name': 'Pair 3: Premium → Budget (Whole Dataset)',
        'source': 'Beauty & Hygiene',  # Premium products
        'target': 'Snacks & Branded Foods',  # Budget products
        'expected_transferability': 'MODERATE',  # 0.8159 from analysis
        'transferability_score': 0.8159,
        'strategy': 'Partial Finetune',
        'description': 'Moderate transferability - price segment shift'
    },
    4: {
        'name': 'Pair 4: Premium → Mass-Market Beauty',
        'source': 'Gourmet & World Food',  # Premium
        'target': 'Beauty & Hygiene',  # Mass-market
        'expected_transferability': 'MODERATE-HIGH',  # 0.8958 from analysis
        'transferability_score': 0.8958,
        'strategy': 'Partial Finetune',
        'description': 'Moderate-high transferability - cross-category premium to mass-market'
    },
    5: {
        'name': 'Pair 5: Eggs, Meat & Fish → Baby Care',
        'source': 'Eggs, Meat & Fish',
        'target': 'Baby Care',
        'expected_transferability': 'LOW',  # 0.8036 from analysis
        'transferability_score': 0.8036,
        'strategy': 'New Model',
        'description': 'Low transferability - very different customer behaviors'
    },
    6: {
        'name': 'Pair 6: Baby Care → Bakery, Cakes & Dairy',
        'source': 'Baby Care',
        'target': 'Bakery, Cakes & Dairy',
        'expected_transferability': 'LOW',  # 0.7414 from analysis
        'transferability_score': 0.7414,
        'strategy': 'New Model',
        'description': 'Low transferability - different purchase patterns'
    },
    7: {
        'name': 'Pair 7: Beverages → Gourmet & World Food',
        'source': 'Beverages',
        'target': 'Gourmet & World Food',
        'expected_transferability': 'MODERATE-HIGH',  # 0.8951 from analysis
        'transferability_score': 0.8951,
        'strategy': 'No Finetune',
        'description': 'Moderate-high transferability - similar premium customer base'
    }
}

# Test Protocol Configuration
TEST_PROTOCOL = {
    'Test 1': {
        'name': 'Zero-Shot Transfer',
        'description': 'Transfer source model as-is to target domain',
        'target_data_percentage': 0,
        'training_mode': 'transfer_only'
    },
    'Test 2': {
        'name': 'Fine-tune 10%',
        'description': 'Fine-tune transferred model with 10% target data',
        'target_data_percentage': 10,
        'training_mode': 'finetune'
    },
    'Test 3': {
        'name': 'Fine-tune 20%',
        'description': 'Fine-tune transferred model with 20% target data',
        'target_data_percentage': 20,
        'training_mode': 'finetune'
    },
    'Test 4': {
        'name': 'Fine-tune 50%',
        'description': 'Fine-tune transferred model with 50% target data',
        'target_data_percentage': 50,
        'training_mode': 'finetune'
    },
    'Test 5': {
        'name': 'Train from Scratch',
        'description': 'Train new model from scratch on 100% target data',
        'target_data_percentage': 100,
        'training_mode': 'from_scratch'
    }
}

# Performance Metrics to Track
METRICS = [
    'silhouette_score',      # Cluster quality (-1 to 1, higher is better)
    'davies_bouldin_index',  # Cluster separation (lower is better)
    'calinski_harabasz',     # Cluster variance ratio (higher is better)
    'inertia',               # Within-cluster sum of squares (lower is better)
    'n_clusters',            # Number of clusters found
    'training_time',         # Time to train/finetune (seconds)
    'prediction_time',       # Time to predict on test set (seconds)
]

# File Paths
PATHS = {
    'models_dir': 'D:/Akash/B.Tech/5th Sem/ADA/Backup/CDAT/src/week2/models',
    'data_dir': 'D:/Akash/B.Tech/5th Sem/ADA/Backup/CDAT/src/week2',
    'results_dir': 'D:/Akash/B.Tech/5th Sem/ADA/Backup/CDAT/src/week3/results',
    'visualizations_dir': 'D:/Akash/B.Tech/5th Sem/ADA/Backup/CDAT/src/week3/visualizations'
}

# Clustering Parameters
CLUSTERING_PARAMS = {
    'algorithm': 'kmeans',
    'k_range': [3, 4, 5, 6],  # Test different numbers of clusters
    'random_state': 42,
    'max_iter': 300,
    'n_init': 10
}

# Experiment Metadata
EXPERIMENT_INFO = {
    'project_name': 'Transfer Learning Customer Segmentation',
    'experiment_date': '2025-11-08',
    'n_domain_pairs': 7,
    'n_tests_per_pair': 5,
    'total_experiments': 35,  # 7 pairs × 5 tests
    'expected_runtime_hours': 2.5  # Estimated total runtime
}

def get_model_path(pair_number):
    """Get path to source domain model"""
    return f"{PATHS['models_dir']}/domain_pair{pair_number}_rfm_kmeans_model.pkl"

def get_rfm_data_path(pair_number, domain='source'):
    """Get path to RFM data"""
    return f"{PATHS['data_dir']}/domain_pair{pair_number}_{domain}_RFM.csv"

def get_transaction_data_path(pair_number, domain='source'):
    """Get path to transaction data"""
    return f"{PATHS['data_dir']}/domain_pair{pair_number}_{domain}_transactions.csv"

def get_results_path(pair_number, test_name):
    """Get path to save individual test results"""
    test_num = test_name.split()[1]  # Extract number from "Test 1", "Test 2", etc.
    return f"{PATHS['results_dir']}/pair{pair_number}_test{test_num}_results.csv"

def print_experiment_summary():
    """Print a summary of the experiment configuration"""
    print("=" * 80)
    print("TRANSFER LEARNING EXPERIMENT CONFIGURATION")
    print("=" * 80)
    print(f"\nProject: {EXPERIMENT_INFO['project_name']}")
    print(f"Date: {EXPERIMENT_INFO['experiment_date']}")
    print(f"\nTotal Domain Pairs: {EXPERIMENT_INFO['n_domain_pairs']}")
    print(f"Tests per Pair: {EXPERIMENT_INFO['n_tests_per_pair']}")
    print(f"Total Experiments: {EXPERIMENT_INFO['total_experiments']}")
    print(f"Estimated Runtime: {EXPERIMENT_INFO['expected_runtime_hours']} hours")
    
    print("\n" + "=" * 80)
    print("DOMAIN PAIRS")
    print("=" * 80)
    for pair_id, pair_info in DOMAIN_PAIRS.items():
        print(f"\nPair {pair_id}: {pair_info['name']}")
        print(f"  Transferability: {pair_info['expected_transferability']} ({pair_info['transferability_score']:.4f})")
        print(f"  Strategy: {pair_info['strategy']}")
        print(f"  Description: {pair_info['description']}")
    
    print("\n" + "=" * 80)
    print("TEST PROTOCOL")
    print("=" * 80)
    for test_id, test_info in TEST_PROTOCOL.items():
        print(f"\n{test_id}: {test_info['name']}")
        print(f"  Target Data: {test_info['target_data_percentage']}%")
        print(f"  Mode: {test_info['training_mode']}")
        print(f"  Description: {test_info['description']}")
    
    print("\n" + "=" * 80)
    print("METRICS TO TRACK")
    print("=" * 80)
    for metric in METRICS:
        print(f"  • {metric}")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    print_experiment_summary()
