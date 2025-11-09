"""
Main Experiment Runner for Transfer Learning Framework
Runs all 5 tests across all 7 domain pairs (35 total experiments)

Usage:
    python run_all_experiments.py                    # Run all experiments
    python run_all_experiments.py --pair 1           # Run only pair 1
    python run_all_experiments.py --test 1           # Run only test 1 for all pairs
    python run_all_experiments.py --pair 1 --test 2  # Run specific pair and test
"""

import pandas as pd
import numpy as np
import pickle
import time
import os
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from experiment_config import (
    DOMAIN_PAIRS, TEST_PROTOCOL, METRICS, PATHS, 
    CLUSTERING_PARAMS, get_model_path, get_rfm_data_path, 
    get_transaction_data_path, get_results_path
)


class TransferLearningExperiment:
    """Handles transfer learning experiments for customer segmentation"""
    
    def __init__(self, pair_number, verbose=True):
        self.pair_number = pair_number
        self.pair_info = DOMAIN_PAIRS[pair_number]
        self.verbose = verbose
        self.results = []
        
        # Load source model
        self.source_model = self._load_source_model()
        
        # Load data
        self.source_data = self._load_rfm_data('source')
        self.target_data = self._load_rfm_data('target')
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"Initialized Experiment for {self.pair_info['name']}")
            print(f"{'='*80}")
            print(f"Source samples: {len(self.source_data)}")
            print(f"Target samples: {len(self.target_data)}")
            print(f"Expected transferability: {self.pair_info['expected_transferability']}")
            print(f"Transferability score: {self.pair_info['transferability_score']:.4f}")
    
    def _load_source_model(self):
        """Load pre-trained source domain model"""
        model_path = get_model_path(self.pair_number)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Source model not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            model_dict = pickle.load(f)
        
        # Extract the actual model from the dictionary
        if isinstance(model_dict, dict):
            model = model_dict['model']
            self.scaler = model_dict.get('scaler', None)
        else:
            model = model_dict
            self.scaler = None
        
        if self.verbose:
            print(f"✓ Loaded source model from {model_path}")
        
        return model
    
    def _load_rfm_data(self, domain='source'):
        """Load RFM features for source or target domain"""
        data_path = get_rfm_data_path(self.pair_number, domain)
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"RFM data not found: {data_path}")
        
        df = pd.read_csv(data_path)
        
        # Extract RFM features (column names are capitalized)
        rfm_features = ['Recency', 'Frequency', 'Monetary']
        X = df[rfm_features].values
        
        if self.verbose:
            print(f"✓ Loaded {domain} RFM data: {X.shape}")
        
        return X
    
    def _split_target_data(self, percentage):
        """Split target data for fine-tuning"""
        if percentage == 0:
            return None, self.target_data
        
        n_samples = int(len(self.target_data) * percentage / 100)
        indices = np.random.choice(len(self.target_data), n_samples, replace=False)
        
        train_indices = indices
        test_mask = np.ones(len(self.target_data), dtype=bool)
        test_mask[train_indices] = False
        test_indices = np.where(test_mask)[0]
        
        X_train = self.target_data[train_indices]
        X_test = self.target_data[test_indices]
        
        return X_train, X_test
    
    def _evaluate_model(self, model, X, use_source_scaler=True):
        """
        Calculate clustering performance metrics
        
        Parameters:
        -----------
        model : sklearn model
            The clustering model to evaluate
        X : array
            Data to predict on (unscaled)
        use_source_scaler : bool
            Whether to use the source domain scaler (True for zero-shot transfer)
            or the model was trained without scaling (False for fine-tuned models)
        """
        # Apply scaling if needed
        if use_source_scaler and self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X
        
        # Predict clusters
        start_time = time.time()
        y_pred = model.predict(X_scaled)
        prediction_time = time.time() - start_time
        
        # Check number of unique clusters
        n_clusters_found = len(np.unique(y_pred))
        
        # Calculate metrics (handle case where only 1 cluster is predicted)
        # Use scaled data for metric calculation if scaler was used
        X_for_metrics = X_scaled
        
        if n_clusters_found < 2:
            # Can't calculate these metrics with only 1 cluster
            if self.verbose:
                print(f"⚠️  WARNING: Only {n_clusters_found} cluster predicted. Using fallback metrics.")
            metrics = {
                'silhouette_score': -1.0,  # Worst possible score
                'davies_bouldin_index': 999.9,  # Worst possible score
                'calinski_harabasz': 0.0,  # Worst possible score
                'inertia': model.inertia_ if hasattr(model, 'inertia_') else 0.0,
                'n_clusters': n_clusters_found,
                'prediction_time': prediction_time,
                'n_samples': len(X)
            }
        else:
            metrics = {
                'silhouette_score': silhouette_score(X_for_metrics, y_pred),
                'davies_bouldin_index': davies_bouldin_score(X_for_metrics, y_pred),
                'calinski_harabasz': calinski_harabasz_score(X_for_metrics, y_pred),
                'inertia': model.inertia_ if hasattr(model, 'inertia_') else 0.0,
                'n_clusters': n_clusters_found,
                'prediction_time': prediction_time,
                'n_samples': len(X)
            }
        
        return metrics, y_pred
    
    def test_1_zero_shot_transfer(self):
        """Test 1: Transfer source model as-is (0% target data)"""
        test_name = "Test 1"
        
        if self.verbose:
            print(f"\n{'-'*80}")
            print(f"{test_name}: Zero-Shot Transfer")
            print(f"{'-'*80}")
        
        # Use source model directly on target data
        metrics, predictions = self._evaluate_model(self.source_model, self.target_data)
        
        result = {
            'pair_number': self.pair_number,
            'pair_name': self.pair_info['name'],
            'test_number': 1,
            'test_name': TEST_PROTOCOL[test_name]['name'],
            'target_data_percentage': 0,
            'training_mode': 'transfer_only',
            'training_time': 0.0,  # No training
            **metrics
        }
        
        if self.verbose:
            print(f"Silhouette Score: {metrics['silhouette_score']:.4f}")
            print(f"Davies-Bouldin Index: {metrics['davies_bouldin_index']:.4f}")
            print(f"Clusters found: {metrics['n_clusters']}")
        
        self.results.append(result)
        return result
    
    def test_2_finetune_10(self):
        """Test 2: Fine-tune with 10% target data"""
        return self._finetune_test(test_num=2, percentage=10)
    
    def test_3_finetune_20(self):
        """Test 3: Fine-tune with 20% target data"""
        return self._finetune_test(test_num=3, percentage=20)
    
    def test_4_finetune_50(self):
        """Test 4: Fine-tune with 50% target data"""
        return self._finetune_test(test_num=4, percentage=50)
    
    def _finetune_test(self, test_num, percentage):
        """Generic fine-tuning test"""
        test_name = f"Test {test_num}"
        
        if self.verbose:
            print(f"\n{'-'*80}")
            print(f"{test_name}: Fine-tune with {percentage}% target data")
            print(f"{'-'*80}")
        
        # Split target data
        X_train, X_test = self._split_target_data(percentage)
        
        # For fine-tuning, we DON'T use source scaler
        # Instead, train directly on target data (unscaled)
        # This allows the model to adapt to target data's actual scale
        
        # Create new model with same parameters as source
        n_clusters = self.source_model.n_clusters
        model = KMeans(
            n_clusters=n_clusters,
            random_state=CLUSTERING_PARAMS['random_state'],
            max_iter=CLUSTERING_PARAMS['max_iter'],
            n_init=CLUSTERING_PARAMS['n_init']
        )
        
        # Fine-tune on target training data (unscaled - fresh start)
        start_time = time.time()
        model.fit(X_train)
        training_time = time.time() - start_time
        
        # Evaluate on target test data (no source scaler)
        metrics, predictions = self._evaluate_model(model, X_test, use_source_scaler=False)
        
        result = {
            'pair_number': self.pair_number,
            'pair_name': self.pair_info['name'],
            'test_number': test_num,
            'test_name': TEST_PROTOCOL[test_name]['name'],
            'target_data_percentage': percentage,
            'training_mode': 'finetune',
            'training_time': training_time,
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            **metrics
        }
        
        if self.verbose:
            print(f"Training samples: {len(X_train)}")
            print(f"Test samples: {len(X_test)}")
            print(f"Training time: {training_time:.2f}s")
            print(f"Silhouette Score: {metrics['silhouette_score']:.4f}")
            print(f"Davies-Bouldin Index: {metrics['davies_bouldin_index']:.4f}")
        
        self.results.append(result)
        return result
    
    def test_5_from_scratch(self):
        """Test 5: Train from scratch on 100% target data"""
        test_name = "Test 5"
        
        if self.verbose:
            print(f"\n{'-'*80}")
            print(f"{test_name}: Train from scratch on 100% target data")
            print(f"{'-'*80}")
        
        # Train on unscaled target data (fresh start, no source influence)
        best_k = self._find_optimal_k(self.target_data)
        
        # Train new model from scratch
        model = KMeans(
            n_clusters=best_k,
            random_state=CLUSTERING_PARAMS['random_state'],
            max_iter=CLUSTERING_PARAMS['max_iter'],
            n_init=CLUSTERING_PARAMS['n_init']
        )
        
        start_time = time.time()
        model.fit(self.target_data)
        training_time = time.time() - start_time
        
        # Evaluate on same data (no source scaler)
        metrics, predictions = self._evaluate_model(model, self.target_data, use_source_scaler=False)
        
        result = {
            'pair_number': self.pair_number,
            'pair_name': self.pair_info['name'],
            'test_number': 5,
            'test_name': TEST_PROTOCOL[test_name]['name'],
            'target_data_percentage': 100,
            'training_mode': 'from_scratch',
            'training_time': training_time,
            'optimal_k': best_k,
            **metrics
        }
        
        if self.verbose:
            print(f"Optimal k: {best_k}")
            print(f"Training time: {training_time:.2f}s")
            print(f"Silhouette Score: {metrics['silhouette_score']:.4f}")
            print(f"Davies-Bouldin Index: {metrics['davies_bouldin_index']:.4f}")
        
        self.results.append(result)
        return result
    
    def _find_optimal_k(self, X):
        """Find optimal number of clusters using silhouette score"""
        best_k = 3
        best_score = -1
        
        for k in CLUSTERING_PARAMS['k_range']:
            model = KMeans(n_clusters=k, random_state=CLUSTERING_PARAMS['random_state'])
            model.fit(X)
            score = silhouette_score(X, model.labels_)
            
            if score > best_score:
                best_score = score
                best_k = k
        
        return best_k
    
    def run_all_tests(self):
        """Run all 5 tests for this domain pair"""
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"RUNNING ALL TESTS FOR {self.pair_info['name']}")
            print(f"{'='*80}")
        
        start_time = time.time()
        
        # Run all 5 tests
        self.test_1_zero_shot_transfer()
        self.test_2_finetune_10()
        self.test_3_finetune_20()
        self.test_4_finetune_50()
        self.test_5_from_scratch()
        
        total_time = time.time() - start_time
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"COMPLETED ALL TESTS FOR PAIR {self.pair_number}")
            print(f"Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
            print(f"{'='*80}")
        
        return self.results
    
    def save_results(self, output_dir=None):
        """Save results to CSV"""
        if output_dir is None:
            output_dir = PATHS['results_dir']
        
        os.makedirs(output_dir, exist_ok=True)
        
        df = pd.DataFrame(self.results)
        output_path = f"{output_dir}/experiment_pair{self.pair_number}_results.csv"
        df.to_csv(output_path, index=False)
        
        if self.verbose:
            print(f"\n✓ Results saved to: {output_path}")
        
        return output_path


def run_single_pair(pair_number, verbose=True):
    """Run all tests for a single domain pair"""
    experiment = TransferLearningExperiment(pair_number, verbose=verbose)
    results = experiment.run_all_tests()
    experiment.save_results()
    return results


def run_all_pairs(verbose=True):
    """Run all tests for all 7 domain pairs"""
    print("\n" + "="*80)
    print("TRANSFER LEARNING EXPERIMENT: ALL 7 DOMAIN PAIRS")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total experiments: 35 (7 pairs × 5 tests)")
    print("="*80)
    
    all_results = []
    start_time = time.time()
    
    for pair_num in range(1, 8):  # Pairs 1-7
        print(f"\n\n{'#'*80}")
        print(f"# PAIR {pair_num} / 7")
        print(f"{'#'*80}")
        
        results = run_single_pair(pair_num, verbose=verbose)
        all_results.extend(results)
    
    total_time = time.time() - start_time
    
    # Save combined results
    os.makedirs(PATHS['results_dir'], exist_ok=True)
    df_all = pd.DataFrame(all_results)
    output_path = f"{PATHS['results_dir']}/ALL_EXPERIMENTS_RESULTS.csv"
    df_all.to_csv(output_path, index=False)
    
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETED!")
    print("="*80)
    print(f"Total experiments: {len(all_results)}")
    print(f"Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n✓ Combined results saved to: {output_path}")
    print("="*80)
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run transfer learning experiments')
    parser.add_argument('--pair', type=int, help='Run specific pair number (1-7)')
    parser.add_argument('--test', type=int, help='Run specific test number (1-5)')
    parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')
    
    args = parser.parse_args()
    
    verbose = not args.quiet
    
    if args.pair:
        # Run specific pair
        if args.pair < 1 or args.pair > 7:
            print("Error: Pair number must be between 1 and 7")
        else:
            run_single_pair(args.pair, verbose=verbose)
    else:
        # Run all pairs
        run_all_pairs(verbose=verbose)
