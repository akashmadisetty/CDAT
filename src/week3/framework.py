"""
Transfer Learning Framework - Core Implementation
=================================================
Main framework class that integrates metrics, decision engine, and transfer execution

This is the primary interface for:
1. Calculating transferability between domains
2. Getting strategy recommendations
3. Executing transfer learning workflows

Author: Member 3 (Research Lead)
Date: Week 3-4, 2024
"""

import os
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from metrics import TransferabilityMetrics, quick_transferability_check
from decision_engine import (DecisionEngine, TransferRecommendation, 
                             TransferStrategy, TransferabilityLevel)

# Import Week 2 metric functions for compatibility
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'week2'))
try:
    from calculate_transferability_with_rfm import (
        calculate_mmd,
        calculate_js_divergence,
        calculate_correlation_stability,
        calculate_ks_statistic,
        calculate_wasserstein
    )
    WEEK2_METRICS_AVAILABLE = True
except ImportError:
    WEEK2_METRICS_AVAILABLE = False
    print("⚠️  Week 2 metrics not found, using built-in metrics")


class TransferLearningFramework:
    """
    Complete Transfer Learning Framework for Customer Segmentation
    
    Integrates:
    - Transferability assessment (metrics.py)
    - Strategy recommendation (decision_engine.py)
    - Transfer execution (model loading and adaptation)
    """
    
    def __init__(self, 
                 source_model=None,
                 source_data=None,
                 target_data=None,
                 model_path: Optional[str] = None,
                 rfm_features: list = None,
                 use_learned_weights: bool = True):
        """
        Initialize the Transfer Learning Framework
        
        Parameters:
        -----------
        source_model : sklearn model or path, optional
            Trained source domain model (or path to .pkl file)
        source_data : pandas.DataFrame, optional
            Source domain RFM data
        target_data : pandas.DataFrame, optional
            Target domain RFM data
        model_path : str, optional
            Path to load pre-trained model
        rfm_features : list, optional
            List of RFM feature names (default: ['Recency', 'Frequency', 'Monetary'])
        use_learned_weights : bool, optional
            Use learned weights from calibration if available (default: True)
        """
        self.source_model = source_model
        self.source_data = source_data
        self.target_data = target_data
        self.use_learned_weights = use_learned_weights
        
        # Default RFM features
        self.rfm_features = rfm_features or ['Recency', 'Frequency', 'Monetary']
        
        # Initialize components
        self.metrics_calculator = TransferabilityMetrics()
        self.decision_engine = DecisionEngine()
        
        # Storage for results
        self.transferability_metrics = None
        self.composite_score = None
        self.recommendation = None
        
        # Load model if path provided
        if model_path is not None:
            self.load_source_model(model_path)
    
    def load_source_model(self, model_path: str):
        """
        Load a pre-trained source domain model
        
        Parameters:
        -----------
        model_path : str
            Path to .pkl file containing the trained model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            loaded_obj = pickle.load(f)
        
        # Handle case where pickle file contains a dict with 'model' key
        if isinstance(loaded_obj, dict):
            if 'model' in loaded_obj:
                self.source_model = loaded_obj['model']
            elif 'kmeans_model' in loaded_obj:
                self.source_model = loaded_obj['kmeans_model']
            else:
                # Assume the dict IS the model info, take first model-like value
                for key, value in loaded_obj.items():
                    if hasattr(value, 'predict'):
                        self.source_model = value
                        break
                else:
                    raise ValueError(f"Loaded dict does not contain a model object. Keys: {list(loaded_obj.keys())}")
        else:
            # Direct model object
            self.source_model = loaded_obj
        
        print(f"✓ Loaded source model from: {model_path}")
    
    def load_data(self, 
                  source_path: str, 
                  target_path: str,
                  validate: bool = True):
        """
        Load source and target domain RFM data
        
        Parameters:
        -----------
        source_path : str
            Path to source domain RFM CSV file
        target_path : str
            Path to target domain RFM CSV file
        validate : bool
            Whether to validate data (check for required columns, etc.)
        """
        self.source_data = pd.read_csv(source_path)
        self.target_data = pd.read_csv(target_path)
        
        print(f"✓ Loaded source data: {len(self.source_data)} customers")
        print(f"✓ Loaded target data: {len(self.target_data)} customers")
        
        if validate:
            self._validate_data()
    
    def _validate_data(self):
        """Validate that data has required RFM features"""
        for feature in self.rfm_features:
            if feature not in self.source_data.columns:
                raise ValueError(f"Source data missing required feature: {feature}")
            if feature not in self.target_data.columns:
                raise ValueError(f"Target data missing required feature: {feature}")
        
        print("✓ Data validation passed")
    
    def calculate_transferability(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Calculate all transferability metrics between source and target domains
        
        Parameters:
        -----------
        verbose : bool
            Whether to print detailed results
            
        Returns:
        --------
        results : dict
            Dictionary containing all metrics and composite score
        """
        if self.source_data is None or self.target_data is None:
            raise ValueError("Source and target data must be loaded first. Use load_data()")
        
        # Extract RFM features
        X_source = self.source_data[self.rfm_features]
        X_target = self.target_data[self.rfm_features]
        
        # Calculate all metrics
        self.transferability_metrics = self.metrics_calculator.calculate_all_metrics(
            X_source, X_target, scale=True
        )
        
        # Compute composite score
        self.composite_score = self.metrics_calculator.compute_composite_score(
            self.transferability_metrics,
            use_learned_weights=self.use_learned_weights
        )
        
        if verbose:
            self._print_transferability_report()
        
        return {
            'metrics': self.transferability_metrics,
            'composite_score': self.composite_score
        }
    
    def _print_transferability_report(self):
        """Print a detailed transferability report"""
        print("\n" + "="*70)
        print("TRANSFERABILITY ANALYSIS REPORT")
        print("="*70)
        
        print(f"\n📊 Dataset Information:")
        print(f"   Source customers: {self.transferability_metrics['n_source_samples']:,}")
        print(f"   Target customers: {self.transferability_metrics['n_target_samples']:,}")
        print(f"   Features: {self.transferability_metrics['n_features']} (RFM)")
        
        print(f"\n📈 Individual Metrics:")
        print(f"   MMD (Maximum Mean Discrepancy):     {self.transferability_metrics['mmd']:.4f}")
        print(f"      → Lower is better. Range: [0, ∞), typical [0, 2]")
        
        print(f"\n   JS Divergence (Jensen-Shannon):      {self.transferability_metrics['js_divergence']:.4f}")
        print(f"      → Lower is better. Range: [0, 1]")
        
        print(f"\n   Correlation Stability:               {self.transferability_metrics['correlation_stability']:.4f}")
        print(f"      → Higher is better. Range: [0, 1]")
        
        print(f"\n   KS Statistic (Kolmogorov-Smirnov):   {self.transferability_metrics['ks_statistic']:.4f}")
        print(f"      → Lower is better. Range: [0, 1]")
        
        print(f"\n   Wasserstein Distance:                {self.transferability_metrics['wasserstein_distance']:.4f}")
        print(f"      → Lower is better. Range: [0, ∞)")
        
        print(f"\n{'='*70}")
        print(f"🎯 COMPOSITE TRANSFERABILITY SCORE: {self.composite_score:.4f}")
        print(f"{'='*70}\n")
    
    def calculate_confidence_interval(self, n_bootstrap=500, confidence_level=0.95, verbose=True):
        """
        Calculate confidence interval for transferability score using bootstrap resampling.
        
        Parameters:
        -----------
        n_bootstrap : int
            Number of bootstrap samples (default: 5000)
        confidence_level : float
            Confidence level (default: 0.95 for 95% CI)
        verbose : bool
            Print progress (default: True)
        
        Returns:
        --------
        dict : {
            'score': mean score,
            'ci_lower': lower bound,
            'ci_upper': upper bound,
            'std_error': standard error,
            'sample_scores': all bootstrap scores
        }
        """
        if self.source_data is None or self.target_data is None:
            raise ValueError("Load source and target data first using load_data()")
        
        if verbose:
            print(f"\n🔄 Calculating {confidence_level*100:.0f}% confidence interval...")
            print(f"   Bootstrap samples: {n_bootstrap}")
        
        bootstrap_scores = []
        
        for i in range(n_bootstrap):
            if verbose and (i+1) % 200 == 0:
                print(f"   Progress: {i+1}/{n_bootstrap}")
            
            # Resample with replacement
            source_sample = self.source_data.sample(n=len(self.source_data), replace=True)
            target_sample = self.target_data.sample(n=len(self.target_data), replace=True)
            
            # Create temporary framework instance
            temp_framework = TransferLearningFramework(
                source_model=self.source_model,
                source_data=source_sample,
                target_data=target_sample,
                use_learned_weights=self.use_learned_weights
            )
            
            # Calculate transferability
            temp_framework.calculate_transferability(verbose=False)
            bootstrap_scores.append(temp_framework.composite_score)
        
        # Calculate statistics
        bootstrap_scores = np.array(bootstrap_scores)
        mean_score = np.mean(bootstrap_scores)
        std_error = np.std(bootstrap_scores)
        
        # Calculate confidence interval
        alpha = 1 - confidence_level
        ci_lower = np.percentile(bootstrap_scores, alpha/2 * 100)
        ci_upper = np.percentile(bootstrap_scores, (1 - alpha/2) * 100)
        
        self.confidence_interval = {
            'score': mean_score,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'std_error': std_error,
            'confidence_level': confidence_level,
            'n_bootstrap': n_bootstrap,
            'sample_scores': bootstrap_scores
        }
        
        if verbose:
            print(f"\n📊 Confidence Interval Results:")
            print(f"   Mean Score: {mean_score:.4f}")
            print(f"   {confidence_level*100:.0f}% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
            print(f"   Standard Error: {std_error:.4f}")
            print(f"   Original Score: {self.composite_score:.4f}")
        
        return self.confidence_interval
    
    def get_confidence_interval_summary(self):
        """Get formatted confidence interval summary"""
        if not hasattr(self, 'confidence_interval'):
            return "Confidence interval not calculated. Run calculate_confidence_interval() first."
        
        ci = self.confidence_interval
        return (
            f"Score: {ci['score']:.4f} "
            f"({ci['confidence_level']*100:.0f}% CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}], "
            f"SE: {ci['std_error']:.4f})"
        )
    
    def recommend_strategy(self, verbose: bool = True) -> TransferRecommendation:
        """
        Get transfer learning strategy recommendation
        
        Parameters:
        -----------
        verbose : bool
            Whether to print the recommendation
            
        Returns:
        --------
        recommendation : TransferRecommendation
            Complete recommendation with strategy, confidence, and reasoning
        """
        if self.transferability_metrics is None or self.composite_score is None:
            raise ValueError("Calculate transferability first using calculate_transferability()")
        
        # Get recommendation from decision engine
        self.recommendation = self.decision_engine.recommend_strategy(
            self.composite_score,
            self.transferability_metrics
        )
        
        if verbose:
            print(self.recommendation)
        
        return self.recommendation
    
    def execute_transfer(self, 
                        strategy: Optional[TransferStrategy] = None,
                        target_data_fraction: Optional[float] = None) -> Any:
        """
        Execute the transfer learning workflow
        
        Parameters:
        -----------
        strategy : TransferStrategy, optional
            Strategy to use. If None, uses recommended strategy
        target_data_fraction : float, optional
            Fraction of target data to use for fine-tuning (0.0 to 1.0)
            If None, uses recommended percentage
            
        Returns:
        --------
        transferred_model : sklearn model
            The transferred/fine-tuned model ready for use
        """
        if self.source_model is None:
            raise ValueError("No source model available. Load or train a source model first.")
        
        if self.target_data is None:
            raise ValueError("No target data available. Load target data first.")
        
        # Use recommendation if strategy not specified
        if strategy is None:
            if self.recommendation is None:
                self.recommend_strategy(verbose=False)
            strategy = self.recommendation.strategy
        
        # Use recommended data fraction if not specified
        if target_data_fraction is None:
            if self.recommendation is None:
                self.recommend_strategy(verbose=False)
            target_data_fraction = self.recommendation.target_data_percentage / 100.0
        
        print(f"\n🚀 Executing Transfer Strategy: {strategy.value}")
        print(f"   Using {target_data_fraction*100:.1f}% of target data\n")
        
        # Execute based on strategy
        if strategy == TransferStrategy.TRANSFER_AS_IS:
            return self._transfer_as_is()
        
        elif strategy in [TransferStrategy.FINE_TUNE_LIGHT, 
                         TransferStrategy.FINE_TUNE_MODERATE,
                         TransferStrategy.FINE_TUNE_HEAVY]:
            return self._fine_tune_model(target_data_fraction)
        
        elif strategy == TransferStrategy.TRAIN_FROM_SCRATCH:
            return self._train_from_scratch()
        
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    def _transfer_as_is(self):
        """Transfer the source model directly without modification"""
        print("✓ Transferring source model as-is (no fine-tuning)")
        print("  The source model will be used directly on target domain")
        
        # Return a copy of the source model
        return self.source_model
    
    def _fine_tune_model(self, data_fraction: float):
        """
        Fine-tune the source model on target data
        
        Parameters:
        -----------
        data_fraction : float
            Fraction of target data to use (0.0 to 1.0)
        """
        from sklearn.base import clone
        
        print(f"🔄 Fine-tuning source model...")
        print(f"   Using {data_fraction*100:.1f}% of target data ({int(len(self.target_data)*data_fraction)} customers)")
        
        # Sample target data
        n_samples = int(len(self.target_data) * data_fraction)
        target_sample = self.target_data.sample(n=n_samples, random_state=42)
        
        # Extract features
        X_target = target_sample[self.rfm_features].values
        
        # Create a copy of the source model
        fine_tuned_model = clone(self.source_model)
        
        # Re-train on target data (fine-tuning)
        # Note: This assumes the model has a fit() method (scikit-learn compatible)
        if hasattr(fine_tuned_model, 'fit'):
            fine_tuned_model.fit(X_target)
            print("✓ Fine-tuning complete")
        else:
            print("⚠ Warning: Model does not have fit() method. Returning source model.")
            return self.source_model
        
        return fine_tuned_model
    
    def _train_from_scratch(self):
        """Train a new model from scratch on target data"""
        from sklearn.cluster import KMeans
        
        print("🔨 Training new model from scratch on 100% target data...")
        
        X_target = self.target_data[self.rfm_features].values
        
        # Train new K-Means model (assuming same algorithm as source)
        # Use same k as source model if possible
        if hasattr(self.source_model, 'n_clusters'):
            k = self.source_model.n_clusters
        else:
            k = 5  # Default
        
        new_model = KMeans(n_clusters=k, random_state=42, n_init=10)
        new_model.fit(X_target)
        
        print(f"✓ New model trained with k={k} clusters")
        
        return new_model
    
    def evaluate_transfer(self, transferred_model, metric='silhouette'):
        """
        Evaluate the transferred model on target data
        
        Parameters:
        -----------
        transferred_model : sklearn model
            The transferred/fine-tuned model
        metric : str
            Evaluation metric ('silhouette', 'davies_bouldin', 'calinski_harabasz')
            
        Returns:
        --------
        score : float
            Evaluation score on target domain
        """
        from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
        
        X_target = self.target_data[self.rfm_features].values
        labels = transferred_model.predict(X_target)
        
        if metric == 'silhouette':
            score = silhouette_score(X_target, labels)
        elif metric == 'davies_bouldin':
            score = davies_bouldin_score(X_target, labels)
        elif metric == 'calinski_harabasz':
            score = calinski_harabasz_score(X_target, labels)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        print(f"\n📊 Evaluation on Target Domain:")
        print(f"   {metric.replace('_', ' ').title()} Score: {score:.4f}")
        
        return score
    
    def save_results(self, output_dir: str, pair_name: str):
        """
        Save analysis results to files
        
        Parameters:
        -----------
        output_dir : str
            Directory to save results
        pair_name : str
            Name/identifier for this domain pair
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics
        metrics_df = pd.DataFrame([self.transferability_metrics])
        metrics_df['pair_name'] = pair_name
        metrics_df['composite_score'] = self.composite_score
        metrics_path = os.path.join(output_dir, f'{pair_name}_transferability_metrics.csv')
        metrics_df.to_csv(metrics_path, index=False)
        print(f"✓ Saved metrics to: {metrics_path}")
        
        # Save recommendation
        if self.recommendation:
            rec_dict = {
                'pair_name': pair_name,
                'transferability_level': self.recommendation.transferability_level.value,
                'strategy': self.recommendation.strategy.value,
                'composite_score': self.recommendation.composite_score,
                'confidence': self.recommendation.confidence,
                'target_data_percentage': self.recommendation.target_data_percentage,
                'reasoning': self.recommendation.reasoning,
                'expected_performance': self.recommendation.expected_performance
            }
            rec_df = pd.DataFrame([rec_dict])
            rec_path = os.path.join(output_dir, f'{pair_name}_recommendation.csv')
            rec_df.to_csv(rec_path, index=False)
            print(f"✓ Saved recommendation to: {rec_path}")
    
    def compare_with_baseline(self, baseline_score: float, transferred_score: float):
        """
        Compare transferred model performance with baseline
        
        Parameters:
        -----------
        baseline_score : float
            Baseline model score (e.g., trained from scratch on target)
        transferred_score : float
            Transferred model score
        """
        improvement = ((transferred_score - baseline_score) / baseline_score) * 100
        
        print("\n" + "="*70)
        print("TRANSFER PERFORMANCE COMPARISON")
        print("="*70)
        print(f"Baseline (train from scratch): {baseline_score:.4f}")
        print(f"Transferred model:             {transferred_score:.4f}")
        print(f"Improvement:                   {improvement:+.2f}%")
        
        if improvement > 0:
            print("\n✅ Transfer learning successful - outperforms baseline")
        elif improvement > -5:
            print("\n⚠️  Transfer comparable to baseline - marginal benefit")
        else:
            print("\n❌ Negative transfer - baseline is better")
        print("="*70 + "\n")


def quick_transfer_assessment(source_rfm_path: str, 
                              target_rfm_path: str,
                              pair_name: str = "Domain Pair"):
    """
    Convenience function for quick transfer assessment
    
    Parameters:
    -----------
    source_rfm_path : str
        Path to source RFM CSV file
    target_rfm_path : str
        Path to target RFM CSV file
    pair_name : str
        Name for this domain pair
        
    Returns:
    --------
    recommendation : TransferRecommendation
        Strategy recommendation
    """
    print(f"\n🔍 Quick Transfer Assessment: {pair_name}")
    print("="*70 + "\n")
    
    # Initialize framework
    framework = TransferLearningFramework()
    
    # Load data
    framework.load_data(source_rfm_path, target_rfm_path)
    
    # Calculate transferability
    framework.calculate_transferability(verbose=True)
    
    # Get recommendation
    recommendation = framework.recommend_strategy(verbose=True)
    
    return recommendation


if __name__ == "__main__":
    print("Transfer Learning Framework - Core Module")
    print("="*70)
    print("\nThis is the main framework class for transfer learning in customer segmentation.")
    print("\nQuick start:")
    print("  from framework import TransferLearningFramework")
    print("  fw = TransferLearningFramework()")
    print("  fw.load_data('source_RFM.csv', 'target_RFM.csv')")
    print("  fw.calculate_transferability()")
    print("  fw.recommend_strategy()")
