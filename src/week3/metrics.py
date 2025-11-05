"""
Research-Backed Transferability Metrics Module
===============================================
Consolidates all transferability metrics used in Week 1 and Week 2

This module implements state-of-the-art metrics for measuring domain similarity
and predicting transfer learning success in customer segmentation tasks.

Metrics Included:
-----------------
1. Maximum Mean Discrepancy (MMD) - Gold standard in domain adaptation
2. Jensen-Shannon Divergence (JS) - Symmetric KL divergence
3. Kolmogorov-Smirnov Statistic (KS) - Distribution difference test
4. Wasserstein Distance - Earth Mover's Distance
5. Correlation Stability - Feature relationship preservation

References:
-----------
- Gretton et al. (2012): "A Kernel Two-Sample Test"
- Ben-David et al. (2010): "A theory of learning from different domains"
- Pan & Yang (2010): "A survey on transfer learning"

Author: Member 3 (Research Lead)
Date: Week 3-4, 2024
"""

import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import rbf_kernel
import warnings
warnings.filterwarnings('ignore')


class TransferabilityMetrics:
    """
    Unified class for calculating all transferability metrics
    """
    
    def __init__(self, gamma=1.0, n_bins=50):
        """
        Initialize the metrics calculator
        
        Parameters:
        -----------
        gamma : float
            RBF kernel bandwidth for MMD calculation (default: 1.0)
        n_bins : int
            Number of bins for histogram-based metrics (default: 50)
        """
        self.gamma = gamma
        self.n_bins = n_bins
        self.scaler = StandardScaler()
    
    def calculate_mmd(self, X_source, X_target):
        """
        Maximum Mean Discrepancy (MMD)
        
        Measures distribution difference in reproducing kernel Hilbert space (RKHS).
        Lower values indicate more similar distributions → better transferability.
        
        Parameters:
        -----------
        X_source : numpy.ndarray, shape (n_source, n_features)
            Source domain feature matrix
        X_target : numpy.ndarray, shape (n_target, n_features)
            Target domain feature matrix
            
        Returns:
        --------
        mmd : float
            MMD score. Range: [0, ∞), typically [0, 2]
            Lower = more similar = better transferability
            
        Reference:
        ----------
        Gretton et al. (2012) "A Kernel Two-Sample Test"
        """
        n_source = len(X_source)
        n_target = len(X_target)
        
        # Compute RBF kernel matrices
        K_ss = rbf_kernel(X_source, X_source, gamma=self.gamma)
        K_tt = rbf_kernel(X_target, X_target, gamma=self.gamma)
        K_st = rbf_kernel(X_source, X_target, gamma=self.gamma)
        
        # MMD² = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]
        mmd_squared = (
            K_ss.sum() / (n_source * n_source) - 
            2 * K_st.sum() / (n_source * n_target) + 
            K_tt.sum() / (n_target * n_target)
        )
        
        return np.sqrt(max(0, mmd_squared))
    
    def calculate_js_divergence(self, X_source, X_target):
        """
        Jensen-Shannon Divergence (averaged across features)
        
        Symmetric version of KL divergence. Measures information-theoretic
        distance between probability distributions.
        
        Parameters:
        -----------
        X_source : numpy.ndarray, shape (n_source, n_features)
            Source domain feature matrix
        X_target : numpy.ndarray, shape (n_target, n_features)
            Target domain feature matrix
            
        Returns:
        --------
        js_div : float
            Average JS divergence across all features
            Range: [0, 1]
            Lower = more similar = better transferability
            
        Reference:
        ----------
        Lin (1991) "Divergence measures based on the Shannon entropy"
        """
        n_features = X_source.shape[1]
        js_scores = []
        
        for col_idx in range(n_features):
            source_col = X_source[:, col_idx]
            target_col = X_target[:, col_idx]
            
            # Create histogram bins
            bins = np.linspace(
                min(source_col.min(), target_col.min()),
                max(source_col.max(), target_col.max()),
                self.n_bins
            )
            
            # Compute histograms
            source_hist, _ = np.histogram(source_col, bins=bins, density=True)
            target_hist, _ = np.histogram(target_col, bins=bins, density=True)
            
            # Normalize to probability distributions
            source_hist = source_hist / (source_hist.sum() + 1e-10)
            target_hist = target_hist / (target_hist.sum() + 1e-10)
            
            # Add epsilon to avoid log(0)
            source_hist += 1e-10
            target_hist += 1e-10
            
            # Calculate JS divergence
            js_scores.append(jensenshannon(source_hist, target_hist))
        
        return np.mean(js_scores)
    
    def calculate_correlation_stability(self, X_source, X_target):
        """
        Correlation Stability
        
        Measures how similar feature correlation structures are between domains.
        Important for ensuring relationships learned in source domain
        hold in target domain.
        
        Parameters:
        -----------
        X_source : numpy.ndarray, shape (n_source, n_features)
            Source domain feature matrix
        X_target : numpy.ndarray, shape (n_target, n_features)
            Target domain feature matrix
            
        Returns:
        --------
        stability : float
            Correlation stability score
            Range: [0, 1]
            Higher = more stable = better transferability
            
        Reference:
        ----------
        Storkey (2009) "When training and test sets are different"
        """
        if X_source.shape[1] < 2:
            # Perfect stability if only 1 feature (no correlations to compare)
            return 1.0
        
        # Compute correlation matrices
        corr_source = np.corrcoef(X_source.T)
        corr_target = np.corrcoef(X_target.T)
        
        # Compute Frobenius norm of difference
        diff = corr_source - corr_target
        frobenius_norm = np.sqrt(np.sum(diff ** 2))
        
        # Normalize by maximum possible distance
        n_features = X_source.shape[1]
        max_distance = np.sqrt(2 * n_features ** 2)
        
        # Convert to similarity score (1 - normalized distance)
        stability = 1 - (frobenius_norm / max_distance)
        
        return stability
    
    def calculate_ks_statistic(self, X_source, X_target):
        """
        Kolmogorov-Smirnov Statistic (averaged across features)
        
        Non-parametric test for distribution difference.
        Measures maximum distance between CDFs.
        
        Parameters:
        -----------
        X_source : numpy.ndarray, shape (n_source, n_features)
            Source domain feature matrix
        X_target : numpy.ndarray, shape (n_target, n_features)
            Target domain feature matrix
            
        Returns:
        --------
        ks_stat : float
            Average KS statistic across all features
            Range: [0, 1]
            Lower = more similar = better transferability
            
        Reference:
        ----------
        Massey (1951) "The Kolmogorov-Smirnov test for goodness of fit"
        """
        n_features = X_source.shape[1]
        ks_scores = []
        
        for col_idx in range(n_features):
            ks_stat, _ = ks_2samp(X_source[:, col_idx], X_target[:, col_idx])
            ks_scores.append(ks_stat)
        
        return np.mean(ks_scores)
    
    def calculate_wasserstein(self, X_source, X_target):
        """
        Wasserstein Distance (Earth Mover's Distance)
        
        Measures minimum cost to transform one distribution into another.
        Geometric interpretation of distribution difference.
        
        Parameters:
        -----------
        X_source : numpy.ndarray, shape (n_source, n_features)
            Source domain feature matrix
        X_target : numpy.ndarray, shape (n_target, n_features)
            Target domain feature matrix
            
        Returns:
        --------
        w_dist : float
            Average Wasserstein distance across all features
            Range: [0, ∞)
            Lower = more similar = better transferability
            
        Reference:
        ----------
        Arjovsky et al. (2017) "Wasserstein GAN"
        """
        n_features = X_source.shape[1]
        w_distances = []
        
        for col_idx in range(n_features):
            w_dist = wasserstein_distance(
                X_source[:, col_idx], 
                X_target[:, col_idx]
            )
            w_distances.append(w_dist)
        
        return np.mean(w_distances)
    
    def calculate_all_metrics(self, X_source, X_target, scale=True):
        """
        Calculate all transferability metrics at once
        
        Parameters:
        -----------
        X_source : numpy.ndarray or pandas.DataFrame
            Source domain feature matrix
        X_target : numpy.ndarray or pandas.DataFrame
            Target domain feature matrix
        scale : bool
            Whether to standardize features before calculation (default: True)
            
        Returns:
        --------
        metrics : dict
            Dictionary containing all metric values and metadata
        """
        # Convert to numpy arrays if needed
        if hasattr(X_source, 'values'):
            X_source = X_source.values
        if hasattr(X_target, 'values'):
            X_target = X_target.values
        
        # Standardize features if requested
        if scale:
            X_source_scaled = self.scaler.fit_transform(X_source)
            X_target_scaled = self.scaler.transform(X_target)
        else:
            X_source_scaled = X_source
            X_target_scaled = X_target
        
        # Calculate all metrics
        metrics = {
            'mmd': self.calculate_mmd(X_source_scaled, X_target_scaled),
            'js_divergence': self.calculate_js_divergence(X_source_scaled, X_target_scaled),
            'correlation_stability': self.calculate_correlation_stability(X_source_scaled, X_target_scaled),
            'ks_statistic': self.calculate_ks_statistic(X_source_scaled, X_target_scaled),
            'wasserstein_distance': self.calculate_wasserstein(X_source_scaled, X_target_scaled),
            'n_source_samples': len(X_source),
            'n_target_samples': len(X_target),
            'n_features': X_source.shape[1]
        }
        
        return metrics
    
    def compute_composite_score(self, metrics, weights=None):
        """
        Compute a single composite transferability score
        
        Combines multiple metrics into a unified score using weighted average.
        Higher score = better transferability.
        
        Parameters:
        -----------
        metrics : dict
            Dictionary of individual metric values
        weights : dict, optional
            Custom weights for each metric. If None, uses research-backed defaults.
            
        Returns:
        --------
        score : float
            Composite transferability score in range [0, 1]
            Higher = better transferability
        """
        # Default weights based on calibration results
        # These are data-driven weights optimized on Week 2 experimental data
        if weights is None:
            weights = {
                'mmd': 0.30,              # Primary metric - captures overall distribution difference
                'js_divergence': 0.25,    # Information-theoretic distance measure
                'correlation_stability': 0.20,  # Ensures feature relationships transfer
                'ks_statistic': 0.15,     # Non-parametric distribution test
                'wasserstein_distance': 0.10   # Geometric distance measure
            }
        
        # Normalize metrics to [0, 1] range where higher = better
        # For distance metrics (lower is better), we use: score = 1 - normalized_distance
        
        # MMD: typical range [0, 2], invert to similarity
        mmd_similarity = 1 - min(metrics['mmd'] / 2.0, 1.0)
        
        # JS Divergence: range [0, 1], invert to similarity
        js_similarity = 1 - metrics['js_divergence']
        
        # Correlation Stability: already in [0, 1] range, higher is better
        corr_similarity = metrics['correlation_stability']
        
        # KS Statistic: range [0, 1], invert to similarity
        ks_similarity = 1 - metrics['ks_statistic']
        
        # Wasserstein: normalize by typical max value (tunable)
        # Assuming max ~1.5 for normalized RFM features
        w_similarity = 1 - min(metrics['wasserstein_distance'] / 1.5, 1.0)
        
        # Weighted composite score
        composite_score = (
            weights['mmd'] * mmd_similarity +
            weights['js_divergence'] * js_similarity +
            weights['correlation_stability'] * corr_similarity +
            weights['ks_statistic'] * ks_similarity +
            weights['wasserstein_distance'] * w_similarity
        )
        
        return composite_score


def quick_transferability_check(source_data, target_data, features, verbose=True):
    """
    Convenience function for quick transferability assessment
    
    Parameters:
    -----------
    source_data : pandas.DataFrame or numpy.ndarray
        Source domain data
    target_data : pandas.DataFrame or numpy.ndarray
        Target domain data
    features : list or None
        Feature column names (if DataFrames) or None (if arrays)
    verbose : bool
        Whether to print results
        
    Returns:
    --------
    results : dict
        Complete metrics and transferability score
    """
    # Extract features if DataFrames
    if features is not None:
        X_source = source_data[features]
        X_target = target_data[features]
    else:
        X_source = source_data
        X_target = target_data
    
    # Calculate metrics
    calculator = TransferabilityMetrics()
    metrics = calculator.calculate_all_metrics(X_source, X_target)
    composite_score = calculator.compute_composite_score(metrics)
    
    results = {
        'metrics': metrics,
        'composite_score': composite_score
    }
    
    if verbose:
        print("="*70)
        print("TRANSFERABILITY ANALYSIS")
        print("="*70)
        print(f"\nSample Sizes:")
        print(f"  Source: {metrics['n_source_samples']:,} samples")
        print(f"  Target: {metrics['n_target_samples']:,} samples")
        print(f"  Features: {metrics['n_features']}")
        print(f"\nIndividual Metrics:")
        print(f"  MMD:                    {metrics['mmd']:.4f} (lower is better)")
        print(f"  JS Divergence:          {metrics['js_divergence']:.4f} (lower is better)")
        print(f"  Correlation Stability:  {metrics['correlation_stability']:.4f} (higher is better)")
        print(f"  KS Statistic:           {metrics['ks_statistic']:.4f} (lower is better)")
        print(f"  Wasserstein Distance:   {metrics['wasserstein_distance']:.4f} (lower is better)")
        print(f"\n{'='*70}")
        print(f"COMPOSITE TRANSFERABILITY SCORE: {composite_score:.4f}")
        print(f"{'='*70}")
        
        # Simple interpretation
        if composite_score >= 0.85:
            interpretation = "HIGH - Excellent transfer potential"
        elif composite_score >= 0.70:
            interpretation = "MODERATE - Transfer with fine-tuning recommended"
        else:
            interpretation = "LOW - Consider training from scratch"
        
        print(f"\nInterpretation: {interpretation}\n")
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Transferability Metrics Module")
    print("="*70)
    print("\nThis module provides research-backed metrics for assessing")
    print("transfer learning feasibility between customer domains.\n")
    print("Import this module in your framework code:")
    print("  from metrics import TransferabilityMetrics, quick_transferability_check\n")
