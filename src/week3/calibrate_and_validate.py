"""
Framework Calibration & Validation for 7 Domain Pairs
======================================================
Uses experimental results to calibrate thresholds and validate framework accuracy

This script:
1. Loads results from all 35 experiments (7 pairs × 5 tests)
2. Analyzes correlation between predicted transferability and actual performance
3. Recalibrates HIGH/MODERATE/LOW thresholds
4. Validates framework recommendations
5. Generates accuracy report

Author: Member 3 (Research Lead)
Date: Week 3-4, Updated for 7 pairs
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import LeaveOneOut
import os
import sys
import json
from pathlib import Path

# Import framework components
from experiment_config import DOMAIN_PAIRS, PATHS
from decision_engine import DecisionEngine, TransferabilityLevel, TransferStrategy
from metrics import TransferabilityMetrics

sns.set_style("whitegrid")


class FrameworkCalibrator:
    """Calibrate and validate the transfer learning framework using experimental results"""
    
    def __init__(self, results_file=None):
        if results_file is None:
            results_file = f"{PATHS['results_dir']}/ALL_EXPERIMENTS_RESULTS.csv"
        
        self.results_file = results_file
        self.df = None
        self.calibration_data = None
        self.optimal_thresholds = None
        self.optimal_weights = None
        self.learned_weights = None
        self.metrics_calculator = TransferabilityMetrics()
        
    def load_results(self):
        """Load experimental results"""
        if not os.path.exists(self.results_file):
            raise FileNotFoundError(f"Results file not found: {self.results_file}")
        
        self.df = pd.read_csv(self.results_file)
        
        # Add transferability scores
        self.df['transferability_score'] = self.df['pair_number'].apply(
            lambda x: DOMAIN_PAIRS[x]['transferability_score']
        )
        self.df['expected_category'] = self.df['pair_number'].apply(
            lambda x: DOMAIN_PAIRS[x]['expected_transferability']
        )
        
        print(f"✓ Loaded {len(self.df)} experiment results from {self.df['pair_number'].nunique()} domain pairs")
        return self.df
    
    def prepare_calibration_data(self):
        """Extract key metrics for calibration"""
        calibration = []
        
        for pair_num in sorted(self.df['pair_number'].unique()):
            df_pair = self.df[self.df['pair_number'] == pair_num]
            pair_info = DOMAIN_PAIRS[pair_num]
            
            # Extract performance metrics
            zero_shot_score = df_pair[df_pair['test_number'] == 1]['silhouette_score'].values[0]
            finetune_10_score = df_pair[df_pair['test_number'] == 2]['silhouette_score'].values[0]
            finetune_20_score = df_pair[df_pair['test_number'] == 3]['silhouette_score'].values[0]
            finetune_50_score = df_pair[df_pair['test_number'] == 4]['silhouette_score'].values[0]
            from_scratch_score = df_pair[df_pair['test_number'] == 5]['silhouette_score'].values[0]
            
            # Best fine-tune score
            best_finetune = max(finetune_10_score, finetune_20_score, finetune_50_score)
            
            # Determine which fine-tune percentage was best
            if best_finetune == finetune_10_score:
                best_finetune_pct = 10
            elif best_finetune == finetune_20_score:
                best_finetune_pct = 20
            else:
                best_finetune_pct = 50
            
            # Calculate improvements
            improvement_from_finetune = best_finetune - zero_shot_score
            
            # Determine actual best strategy
            if zero_shot_score >= from_scratch_score * 0.95:  # Within 95% of from-scratch
                actual_best_strategy = 'transfer_as_is'
                optimal_data_pct = 0
            elif best_finetune > zero_shot_score and best_finetune >= from_scratch_score * 0.95:
                if best_finetune_pct <= 20:
                    actual_best_strategy = 'fine_tune_light'
                elif best_finetune_pct <= 40:
                    actual_best_strategy = 'fine_tune_moderate'
                else:
                    actual_best_strategy = 'fine_tune_heavy'
                optimal_data_pct = best_finetune_pct
            else:
                actual_best_strategy = 'train_from_scratch'
                optimal_data_pct = 100
            
            calibration.append({
                'pair_number': pair_num,
                'pair_name': pair_info['name'],
                'transferability_score': pair_info['transferability_score'],
                'expected_category': pair_info['expected_transferability'],
                'zero_shot_performance': zero_shot_score,
                'best_finetune_performance': best_finetune,
                'from_scratch_performance': from_scratch_score,
                'improvement_from_finetune': improvement_from_finetune,
                'actual_best_strategy': actual_best_strategy,
                'optimal_data_percentage': optimal_data_pct
            })
        
        self.calibration_data = pd.DataFrame(calibration)
        print(f"\n✓ Prepared calibration data for {len(self.calibration_data)} domain pairs")
        return self.calibration_data
    
    def analyze_correlation(self, save_plot=True):
        """Analyze correlation between predicted transferability and actual performance"""
        print("\n" + "="*80)
        print("CORRELATION ANALYSIS: Predicted vs Actual Performance")
        print("="*80)
        print("\nThis validates whether Week 1's domain similarity predictions")
        print("correlate with Week 3's actual clustering performance.")
        print("High correlation = Good framework predictive power!")
        
        # Correlation with zero-shot performance
        corr_zero, p_zero = pearsonr(
            self.calibration_data['transferability_score'],
            self.calibration_data['zero_shot_performance']
        )
        
        # Correlation with improvement from fine-tuning
        corr_improve, p_improve = pearsonr(
            self.calibration_data['transferability_score'],
            self.calibration_data['improvement_from_finetune']
        )
        
        print(f"\nPredicted Transferability vs Zero-Shot Performance:")
        print(f"  Pearson r = {corr_zero:.4f} (p = {p_zero:.4f})")
        if p_zero < 0.05:
            print(f"  ✓ Statistically significant!")
        
        print(f"\nPredicted Transferability vs Improvement from Fine-tuning:")
        print(f"  Pearson r = {corr_improve:.4f} (p = {p_improve:.4f})")
        if corr_improve < -0.3:
            print(f"  ✓ Negative correlation: Lower transferability → More improvement from fine-tuning!")
        
        # Print detailed table
        # NOTE: "DomainSim" = Domain Similarity (Week 1 prediction)
        #       "ZeroShot-Sil" & "Scratch-Sil" = Actual clustering quality (Week 3 experiments)
        print(f"\n{'Pair':<6} {'Category':<15} {'DomainSim':<12} {'ZeroShot-Sil':<13} {'Scratch-Sil':<12} {'Best Strategy':<20}")
        print("-" * 85)
        for _, row in self.calibration_data.iterrows():
            print(f"{row['pair_number']:<6} {row['expected_category']:<15} "
                  f"{row['transferability_score']:<12.4f} "
                  f"{row['zero_shot_performance']:<13.4f} "
                  f"{row['from_scratch_performance']:<12.4f} "
                  f"{row['actual_best_strategy']:<20}")
        
        if save_plot:
            self._plot_correlation()
        
        return {
            'correlation_zero_shot': corr_zero,
            'p_value_zero_shot': p_zero,
            'correlation_improvement': corr_improve,
            'p_value_improvement': p_improve
        }
    
    def _plot_correlation(self):
        """Create correlation plots"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Plot 1: Transferability vs Zero-Shot Performance
        ax1 = axes[0]
        scatter1 = ax1.scatter(
            self.calibration_data['transferability_score'],
            self.calibration_data['zero_shot_performance'],
            c=self.calibration_data['pair_number'],
            s=150,
            cmap='viridis',
            alpha=0.7
        )
        
        for _, row in self.calibration_data.iterrows():
            ax1.annotate(f"P{row['pair_number']}", 
                        (row['transferability_score'], row['zero_shot_performance']),
                        fontsize=10, ha='center')
        
        # Add trend line
        z = np.polyfit(self.calibration_data['transferability_score'], 
                      self.calibration_data['zero_shot_performance'], 1)
        p = np.poly1d(z)
        x_trend = np.linspace(self.calibration_data['transferability_score'].min(),
                             self.calibration_data['transferability_score'].max(), 100)
        ax1.plot(x_trend, p(x_trend), "r--", alpha=0.5, label='Trend line')
        
        corr, _ = pearsonr(self.calibration_data['transferability_score'],
                          self.calibration_data['zero_shot_performance'])
        ax1.text(0.05, 0.95, f'Pearson r = {corr:.3f}', 
                transform=ax1.transAxes, fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                verticalalignment='top')
        
        ax1.set_xlabel('Predicted Transferability Score', fontsize=12)
        ax1.set_ylabel('Actual Zero-Shot Silhouette Score', fontsize=12)
        ax1.set_title('Predicted vs Actual Performance', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Plot 2: Transferability vs Improvement from Fine-tuning
        ax2 = axes[1]
        scatter2 = ax2.scatter(
            self.calibration_data['transferability_score'],
            self.calibration_data['improvement_from_finetune'],
            c=self.calibration_data['pair_number'],
            s=150,
            cmap='plasma',
            alpha=0.7
        )
        
        for _, row in self.calibration_data.iterrows():
            ax2.annotate(f"P{row['pair_number']}", 
                        (row['transferability_score'], row['improvement_from_finetune']),
                        fontsize=10, ha='center')
        
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        corr2, _ = pearsonr(self.calibration_data['transferability_score'],
                           self.calibration_data['improvement_from_finetune'])
        ax2.text(0.05, 0.95, f'Pearson r = {corr2:.3f}', 
                transform=ax2.transAxes, fontsize=12,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                verticalalignment='top')
        
        ax2.set_xlabel('Predicted Transferability Score', fontsize=12)
        ax2.set_ylabel('Improvement from Fine-tuning', fontsize=12)
        ax2.set_title('Does Low Transferability Mean More Benefit from Fine-tuning?', 
                     fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        output_path = f"{PATHS['visualizations_dir']}/calibration_correlation.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Saved correlation plot: {output_path}")
        plt.close()  # Close instead of show to avoid GUI issues
    
    def calibrate_thresholds(self):
        """Determine optimal thresholds based on actual performance"""
        print("\n" + "="*80)
        print("THRESHOLD CALIBRATION")
        print("="*80)
        
        # Sort by transferability score
        sorted_data = self.calibration_data.sort_values('transferability_score', ascending=False)
        
        # Analyze actual strategies
        strategy_counts = sorted_data['actual_best_strategy'].value_counts()
        print(f"\nActual Best Strategies:")
        for strategy, count in strategy_counts.items():
            print(f"  {strategy}: {count} pairs ({count/len(sorted_data)*100:.1f}%)")
        
        # Find natural breakpoints
        scores = sorted_data['transferability_score'].values
        zero_shot_perf = sorted_data['zero_shot_performance'].values
        
        # Strategy 1: Based on performance thresholds
        # HIGH: Zero-shot performs well (>= 95% of from-scratch)
        # MODERATE: Fine-tuning helps significantly
        # LOW: Need to train from scratch
        
        high_pairs = sorted_data[sorted_data['actual_best_strategy'] == 'transfer_as_is']
        moderate_pairs = sorted_data[sorted_data['actual_best_strategy'].str.contains('fine_tune')]
        low_pairs = sorted_data[sorted_data['actual_best_strategy'] == 'train_from_scratch']
        
        if len(high_pairs) > 0:
            high_threshold = high_pairs['transferability_score'].min()
        else:
            high_threshold = 0.90  # Default
        
        if len(moderate_pairs) > 0:
            moderate_threshold = moderate_pairs['transferability_score'].min()
        else:
            moderate_threshold = 0.80  # Default
        
        print(f"\nCalibrated Thresholds (Strategy-based):")
        print(f"  HIGH threshold:     >= {high_threshold:.4f}")
        print(f"  MODERATE threshold: >= {moderate_threshold:.4f}")
        print(f"  LOW threshold:      <  {moderate_threshold:.4f}")
        
        # Alternative: Quantile-based
        q75 = sorted_data['transferability_score'].quantile(0.75)
        q50 = sorted_data['transferability_score'].quantile(0.50)
        q25 = sorted_data['transferability_score'].quantile(0.25)
        
        print(f"\nQuantile-based Thresholds:")
        print(f"  75th percentile: {q75:.4f}")
        print(f"  50th percentile: {q50:.4f}")
        print(f"  25th percentile: {q25:.4f}")
        
        # Use strategy-based thresholds
        self.optimal_thresholds = {
            'high': high_threshold,
            'moderate': moderate_threshold,
            'low': 0.50  # Minimum acceptable
        }
        
        return self.optimal_thresholds
    
    def learn_composite_weights(self, min_pairs=5):
        """
        Learn optimal weights for composite transferability score
        
        Uses Ridge regression to learn weights that best predict zero-shot performance
        from individual transferability metrics (MMD, JS, correlation, KS, Wasserstein).
        
        Parameters:
        -----------
        min_pairs : int
            Minimum number of domain pairs required for learning (default: 5)
            
        Returns:
        --------
        dict : Learned weights and metadata
        """
        print("\n" + "="*80)
        print("LEARNING COMPOSITE SCORE WEIGHTS")
        print("="*80)
        
        n_pairs = len(self.calibration_data)
        print(f"\nNumber of domain pairs available: {n_pairs}")
        
        if n_pairs < min_pairs:
            print(f"⚠️  WARNING: Only {n_pairs} pairs available (minimum {min_pairs} recommended)")
            print("   Using default weights from metrics.py instead of learning")
            self.learned_weights = None
            return None
        
        # Load RFM data and compute transferability metrics for each pair
        print("\nComputing transferability metrics from RFM data...")
        
        metric_names = ['mmd', 'js_divergence', 'correlation_stability', 'ks_statistic', 'wasserstein_distance']
        metrics_matrix = []
        zero_shot_scores = []
        valid_pairs = []
        
        for _, row in self.calibration_data.iterrows():
            pair_num = int(row['pair_number'])
            src_path = Path(f"{PATHS['data_dir']}/domain_pair{pair_num}_source_RFM.csv")
            tgt_path = Path(f"{PATHS['data_dir']}/domain_pair{pair_num}_target_RFM.csv")
            
            if not (src_path.exists() and tgt_path.exists()):
                print(f"  ⚠️  Skipping Pair {pair_num}: RFM files not found")
                continue
            
            try:
                src_data = pd.read_csv(src_path)
                tgt_data = pd.read_csv(tgt_path)
                
                # Ensure RFM columns exist
                rfm_cols = ['Recency', 'Frequency', 'Monetary']
                if not all(c in src_data.columns and c in tgt_data.columns for c in rfm_cols):
                    print(f"  ⚠️  Skipping Pair {pair_num}: Missing RFM columns")
                    continue
                
                # Calculate metrics
                metrics = self.metrics_calculator.calculate_all_metrics(
                    src_data[rfm_cols], 
                    tgt_data[rfm_cols]
                )
                
                metrics_matrix.append([metrics[k] for k in metric_names])
                zero_shot_scores.append(row['zero_shot_performance'])
                valid_pairs.append(pair_num)
                print(f"  ✓ Pair {pair_num}: Metrics computed")
                
            except Exception as e:
                print(f"  ⚠️  Error computing metrics for Pair {pair_num}: {e}")
                continue
        
        if len(valid_pairs) < min_pairs:
            print(f"\n⚠️  ERROR: Only {len(valid_pairs)} valid pairs (need >= {min_pairs})")
            print("   Cannot learn weights. Using defaults.")
            self.learned_weights = None
            return None
        
        # Convert to numpy arrays
        X = np.array(metrics_matrix)
        y = np.array(zero_shot_scores)
        
        print(f"\n✓ Successfully loaded {len(valid_pairs)} pairs for weight learning")
        print(f"  Valid pairs: {valid_pairs}")
        
        # Transform metrics to similarity scores (as in compute_composite_score)
        mmd_sim = 1 - np.minimum(X[:,0] / 2.0, 1.0)
        js_sim = 1 - X[:,1]
        corr_sim = X[:,2]
        ks_sim = 1 - X[:,3]
        w_sim = 1 - np.minimum(X[:,4] / 1.5, 1.0)
        
        S = np.vstack([mmd_sim, js_sim, corr_sim, ks_sim, w_sim]).T
        
        # Fit Ridge regression with Leave-One-Out CV
        print("\nTraining Ridge regression model...")
        print("  - Target: Zero-shot silhouette score")
        print("  - Features: 5 similarity metrics (MMD, JS, Corr, KS, Wasserstein)")
        print("  - Cross-validation: Leave-One-Out (LOO)")
        
        alphas = np.logspace(-3, 3, 50)
        model = RidgeCV(alphas=alphas, cv=LeaveOneOut())
        model.fit(S, y)
        
        # Get coefficients
        coefs = model.coef_
        intercept = model.intercept_
        
        # Normalize to positive weights summing to 1 (for interpretability)
        coefs_pos = np.maximum(coefs, 0)
        if coefs_pos.sum() > 0:
            weights_normalized = coefs_pos / coefs_pos.sum()
        else:
            # All negative — use absolute values
            weights_normalized = np.abs(coefs) / np.abs(coefs).sum()
        
        # Compute predictions and correlation
        y_pred = model.predict(S)
        r_corr, p_corr = pearsonr(y_pred, y)
        
        # Compute LOO R²
        from sklearn.model_selection import cross_val_score
        loo_scores = cross_val_score(model, S, y, cv=LeaveOneOut(), scoring='r2')
        loo_r2_mean = loo_scores.mean()
        
        print(f"\n✓ Weight learning complete!")
        print(f"\nModel Performance:")
        print(f"  - Correlation (predicted vs zero-shot): r = {r_corr:.4f}, p = {p_corr:.4f}")
        print(f"  - LOO Cross-Validation R²: {loo_r2_mean:.4f}")
        print(f"  - Optimal Ridge alpha: {model.alpha_:.4f}")
        
        print(f"\nLearned Weights (normalized):")
        weights_dict = {}
        for i, name in enumerate(metric_names):
            weight_val = float(weights_normalized[i])
            weights_dict[name] = weight_val
            print(f"  {name:25s}: {weight_val:.4f}")
        
        # Save learned weights
        self.learned_weights = {
            'weights': weights_dict,
            'metadata': {
                'n_pairs': len(valid_pairs),
                'valid_pairs': valid_pairs,
                'correlation_r': float(r_corr),
                'correlation_p': float(p_corr),
                'loo_r2': float(loo_r2_mean),
                'ridge_alpha': float(model.alpha_),
                'ridge_intercept': float(intercept),
                'ridge_coefs_raw': coefs.tolist(),
                'method': 'Ridge regression with LOO CV',
                'target': 'zero_shot_silhouette_score',
                'features': metric_names,
                'date_learned': pd.Timestamp.now().isoformat()
            }
        }
        
        # Save to JSON
        weights_path = Path(PATHS['results_dir']) / 'learned_weights.json'
        with open(weights_path, 'w') as f:
            json.dump(self.learned_weights, f, indent=2)
        
        print(f"\n✓ Saved learned weights to: {weights_path}")
        
        # Also save a copy to src/week3 for easy import
        week3_weights_path = Path(__file__).parent / 'learned_weights.json'
        with open(week3_weights_path, 'w') as f:
            json.dump(self.learned_weights, f, indent=2)
        print(f"✓ Saved copy to: {week3_weights_path}")
        
        return self.learned_weights
    
    def validate_framework(self):
        """Validate framework predictions against actual results"""
        print("\n" + "="*80)
        print("FRAMEWORK VALIDATION")
        print("="*80)
        
        # Create decision engine with calibrated thresholds
        engine = DecisionEngine(
            high_threshold=self.optimal_thresholds['high'],
            moderate_threshold=self.optimal_thresholds['moderate'],
            low_threshold=self.optimal_thresholds['low']
        )
        
        correct_predictions = 0
        total_pairs = len(self.calibration_data)
        
        print(f"\n{'Pair':<6} {'Predicted':<20} {'Actual':<20} {'Correct?':<10}")
        print("-" * 60)
        
        for _, row in self.calibration_data.iterrows():
            # Create placeholder metrics for the decision engine
            # These are estimated from the composite transferability score
            # (In real usage, these would come from actual domain analysis)
            t_score = row['transferability_score']
            placeholder_metrics = {
                # Week1 metrics (estimated from composite score)
                'mmd': (1.0 - t_score) * 2.0,  # Inverse relationship, scale to ~[0, 2]
                'js_divergence': 1.0 - t_score,  # Inverse relationship
                'correlation_stability': t_score,  # Direct relationship
                'ks_statistic': 1.0 - t_score,  # Inverse relationship
                'wasserstein_distance': (1.0 - t_score) * 1.5,  # Inverse, scale to ~[0, 1.5]
                # Sample sizes (use actual from data if available)
                'n_source_samples': 1500,  # Typical source domain size
                'n_target_samples': 1200   # Typical target domain size
            }
            
            # Get framework recommendation
            recommendation = engine.recommend_strategy(
                composite_score=row['transferability_score'],
                metrics=placeholder_metrics
            )
            
            predicted_strategy = recommendation.strategy.value
            actual_strategy = row['actual_best_strategy']
            
            # Check if prediction is correct (allow some flexibility)
            is_correct = self._strategies_match(predicted_strategy, actual_strategy)
            
            if is_correct:
                correct_predictions += 1
                result = "✓"
            else:
                result = "✗"
            
            print(f"{row['pair_number']:<6} {predicted_strategy:<20} {actual_strategy:<20} {result:<10}")
        
        accuracy = (correct_predictions / total_pairs) * 100
        
        print("\n" + "="*80)
        print(f"FRAMEWORK ACCURACY: {correct_predictions}/{total_pairs} = {accuracy:.1f}%")
        print("="*80)
        
        # Save validation results
        def get_prediction(row):
            t_score = row['transferability_score']
            metrics = {
                'mmd': (1.0 - t_score) * 2.0,
                'js_divergence': 1.0 - t_score,
                'correlation_stability': t_score,
                'ks_statistic': 1.0 - t_score,
                'wasserstein_distance': (1.0 - t_score) * 1.5,
                'n_source_samples': 1500,
                'n_target_samples': 1200
            }
            return engine.recommend_strategy(t_score, metrics).strategy.value
        
        validation_df = self.calibration_data.copy()
        validation_df['predicted_strategy'] = validation_df.apply(get_prediction, axis=1)
        
        output_path = f"{PATHS['results_dir']}/framework_validation.csv"
        validation_df.to_csv(output_path, index=False)
        print(f"\n✓ Saved validation results: {output_path}")
        
        return {
            'accuracy': accuracy,
            'correct': correct_predictions,
            'total': total_pairs
        }
    
    def _strategies_match(self, predicted, actual):
        """Check if predicted and actual strategies are similar enough"""
        # Exact match
        if predicted == actual:
            return True
        
        # Allow some flexibility
        fine_tune_strategies = ['fine_tune_light', 'fine_tune_moderate', 'fine_tune_heavy']
        
        if predicted in fine_tune_strategies and actual in fine_tune_strategies:
            return True  # Any fine-tune variant is acceptable
        
        return False
    
    def generate_report(self, save=True):
        """Generate complete calibration and validation report"""
        print("\n" + "="*80)
        print("GENERATING CALIBRATION & VALIDATION REPORT")
        print("="*80)
        
        # Run all analyses
        self.load_results()
        self.prepare_calibration_data()
        correlation_results = self.analyze_correlation()
        threshold_results = self.calibrate_thresholds()
        
        # Learn optimal weights for composite score
        weight_results = self.learn_composite_weights(min_pairs=5)
        
        validation_results = self.validate_framework()
        
        if save:
            report_path = f"{PATHS['results_dir']}/calibration_validation_report.txt"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write("="*80 + "\n")
                f.write("TRANSFER LEARNING FRAMEWORK: CALIBRATION & VALIDATION REPORT\n")
                f.write("="*80 + "\n\n")
                
                f.write(f"Dataset: {len(self.calibration_data)} domain pairs\n")
                f.write(f"Total experiments: {len(self.df)} (7 pairs × 5 tests)\n\n")
                
                f.write("="*80 + "\n")
                f.write("CORRELATION ANALYSIS\n")
                f.write("="*80 + "\n\n")
                f.write(f"Predicted vs Zero-Shot Performance:\n")
                f.write(f"  Pearson r = {correlation_results['correlation_zero_shot']:.4f}\n")
                f.write(f"  p-value = {correlation_results['p_value_zero_shot']:.4f}\n\n")
                
                f.write(f"Predicted vs Improvement from Fine-tuning:\n")
                f.write(f"  Pearson r = {correlation_results['correlation_improvement']:.4f}\n")
                f.write(f"  p-value = {correlation_results['p_value_improvement']:.4f}\n\n")
                
                f.write("="*80 + "\n")
                f.write("LEARNED COMPOSITE WEIGHTS\n")
                f.write("="*80 + "\n\n")
                if weight_results:
                    f.write(f"Method: {weight_results['metadata']['method']}\n")
                    f.write(f"Pairs used: {weight_results['metadata']['n_pairs']}\n")
                    f.write(f"Correlation: r = {weight_results['metadata']['correlation_r']:.4f}, ")
                    f.write(f"p = {weight_results['metadata']['correlation_p']:.4f}\n")
                    f.write(f"LOO R²: {weight_results['metadata']['loo_r2']:.4f}\n\n")
                    f.write("Learned Weights:\n")
                    for metric, weight in weight_results['weights'].items():
                        f.write(f"  {metric:25s}: {weight:.4f}\n")
                    f.write(f"\nWeights saved to: learned_weights.json\n\n")
                else:
                    f.write("Weight learning skipped (insufficient data).\n")
                    f.write("Using default weights from metrics.py\n\n")
                
                f.write("="*80 + "\n")
                f.write("CALIBRATED THRESHOLDS\n")
                f.write("="*80 + "\n\n")
                f.write(f"HIGH transferability:     >= {threshold_results['high']:.4f}\n")
                f.write(f"MODERATE transferability: >= {threshold_results['moderate']:.4f}\n")
                f.write(f"LOW transferability:      <  {threshold_results['moderate']:.4f}\n\n")
                
                f.write("="*80 + "\n")
                f.write("FRAMEWORK VALIDATION\n")
                f.write("="*80 + "\n\n")
                f.write(f"Accuracy: {validation_results['accuracy']:.1f}%\n")
                f.write(f"Correct predictions: {validation_results['correct']}/{validation_results['total']}\n\n")
                
                f.write("="*80 + "\n")
                f.write("RECOMMENDATIONS\n")
                f.write("="*80 + "\n\n")
                
                if validation_results['accuracy'] >= 70:
                    f.write("✓ Framework demonstrates good predictive accuracy (>= 70%)\n")
                    f.write("✓ Thresholds are well-calibrated\n")
                    f.write("✓ Ready for deployment\n")
                else:
                    f.write("⚠ Framework accuracy below 70%\n")
                    f.write("→ Consider adjusting metric weights\n")
                    f.write("→ May need more experimental data\n")
            
            print(f"\n✓ Saved full report: {report_path}")
        
        return {
            'correlation': correlation_results,
            'thresholds': threshold_results,
            'validation': validation_results
        }


if __name__ == "__main__":
    print("="*80)
    print("TRANSFER LEARNING FRAMEWORK CALIBRATION & VALIDATION")
    print("For 7 Domain Pairs")
    print("="*80)
    
    calibrator = FrameworkCalibrator()
    results = calibrator.generate_report(save=True)
    
    print("\n" + "="*80)
    print("CALIBRATION & VALIDATION COMPLETE!")
    print("="*80)
    print(f"\nFramework Accuracy: {results['validation']['accuracy']:.1f}%")
    print(f"Optimal Thresholds:")
    print(f"  HIGH:     >= {results['thresholds']['high']:.4f}")
    print(f"  MODERATE: >= {results['thresholds']['moderate']:.4f}")
    print(f"  LOW:      <  {results['thresholds']['moderate']:.4f}")
