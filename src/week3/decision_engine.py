"""
Transfer Learning Decision Engine
==================================
Recommends optimal transfer learning strategies based on transferability scores

This module implements the decision logic for determining when to:
1. Transfer as-is (HIGH transferability)
2. Fine-tune with target data (MODERATE transferability)
3. Train from scratch (LOW transferability)

Includes:
- Threshold-based classification
- Confidence scoring
- Data requirement estimation
- Risk assessment

Author: Member 3 (Research Lead)
Date: Week 3-4, 2024
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Tuple, Optional


class TransferStrategy(Enum):
    """Transfer learning strategy recommendations"""
    TRANSFER_AS_IS = "transfer_as_is"
    FINE_TUNE_LIGHT = "fine_tune_light"      # 10-20% target data
    FINE_TUNE_MODERATE = "fine_tune_moderate"  # 20-40% target data
    FINE_TUNE_HEAVY = "fine_tune_heavy"        # 40-60% target data
    TRAIN_FROM_SCRATCH = "train_from_scratch"


class TransferabilityLevel(Enum):
    """Transferability classification levels"""
    HIGH = "HIGH"
    MODERATE = "MODERATE"
    LOW = "LOW"
    VERY_LOW = "VERY_LOW"


@dataclass
class TransferRecommendation:
    """
    Complete transfer learning recommendation
    """
    strategy: TransferStrategy
    transferability_level: TransferabilityLevel
    composite_score: float
    confidence: float
    target_data_percentage: int
    reasoning: str
    risks: list
    expected_performance: str
    
    def __str__(self):
        return f"""
Transfer Learning Recommendation
{"="*70}
Transferability Level: {self.transferability_level.value}
Composite Score: {self.composite_score:.4f}
Confidence: {self.confidence:.1f}%

Recommended Strategy: {self.strategy.value.replace('_', ' ').title()}
Target Data Required: {self.target_data_percentage}%

Reasoning:
{self.reasoning}

Potential Risks:
{chr(10).join(f'  • {risk}' for risk in self.risks)}

Expected Performance:
{self.expected_performance}
{"="*70}
"""


class DecisionEngine:
    """
    Makes transfer learning decisions based on transferability metrics
    """
    
    def __init__(self, 
                 high_threshold=0.9000,
                 moderate_threshold=0.7254,
                 low_threshold=0.50,
                 metric_weights=None):
        """
        Initialize the decision engine
        
        Parameters:
        -----------
        high_threshold : float
            Minimum score for HIGH transferability (default: 0.9000)
            Calibrated from Week 3 experiments with 7 domain pairs (85.7% accuracy)
            More conservative than original 0.8260 to reduce false positives
        moderate_threshold : float
            Minimum score for MODERATE transferability (default: 0.7254)
            Calibrated from Week 3 experiments with 7 domain pairs
            Based on actual best-performing strategies across all pairs
        low_threshold : float
            Minimum score for LOW transferability (default: 0.50)
            Below this, transfer not recommended
        metric_weights : dict, optional
            Custom weights for individual metrics
        """
        self.high_threshold = high_threshold
        self.moderate_threshold = moderate_threshold
        self.low_threshold = low_threshold
        self.metric_weights = metric_weights
        
        # These thresholds are calibrated using isotonic regression
        # on actual experimental results (Week 2 domain pairs)
        # See calibration.py for the calibration process
    
    def classify_transferability(self, composite_score: float) -> TransferabilityLevel:
        """
        Classify transferability level based on composite score
        
        Parameters:
        -----------
        composite_score : float
            Composite transferability score [0, 1]
            
        Returns:
        --------
        level : TransferabilityLevel
            Classification: HIGH, MODERATE, LOW, or VERY_LOW
        """
        if composite_score >= self.high_threshold:
            return TransferabilityLevel.HIGH
        elif composite_score >= self.moderate_threshold:
            return TransferabilityLevel.MODERATE
        elif composite_score >= self.low_threshold:
            return TransferabilityLevel.LOW
        else:
            return TransferabilityLevel.VERY_LOW
    
    def calculate_confidence(self, metrics: Dict[str, float]) -> float:
        """
        Calculate confidence in the recommendation
        
        Confidence is based on:
        1. Agreement between different metrics
        2. Sample sizes
        3. Metric value stability
        
        Parameters:
        -----------
        metrics : dict
            Dictionary of individual metric values
            
        Returns:
        --------
        confidence : float
            Confidence percentage [0, 100]
        """
        # Convert all metrics to similarity scores (0-1, higher is better)
        mmd_sim = 1 - min(metrics['mmd'] / 2.0, 1.0)
        js_sim = 1 - metrics['js_divergence']
        corr_sim = metrics['correlation_stability']
        ks_sim = 1 - metrics['ks_statistic']
        w_sim = 1 - min(metrics['wasserstein_distance'] / 1.5, 1.0)
        
        similarities = [mmd_sim, js_sim, corr_sim, ks_sim, w_sim]
        
        # 1. Metric agreement: low variance = high confidence
        metric_variance = np.var(similarities)
        agreement_score = 1 - min(metric_variance * 4, 1.0)  # Scale variance
        
        # 2. Sample size confidence: larger samples = higher confidence
        min_samples = min(metrics['n_source_samples'], metrics['n_target_samples'])
        if min_samples >= 1000:
            sample_confidence = 1.0
        elif min_samples >= 500:
            sample_confidence = 0.9
        elif min_samples >= 200:
            sample_confidence = 0.8
        else:
            sample_confidence = 0.6
        
        # 3. Extreme values penalty: very high or very low scores need more caution
        avg_similarity = np.mean(similarities)
        if 0.3 <= avg_similarity <= 0.95:
            stability_score = 1.0
        else:
            stability_score = 0.85
        
        # Combine confidence factors
        confidence = (
            0.5 * agreement_score +
            0.3 * sample_confidence +
            0.2 * stability_score
        ) * 100
        
        return min(confidence, 99.0)  # Cap at 99% (never 100% certain)
    
    def estimate_data_requirement(self, 
                                  transferability_level: TransferabilityLevel,
                                  composite_score: float) -> int:
        """
        Estimate percentage of target data needed for fine-tuning
        
        Parameters:
        -----------
        transferability_level : TransferabilityLevel
            Classification level
        composite_score : float
            Composite transferability score
            
        Returns:
        --------
        percentage : int
            Estimated percentage of target data needed (0-100)
        """
        if transferability_level == TransferabilityLevel.HIGH:
            # High transferability: minimal or no fine-tuning
            return int(max(0, (self.high_threshold - composite_score) * 100))
        
        elif transferability_level == TransferabilityLevel.MODERATE:
            # Moderate: scale linearly between 10% and 50%
            # Higher score within moderate range = less data needed
            range_size = self.high_threshold - self.moderate_threshold
            position = (composite_score - self.moderate_threshold) / range_size
            data_pct = int(50 - position * 40)  # 50% to 10%
            return data_pct
        
        elif transferability_level == TransferabilityLevel.LOW:
            # Low: need substantial data (50-80%)
            range_size = self.moderate_threshold - self.low_threshold
            position = (composite_score - self.low_threshold) / range_size
            data_pct = int(80 - position * 30)  # 80% to 50%
            return data_pct
        
        else:  # VERY_LOW
            # Very low: just train from scratch with 100% data
            return 100
    
    def assess_risks(self, 
                    transferability_level: TransferabilityLevel,
                    metrics: Dict[str, float]) -> list:
        """
        Identify potential risks in the transfer process
        
        Parameters:
        -----------
        transferability_level : TransferabilityLevel
            Classification level
        metrics : dict
            Individual metric values
            
        Returns:
        --------
        risks : list
            List of identified risk factors
        """
        risks = []
        
        # Check individual metrics for specific warnings
        if metrics['mmd'] > 0.5:
            risks.append("High distribution mismatch - source and target are very different")
        
        if metrics['correlation_stability'] < 0.90:
            risks.append("Feature relationships differ between domains - learned patterns may not transfer well")
        
        if metrics['js_divergence'] > 0.3:
            risks.append("Significant divergence in feature distributions")
        
        if metrics['n_target_samples'] < 200:
            risks.append("Small target sample size - results may not be representative")
        
        if metrics['n_source_samples'] < 500:
            risks.append("Limited source data - model may not have learned robust patterns")
        
        # Size mismatch warning
        size_ratio = metrics['n_target_samples'] / metrics['n_source_samples']
        if size_ratio < 0.3 or size_ratio > 3.0:
            risks.append(f"Large domain size difference (ratio: {size_ratio:.2f}) - may affect transfer")
        
        # General risk based on transferability level
        if transferability_level == TransferabilityLevel.LOW:
            risks.append("Low transferability - expect significant performance degradation without fine-tuning")
        elif transferability_level == TransferabilityLevel.VERY_LOW:
            risks.append("Very low transferability - transfer learning may not be beneficial")
        
        if not risks:
            risks.append("No major risks identified - transfer looks promising")
        
        return risks
    
    def recommend_strategy(self, 
                          composite_score: float,
                          metrics: Dict[str, float]) -> TransferRecommendation:
        """
        Generate complete transfer learning recommendation
        
        Parameters:
        -----------
        composite_score : float
            Composite transferability score
        metrics : dict
            Individual metric values
            
        Returns:
        --------
        recommendation : TransferRecommendation
            Complete recommendation with strategy, reasoning, and risks
        """
        # Classify transferability level
        level = self.classify_transferability(composite_score)
        
        # Calculate confidence
        confidence = self.calculate_confidence(metrics)
        
        # Estimate data requirements
        data_pct = self.estimate_data_requirement(level, composite_score)
        
        # Assess risks
        risks = self.assess_risks(level, metrics)
        
        # Determine strategy
        if level == TransferabilityLevel.HIGH:
            strategy = TransferStrategy.TRANSFER_AS_IS
            reasoning = (
                f"Excellent transferability (score: {composite_score:.4f}). "
                f"Source and target domains are highly similar across all metrics. "
                f"The model learned on source domain should perform well on target "
                f"with minimal or no adaptation."
            )
            expected_perf = (
                "Expected to maintain 85-95% of source domain performance on target domain. "
                "May benefit from minimal fine-tuning (0-10% target data) for slight improvements."
            )
        
        elif level == TransferabilityLevel.MODERATE:
            # Choose fine-tuning intensity based on score
            if composite_score >= 0.78:
                strategy = TransferStrategy.FINE_TUNE_LIGHT
                ft_range = "10-20%"
            elif composite_score >= 0.75:
                strategy = TransferStrategy.FINE_TUNE_MODERATE
                ft_range = "20-40%"
            else:
                strategy = TransferStrategy.FINE_TUNE_HEAVY
                ft_range = "40-60%"
            
            reasoning = (
                f"Moderate transferability (score: {composite_score:.4f}). "
                f"Domains share some similarities but have notable differences. "
                f"Transfer learning is viable but requires fine-tuning with {ft_range} "
                f"of target data to adapt to domain-specific patterns."
            )
            expected_perf = (
                f"With {data_pct}% target data for fine-tuning, expect 70-85% of "
                f"source domain performance. Fine-tuning allows model to adapt to "
                f"target-specific patterns while leveraging source knowledge."
            )
        
        elif level == TransferabilityLevel.LOW:
            strategy = TransferStrategy.FINE_TUNE_HEAVY
            reasoning = (
                f"Low transferability (score: {composite_score:.4f}). "
                f"Significant differences between domains. Transfer learning may still "
                f"provide benefit over random initialization, but requires substantial "
                f"fine-tuning (60-80% target data). Consider comparing with training from scratch."
            )
            expected_perf = (
                f"Requires {data_pct}% target data. Expect marginal benefit over training "
                f"from scratch (10-20% improvement). May be more efficient to train "
                f"a new model if sufficient target data is available."
            )
        
        else:  # VERY_LOW
            strategy = TransferStrategy.TRAIN_FROM_SCRATCH
            reasoning = (
                f"Very low transferability (score: {composite_score:.4f}). "
                f"Domains are fundamentally different. Transfer learning is not recommended. "
                f"The source model's knowledge may be irrelevant or even harmful for "
                f"the target domain. Train a new model from scratch using target data."
            )
            expected_perf = (
                "Transfer not recommended. Training from scratch on 100% target data "
                "will likely outperform transferred model. Source domain knowledge "
                "does not apply to this target domain."
            )
        
        return TransferRecommendation(
            strategy=strategy,
            transferability_level=level,
            composite_score=composite_score,
            confidence=confidence,
            target_data_percentage=data_pct,
            reasoning=reasoning,
            risks=risks,
            expected_performance=expected_perf
        )
    
    def compare_strategies(self, 
                          composite_score: float,
                          metrics: Dict[str, float]) -> Dict[str, dict]:
        """
        Compare multiple strategy options with expected outcomes
        
        Useful for decision-making when transferability is borderline
        
        Parameters:
        -----------
        composite_score : float
            Composite transferability score
        metrics : dict
            Individual metric values
            
        Returns:
        --------
        comparison : dict
            Comparison of different strategies with pros/cons
        """
        level = self.classify_transferability(composite_score)
        
        comparison = {
            'transfer_as_is': {
                'feasible': level == TransferabilityLevel.HIGH,
                'effort': 'Minimal',
                'data_needed': '0-10%',
                'time_to_deploy': 'Immediate',
                'expected_accuracy': 'High (if transferability is truly high)',
                'pros': ['Fastest deployment', 'No target data labeling needed', 'Lowest cost'],
                'cons': ['Risk of poor performance if domains differ', 'No adaptation to target specifics']
            },
            'fine_tune': {
                'feasible': level in [TransferabilityLevel.HIGH, TransferabilityLevel.MODERATE],
                'effort': 'Moderate',
                'data_needed': f'{self.estimate_data_requirement(level, composite_score)}%',
                'time_to_deploy': '1-2 weeks',
                'expected_accuracy': 'Good to excellent (with appropriate data)',
                'pros': ['Balances transfer benefits with adaptation', 'Usually best ROI', 'Proven approach'],
                'cons': ['Requires labeled target data', 'Need to tune hyperparameters', 'Risk of overfitting']
            },
            'train_from_scratch': {
                'feasible': True,  # Always possible if you have data
                'effort': 'High',
                'data_needed': '100%',
                'time_to_deploy': '2-4 weeks',
                'expected_accuracy': 'Best possible (given sufficient data)',
                'pros': ['No negative transfer risk', 'Optimized for target domain', 'Clean slate'],
                'cons': ['Requires most labeled data', 'Longest development time', 'Highest cost']
            }
        }
        
        return comparison


if __name__ == "__main__":
    print("Transfer Learning Decision Engine")
    print("="*70)
    print("\nThis module provides intelligent decision-making for transfer learning strategies.")
    print("\nExample usage:")
    print("  from decision_engine import DecisionEngine")
    print("  engine = DecisionEngine()")
    print("  recommendation = engine.recommend_strategy(composite_score, metrics)")
    print("  print(recommendation)")
