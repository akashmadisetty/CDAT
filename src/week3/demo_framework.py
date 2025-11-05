"""
Transfer Learning Framework - Complete Demo
===========================================
Demonstrates end-to-end usage of the framework

This script shows how to:
1. Load source and target domain data
2. Calculate transferability metrics
3. Get strategy recommendations
4. Execute transfer learning
5. Evaluate results

Run this to see the complete framework in action!

Author: Member 3 (Research Lead)
Date: Week 3-4, 2024
"""

import os
import sys
import pandas as pd
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(__file__))

from framework import TransferLearningFramework, quick_transfer_assessment
from metrics import TransferabilityMetrics, quick_transferability_check
from decision_engine import DecisionEngine
from calibration import FrameworkCalibration
from validation import FrameworkValidator


def demo_basic_usage():
    """
    Demo 1: Basic framework usage
    """
    print("\n" + "="*80)
    print("DEMO 1: BASIC FRAMEWORK USAGE")
    print("="*80)
    
    print("\n📚 This demo shows the simplest way to use the framework")
    
    # Define paths (adjust to your Week 2 data location)
    source_rfm = '../week2/domain_pair1_source_RFM.csv'
    target_rfm = '../week2/domain_pair1_target_RFM.csv'
    
    # Check if files exist
    if not os.path.exists(source_rfm):
        print(f"\n⚠️  Demo data not found: {source_rfm}")
        print("   Please run Week 2 scripts first to generate RFM data")
        return
    
    print(f"\n✓ Using data:")
    print(f"  Source: {source_rfm}")
    print(f"  Target: {target_rfm}")
    
    # Method 1: Quick assessment (one-liner)
    print("\n" + "-"*80)
    print("Method 1: Quick Assessment")
    print("-"*80)
    
    recommendation = quick_transfer_assessment(
        source_rfm, 
        target_rfm, 
        pair_name="Cleaning & Household → Foodgrains"
    )
    
    print(f"\n✅ Quick result: {recommendation.transferability_level.value}")
    print(f"   Strategy: {recommendation.strategy.value}")
    print(f"   Confidence: {recommendation.confidence:.1f}%")
    
    # Method 2: Step-by-step (more control)
    print("\n" + "-"*80)
    print("Method 2: Step-by-Step Analysis")
    print("-"*80)
    
    # Initialize framework
    framework = TransferLearningFramework()
    
    # Load data
    print("\n1️⃣  Loading data...")
    framework.load_data(source_rfm, target_rfm)
    
    # Calculate transferability
    print("\n2️⃣  Calculating transferability...")
    results = framework.calculate_transferability(verbose=True)
    
    # Get recommendation
    print("\n3️⃣  Getting strategy recommendation...")
    recommendation = framework.recommend_strategy(verbose=True)
    
    # Save results
    print("\n4️⃣  Saving results...")
    framework.save_results(output_dir='demo_output', pair_name='demo_pair1')
    
    print("\n✅ Demo 1 complete!")


def demo_transfer_execution():
    """
    Demo 2: Execute transfer learning workflow
    """
    print("\n" + "="*80)
    print("DEMO 2: TRANSFER LEARNING EXECUTION")
    print("="*80)
    
    print("\n📚 This demo shows how to actually execute the transfer")
    
    source_rfm = '../week2/domain_pair1_source_RFM.csv'
    target_rfm = '../week2/domain_pair1_target_RFM.csv'
    source_model = '../week2/models/domain_pair1_rfm_kmeans_model.pkl'
    
    # Check if files exist
    if not all(os.path.exists(f) for f in [source_rfm, target_rfm, source_model]):
        print("\n⚠️  Demo data not found. Required files:")
        print(f"   - {source_rfm}")
        print(f"   - {target_rfm}")
        print(f"   - {source_model}")
        print("   Run Week 2 scripts first to generate these files")
        return
    
    # Initialize framework with model
    framework = TransferLearningFramework()
    
    # Load everything
    print("\n1️⃣  Loading data and model...")
    framework.load_source_model(source_model)
    framework.load_data(source_rfm, target_rfm)
    
    # Analyze transferability
    print("\n2️⃣  Analyzing transferability...")
    framework.calculate_transferability(verbose=False)
    recommendation = framework.recommend_strategy(verbose=True)
    
    # Execute transfer based on recommendation
    print("\n3️⃣  Executing transfer...")
    transferred_model = framework.execute_transfer()
    
    # Evaluate on target domain
    print("\n4️⃣  Evaluating transferred model...")
    score = framework.evaluate_transfer(transferred_model, metric='silhouette')
    
    print(f"\n✅ Transfer execution complete!")
    print(f"   Final model silhouette score on target: {score:.4f}")


def demo_compare_strategies():
    """
    Demo 3: Compare different transfer strategies
    """
    print("\n" + "="*80)
    print("DEMO 3: STRATEGY COMPARISON")
    print("="*80)
    
    print("\n📚 This demo compares Transfer as-is vs Fine-tune vs Train from scratch")
    
    source_rfm = '../week2/domain_pair1_source_RFM.csv'
    target_rfm = '../week2/domain_pair1_target_RFM.csv'
    source_model = '../week2/models/domain_pair1_rfm_kmeans_model.pkl'
    
    if not all(os.path.exists(f) for f in [source_rfm, target_rfm, source_model]):
        print("\n⚠️  Demo data not found")
        return
    
    # Setup
    framework = TransferLearningFramework()
    framework.load_source_model(source_model)
    framework.load_data(source_rfm, target_rfm)
    framework.calculate_transferability(verbose=False)
    
    # Get decision engine
    engine = DecisionEngine()
    
    # Compare all strategies
    print("\n📊 Comparing strategies...")
    comparison = engine.compare_strategies(
        framework.composite_score,
        framework.transferability_metrics
    )
    
    print("\n" + "-"*80)
    for strategy_name, info in comparison.items():
        print(f"\n{strategy_name.upper().replace('_', ' ')}")
        print(f"  Feasible: {info['feasible']}")
        print(f"  Effort: {info['effort']}")
        print(f"  Data needed: {info['data_needed']}")
        print(f"  Time to deploy: {info['time_to_deploy']}")
        print(f"  Expected accuracy: {info['expected_accuracy']}")
        print(f"  Pros: {', '.join(info['pros'][:2])}")
        print(f"  Cons: {', '.join(info['cons'][:2])}")
    
    print("\n✅ Strategy comparison complete!")


def demo_metrics_detail():
    """
    Demo 4: Deep dive into individual metrics
    """
    print("\n" + "="*80)
    print("DEMO 4: METRICS DEEP DIVE")
    print("="*80)
    
    print("\n📚 This demo explains each transferability metric in detail")
    
    source_rfm = '../week2/domain_pair1_source_RFM.csv'
    target_rfm = '../week2/domain_pair1_target_RFM.csv'
    
    if not os.path.exists(source_rfm) or not os.path.exists(target_rfm):
        print("\n⚠️  Demo data not found")
        return
    
    # Load data
    source_data = pd.read_csv(source_rfm)
    target_data = pd.read_csv(target_rfm)
    
    rfm_features = ['Recency', 'Frequency', 'Monetary']
    
    # Initialize metrics calculator
    calculator = TransferabilityMetrics()
    
    print("\n📊 Calculating individual metrics...")
    print("\nSource: ", source_rfm)
    print("Target: ", target_rfm)
    print(f"\nSource samples: {len(source_data):,}")
    print(f"Target samples: {len(target_data):,}")
    
    # Extract features
    X_source = source_data[rfm_features]
    X_target = target_data[rfm_features]
    
    # Calculate each metric separately with explanation
    print("\n" + "-"*80)
    print("1. Maximum Mean Discrepancy (MMD)")
    print("-"*80)
    mmd = calculator.calculate_mmd(
        calculator.scaler.fit_transform(X_source),
        calculator.scaler.transform(X_target)
    )
    print(f"Value: {mmd:.4f}")
    print("Interpretation: Measures distribution difference in kernel space")
    print("Range: [0, ∞), typically [0, 2]")
    print("Better: Lower values (more similar distributions)")
    if mmd < 0.1:
        print("✅ Excellent - distributions are very similar")
    elif mmd < 0.3:
        print("✓ Good - distributions are reasonably similar")
    else:
        print("⚠️  High difference - domains may be quite different")
    
    print("\n" + "-"*80)
    print("2. Jensen-Shannon Divergence")
    print("-"*80)
    js = calculator.calculate_js_divergence(
        calculator.scaler.fit_transform(X_source),
        calculator.scaler.transform(X_target)
    )
    print(f"Value: {js:.4f}")
    print("Interpretation: Information-theoretic distance between distributions")
    print("Range: [0, 1]")
    print("Better: Lower values")
    if js < 0.15:
        print("✅ Excellent - low information divergence")
    elif js < 0.3:
        print("✓ Good - moderate divergence")
    else:
        print("⚠️  High divergence")
    
    print("\n" + "-"*80)
    print("3. Correlation Stability")
    print("-"*80)
    corr_stab = calculator.calculate_correlation_stability(
        calculator.scaler.fit_transform(X_source),
        calculator.scaler.transform(X_target)
    )
    print(f"Value: {corr_stab:.4f}")
    print("Interpretation: How similar are feature relationships?")
    print("Range: [0, 1]")
    print("Better: Higher values (more stable correlations)")
    if corr_stab > 0.95:
        print("✅ Excellent - feature relationships are very stable")
    elif corr_stab > 0.85:
        print("✓ Good - reasonably stable")
    else:
        print("⚠️  Feature relationships differ significantly")
    
    # Calculate composite score
    print("\n" + "="*80)
    metrics = calculator.calculate_all_metrics(X_source, X_target)
    composite = calculator.compute_composite_score(metrics)
    
    print(f"COMPOSITE TRANSFERABILITY SCORE: {composite:.4f}")
    print("="*80)
    
    if composite >= 0.85:
        print("\n✅ HIGH transferability - Transfer as-is recommended")
    elif composite >= 0.70:
        print("\n✓ MODERATE transferability - Fine-tuning recommended")
    else:
        print("\n⚠️  LOW transferability - Consider training from scratch")
    
    print("\n✅ Metrics deep dive complete!")


def demo_full_workflow():
    """
    Demo 5: Complete end-to-end workflow on all 4 pairs
    """
    print("\n" + "="*80)
    print("DEMO 5: COMPLETE WORKFLOW - ALL 4 DOMAIN PAIRS")
    print("="*80)
    
    print("\n📚 This demo runs the complete framework on all domain pairs")
    
    # Define all 4 pairs
    pairs = [
        {
            'name': 'Pair 1: Cleaning & Household → Foodgrains',
            'source': '../week2/domain_pair1_source_RFM.csv',
            'target': '../week2/domain_pair1_target_RFM.csv'
        },
        {
            'name': 'Pair 2: Snacks → Fruits & Vegetables',
            'source': '../week2/domain_pair2_source_RFM.csv',
            'target': '../week2/domain_pair2_target_RFM.csv'
        },
        {
            'name': 'Pair 3: Premium → Budget',
            'source': '../week2/domain_pair3_source_RFM.csv',
            'target': '../week2/domain_pair3_target_RFM.csv'
        },
        {
            'name': 'Pair 4: Popular → Niche Brands',
            'source': '../week2/domain_pair4_source_RFM.csv',
            'target': '../week2/domain_pair4_target_RFM.csv'
        }
    ]
    
    results_summary = []
    
    for i, pair in enumerate(pairs, 1):
        print(f"\n{'='*80}")
        print(f"Processing {i}/4: {pair['name']}")
        print('='*80)
        
        if not os.path.exists(pair['source']) or not os.path.exists(pair['target']):
            print(f"⚠️  Data not found, skipping...")
            continue
        
        # Initialize framework
        framework = TransferLearningFramework()
        framework.load_data(pair['source'], pair['target'], validate=True)
        
        # Analyze
        results = framework.calculate_transferability(verbose=False)
        recommendation = framework.recommend_strategy(verbose=False)
        
        # Store results
        results_summary.append({
            'pair': pair['name'],
            'score': results['composite_score'],
            'level': recommendation.transferability_level.value,
            'strategy': recommendation.strategy.value,
            'confidence': recommendation.confidence,
            'data_needed': recommendation.target_data_percentage
        })
        
        print(f"   Score: {results['composite_score']:.4f}")
        print(f"   Level: {recommendation.transferability_level.value}")
        print(f"   Strategy: {recommendation.strategy.value}")
        print(f"   Data needed: {recommendation.target_data_percentage}%")
    
    # Summary table
    if len(results_summary) > 0:
        print("\n" + "="*80)
        print("SUMMARY: ALL DOMAIN PAIRS")
        print("="*80)
        
        df = pd.DataFrame(results_summary)
        print("\n", df.to_string(index=False))
        
        # Save to CSV
        df.to_csv('demo_output/all_pairs_summary.csv', index=False)
        print("\n✓ Saved summary to: demo_output/all_pairs_summary.csv")
    
    print("\n✅ Full workflow demo complete!")


def run_all_demos():
    """
    Run all demos in sequence
    """
    print("\n" + "="*80)
    print("TRANSFER LEARNING FRAMEWORK - COMPLETE DEMONSTRATION")
    print("="*80)
    print("\nThis script will demonstrate all capabilities of the framework")
    print("across 5 different scenarios.")
    
    # Create output directory
    os.makedirs('demo_output', exist_ok=True)
    
    try:
        demo_basic_usage()
        input("\n\nPress Enter to continue to Demo 2...")
        
        demo_transfer_execution()
        input("\n\nPress Enter to continue to Demo 3...")
        
        demo_compare_strategies()
        input("\n\nPress Enter to continue to Demo 4...")
        
        demo_metrics_detail()
        input("\n\nPress Enter to continue to Demo 5...")
        
        demo_full_workflow()
        
    except KeyboardInterrupt:
        print("\n\nDemos interrupted by user")
    except Exception as e:
        print(f"\n\n⚠️  Error during demos: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("ALL DEMOS COMPLETE!")
    print("="*80)
    print("\n📁 Check the 'demo_output' directory for saved results")
    print("\n✅ You now know how to use the Transfer Learning Framework!")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("TRANSFER LEARNING FRAMEWORK - DEMO SCRIPT")
    print("="*80)
    print("\nAvailable demos:")
    print("  1. Basic framework usage")
    print("  2. Transfer execution")
    print("  3. Strategy comparison")
    print("  4. Metrics deep dive")
    print("  5. Complete workflow (all 4 pairs)")
    print("  6. Run ALL demos")
    
    choice = input("\nEnter demo number (1-6) or press Enter for all: ").strip()
    
    if choice == '1':
        demo_basic_usage()
    elif choice == '2':
        demo_transfer_execution()
    elif choice == '3':
        demo_compare_strategies()
    elif choice == '4':
        demo_metrics_detail()
    elif choice == '5':
        demo_full_workflow()
    else:
        run_all_demos()
