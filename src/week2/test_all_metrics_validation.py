"""
All Metrics Validation Script
Tests MMD, JS Divergence, Correlation Stability, KS, Wasserstein
"""

import numpy as np
from scipy.stats import wasserstein_distance, ks_2samp
from scipy.spatial.distance import jensenshannon

# ============================================================================
# METRIC IMPLEMENTATIONS
# ============================================================================

def compute_correlation_stability(X_source, X_target):
    """Measure correlation matrix similarity"""
    if X_source.shape[1] < 2:
        return 1.0  # Perfect stability if only 1 feature
    
    corr_s = np.corrcoef(X_source.T)
    corr_t = np.corrcoef(X_target.T)
    
    diff = corr_s - corr_t
    frobenius = np.sqrt(np.sum(diff ** 2))
    
    n_features = X_source.shape[1]
    max_dist = np.sqrt(2 * n_features ** 2)
    
    return 1 - (frobenius / max_dist)

def compute_ks_statistic(X_source, X_target):
    """Kolmogorov-Smirnov test statistic"""
    n_features = X_source.shape[1]
    ks_scores = []
    
    for i in range(n_features):
        ks_stat, _ = ks_2samp(X_source[:, i], X_target[:, i])
        ks_scores.append(ks_stat)
    
    return np.mean(ks_scores)

def compute_wasserstein(X_source, X_target):
    """Wasserstein distance (Earth Mover's Distance)"""
    n_features = X_source.shape[1]
    w_scores = []
    
    for i in range(n_features):
        w_dist = wasserstein_distance(X_source[:, i], X_target[:, i])
        w_scores.append(w_dist)
    
    return np.mean(w_scores)

def compute_js_divergence(X_source, X_target, n_bins=50):
    """Jensen-Shannon Divergence (symmetric KL)"""
    n_features = X_source.shape[1]
    js_scores = []
    
    for i in range(n_features):
        min_val = min(X_source[:, i].min(), X_target[:, i].min())
        max_val = max(X_source[:, i].max(), X_target[:, i].max())
        bins = np.linspace(min_val, max_val, n_bins)
        
        hist_s, _ = np.histogram(X_source[:, i], bins=bins, density=True)
        hist_t, _ = np.histogram(X_target[:, i], bins=bins, density=True)
        
        hist_s = hist_s / (hist_s.sum() + 1e-10)
        hist_t = hist_t / (hist_t.sum() + 1e-10)
        
        js = jensenshannon(hist_s, hist_t)
        js_scores.append(js)
    
    return np.mean(js_scores)

def compute_mmd(X_source, X_target, gamma=None):
    """
    Maximum Mean Discrepancy with automatic gamma estimation
    
    Args:
        X_source: Source domain data (n_source, n_features)
        X_target: Target domain data (n_target, n_features)
        gamma: RBF kernel bandwidth. If None, uses median heuristic
    
    Returns:
        mmd: MMD value (float)
    """
    
    # Auto-estimate gamma using median heuristic
    if gamma is None:
        # Combine source and target
        X_combined = np.vstack([X_source, X_target])
        
        # Calculate pairwise distances
        from scipy.spatial.distance import pdist
        pairwise_dists = pdist(X_combined, metric='euclidean')
        
        # Median heuristic: gamma = 1 / (2 * median^2)
        median_dist = np.median(pairwise_dists)
        gamma = 1.0 / (2 * median_dist ** 2)
        
        # Safeguard against extreme values
        gamma = np.clip(gamma, 0.001, 100)
    
    def rbf_kernel(X, Y, gamma):
        """RBF (Gaussian) kernel"""
        # Compute squared Euclidean distances
        XX = np.sum(X ** 2, axis=1)[:, None]
        YY = np.sum(Y ** 2, axis=1)[None, :]
        XY = np.dot(X, Y.T)
        
        sq_dists = XX + YY - 2 * XY
        
        # RBF kernel: K(x,y) = exp(-gamma * ||x-y||^2)
        K = np.exp(-gamma * sq_dists)
        return K
    
    n_source = X_source.shape[0]
    n_target = X_target.shape[0]
    
    # Compute kernel matrices
    K_ss = rbf_kernel(X_source, X_source, gamma)
    K_tt = rbf_kernel(X_target, X_target, gamma)
    K_st = rbf_kernel(X_source, X_target, gamma)
    
    # MMD^2 = E[K(x,x')] + E[K(y,y')] - 2*E[K(x,y)]
    mmd_squared = (
        (np.sum(K_ss) - np.trace(K_ss)) / (n_source * (n_source - 1)) +
        (np.sum(K_tt) - np.trace(K_tt)) / (n_target * (n_target - 1)) -
        2 * np.sum(K_st) / (n_source * n_target)
    )
    
    # Return MMD (not squared)
    return np.sqrt(max(mmd_squared, 0))


# ============================================================================
# UNIFIED TEST FRAMEWORK
# ============================================================================

class MetricValidator:
    """Framework for testing all metrics"""
    
    def __init__(self):
        self.results = {}
        # Metric-specific thresholds
        self.thresholds = {
            'MMD': {
                'near_zero': 0.15,
                'low': 0.15,
                'moderate': (0.15, 0.6),
                'large': 0.4,
                'high': 0.8
            },
            'JS Divergence': {
                'near_zero': 0.05,
                'low': 0.35,  # Accounts for sampling noise
                'moderate': (0.3, 0.7),
                'large': 0.6,
                'high': 0.8
            },
            'Correlation Stability': {
                'near_zero': 0.15,
                'low': 0.3,
                'moderate': (0.4, 0.8),
                'large': 0.6,
                'high': 0.75
            },
            'KS Statistic': {
                'near_zero': 0.15,
                'low': 0.15,
                'moderate': (0.15, 0.6),
                'large': 0.5,
                'high': 0.8
            },
            'Wasserstein Distance': {
                'near_zero': 0.25,  # Wasserstein has larger baseline
                'low': 0.25,
                'moderate': (0.25, 2.5),
                'large': 2.0,
                'high': 3.0
            }
        }
    
    def validate_value(self, metric_name, value, expected_behavior):
        """Validate value against metric-specific thresholds"""
        thresholds = self.thresholds.get(metric_name, self.thresholds['MMD'])
        
        if expected_behavior in ["near_zero", "low"]:
            return value < thresholds[expected_behavior]
        elif expected_behavior == "large":
            return value > thresholds['large']
        elif expected_behavior == "moderate":
            low, high = thresholds['moderate']
            return low < value < high
        elif expected_behavior == "high":
            return value > thresholds['high']
        else:
            print(f"⚠️  Unknown expected behavior: {expected_behavior}")
            return True
    
    def test_metric(self, metric_name, metric_func, all_test_cases):
        """Run test cases for a specific metric"""
        print(f"\n{'='*60}")
        print(f"TESTING: {metric_name}")
        print(f"{'='*60}")
        
        # Get test cases for this specific metric
        test_cases = all_test_cases.get(metric_name, [])
        
        if not test_cases:
            print(f"⚠️  No test cases defined for {metric_name}")
            return False
        
        passed = []
        
        for i, (name, X_source, X_target, expected_behavior) in enumerate(test_cases, 1):
            print(f"\nTest {i}: {name}")
            print("-" * 40)
            
            try:
                value = metric_func(X_source, X_target)
                print(f"Value: {value:.4f}")
                print(f"Expected: {expected_behavior}")
                
                # Use metric-specific validation
                test_passed = self.validate_value(metric_name, value, expected_behavior)
                
                status = '✅ PASS' if test_passed else '❌ FAIL'
                print(f"Result: {status}")
                passed.append(test_passed)
                
            except Exception as e:
                print(f"❌ ERROR: {e}")
                import traceback
                traceback.print_exc()
                passed.append(False)
        
        self.results[metric_name] = {
            'total': len(test_cases),
            'passed': sum(passed),
            'success_rate': sum(passed) / len(test_cases) * 100
        }
        
        return all(passed)


# ============================================================================
# TEST CASE DEFINITIONS
# ============================================================================

def generate_test_cases():
    """Create standard test cases for all metrics"""
    np.random.seed(42)
    
    # Case 1: Identical distributions (from same sampling)
    X_identical_1 = np.random.normal(0, 1, (200, 3))
    X_identical_2 = np.random.normal(0, 1, (200, 3))
    
    # Case 2: Different means (significant shift)
    X_mean_0 = np.random.normal(0, 1, (200, 3))
    X_mean_5 = np.random.normal(5, 1, (200, 3))
    
    # Case 3: Different variances (spread change)
    X_var_1 = np.random.normal(0, 1, (200, 3))
    X_var_3 = np.random.normal(0, 3, (200, 3))
    
    # Case 4: Different distributions (normal vs uniform)
    X_normal = np.random.normal(0, 1, (200, 3))
    X_uniform = np.random.uniform(-2, 2, (200, 3))
    
    # Case 5: Correlated data (for correlation stability)
    X_corr_1 = np.random.normal(0, 1, (200, 3))
    X_corr_1[:, 1] = X_corr_1[:, 0] * 0.8 + np.random.normal(0, 0.2, 200)
    
    X_corr_2 = np.random.normal(0, 1, (200, 3))
    X_corr_2[:, 1] = X_corr_2[:, 0] * 0.8 + np.random.normal(0, 0.2, 200)
    
    # Case 6: More extreme distributions
    X_exp = np.random.exponential(1, (200, 3))
    
    # Return dictionary with test cases for each metric
    return {
        'MMD': [
            ("Identical distributions", X_identical_1, X_identical_2, "near_zero"),
            ("Different means", X_mean_0, X_mean_5, "large"),
            ("Different variances", X_var_1, X_var_3, "moderate"),
            ("Similar shapes (Normal vs Uniform)", X_normal, X_uniform, "near_zero"),  # Similar mean/var
        ],
        'JS Divergence': [
            ("Identical distributions", X_identical_1, X_identical_2, "low"),  # Sampling noise expected
            ("Different means", X_mean_0, X_mean_5, "large"),
            ("Different variances", X_var_1, X_var_3, "moderate"),
            ("Different shapes", X_normal, X_uniform, "moderate"),
        ],
        'Correlation Stability': [
            ("Identical correlations", X_corr_1, X_corr_2, "high"),
            ("Same mean, diff variance (preserves corr)", X_var_1, X_var_3, "high"),
            ("Different means (preserves corr)", X_mean_0, X_mean_5, "high"),
            ("Uncorrelated data", X_identical_1, X_identical_2, "high"),
        ],
        'KS Statistic': [
            ("Identical distributions", X_identical_1, X_identical_2, "near_zero"),
            ("Different means", X_mean_0, X_mean_5, "large"),
            ("Different variances", X_var_1, X_var_3, "moderate"),
            ("Different shapes", X_normal, X_uniform, "moderate"),
        ],
        'Wasserstein Distance': [
            ("Identical distributions", X_identical_1, X_identical_2, "low"),  # Baseline ~0.15
            ("Different means", X_mean_0, X_mean_5, "large"),
            ("Different variances", X_var_1, X_var_3, "moderate"),  # ~1.8 is moderate for Wasserstein
            ("Different shapes", X_normal, X_uniform, "moderate"),
        ]
    }


# ============================================================================
# METRIC-SPECIFIC TESTS
# ============================================================================

def test_js_divergence_validation():
    """Specific tests for JS Divergence properties"""
    print("\n" + "🔵"*30)
    print("JS DIVERGENCE SPECIFIC VALIDATION")
    print("🔵"*30)
    
    np.random.seed(42)
    X1 = np.random.normal(0, 1, (100, 2))
    X2 = np.random.normal(10, 1, (100, 2))
    
    # Property 1: JS is bounded [0, 1]
    js = compute_js_divergence(X1, X2)
    
    print(f"\nProperty 1: JS ∈ [0, 1]")
    print(f"Computed JS: {js:.4f}")
    bounded = 0 <= js <= 1
    print(f"Result: {'✅ PASS' if bounded else '❌ FAIL'}")
    
    # Property 2: Symmetry
    js_forward = compute_js_divergence(X1, X2)
    js_backward = compute_js_divergence(X2, X1)
    symmetric = abs(js_forward - js_backward) < 0.01
    
    print(f"\nProperty 2: Symmetry")
    print(f"JS(A→B): {js_forward:.4f}")
    print(f"JS(B→A): {js_backward:.4f}")
    print(f"Difference: {abs(js_forward - js_backward):.6f}")
    print(f"Result: {'✅ PASS' if symmetric else '❌ FAIL'}")
    
    # Property 3: Self-comparison should be near zero
    js_self = compute_js_divergence(X1, X1)
    self_zero = js_self < 0.01
    
    print(f"\nProperty 3: JS(X, X) ≈ 0")
    print(f"JS(X, X): {js_self:.6f}")
    print(f"Result: {'✅ PASS' if self_zero else '❌ FAIL'}")
    
    return bounded and symmetric and self_zero


def test_correlation_stability_validation():
    """Specific tests for Correlation Stability properties"""
    print("\n" + "🟢"*30)
    print("CORRELATION STABILITY VALIDATION")
    print("🟢"*30)
    
    np.random.seed(42)
    n = 200
    
    # Test 1: Same correlation structure
    X1 = np.random.normal(0, 1, (n, 3))
    X1[:, 1] = X1[:, 0] * 0.9 + np.random.normal(0, 0.1, n)
    
    X2 = np.random.normal(0, 1, (n, 3))
    X2[:, 1] = X2[:, 0] * 0.9 + np.random.normal(0, 0.1, n)
    
    corr_stab = compute_correlation_stability(X1, X2)
    
    print(f"\nTest 1: Same correlation pattern")
    print(f"Correlation Stability: {corr_stab:.4f}")
    print(f"Expected: High (>0.8)")
    test1_passed = corr_stab > 0.8
    print(f"Result: {'✅ PASS' if test1_passed else '❌ FAIL'}")
    
    # Test 2: Different correlation structures
    X3 = np.random.normal(0, 1, (n, 3))
    X3[:, 1] = -X3[:, 0] * 0.9 + np.random.normal(0, 0.1, n)  # Negative correlation
    
    corr_stab2 = compute_correlation_stability(X1, X3)
    
    print(f"\nTest 2: Different correlation patterns")
    print(f"Correlation Stability: {corr_stab2:.4f}")
    print(f"Expected: Low (<0.7)")
    test2_passed = corr_stab2 < 0.7
    print(f"Result: {'✅ PASS' if test2_passed else '❌ FAIL'}")
    
    # Test 3: Self-comparison should be perfect
    corr_self = compute_correlation_stability(X1, X1)
    test3_passed = corr_self > 0.99
    
    print(f"\nTest 3: Self-comparison")
    print(f"Correlation Stability: {corr_self:.4f}")
    print(f"Expected: Near perfect (>0.99)")
    print(f"Result: {'✅ PASS' if test3_passed else '❌ FAIL'}")
    
    return test1_passed and test2_passed and test3_passed


def test_mmd_validation():
    """Specific tests for MMD properties"""
    print("\n" + "🟣"*30)
    print("MMD SPECIFIC VALIDATION")
    print("🟣"*30)
    
    np.random.seed(42)
    X1 = np.random.normal(0, 1, (100, 2))
    X2 = np.random.normal(5, 1, (100, 2))
    
    # Property 1: Non-negative
    mmd = compute_mmd(X1, X2)
    non_negative = mmd >= 0
    
    print(f"\nProperty 1: MMD ≥ 0")
    print(f"Computed MMD: {mmd:.4f}")
    print(f"Result: {'✅ PASS' if non_negative else '❌ FAIL'}")
    
    # Property 2: Self-comparison should be near zero
    mmd_self = compute_mmd(X1, X1)
    self_zero = mmd_self < 0.01
    
    print(f"\nProperty 2: MMD(X, X) ≈ 0")
    print(f"MMD(X, X): {mmd_self:.6f}")
    print(f"Result: {'✅ PASS' if self_zero else '❌ FAIL'}")
    
    return non_negative and self_zero


def test_wasserstein_validation():
    """Specific tests for Wasserstein Distance properties"""
    print("\n" + "🟠"*30)
    print("WASSERSTEIN DISTANCE VALIDATION")
    print("🟠"*30)
    
    np.random.seed(42)
    X1 = np.random.normal(0, 1, (100, 2))
    X2 = np.random.normal(5, 1, (100, 2))
    
    # Property 1: Non-negative
    w_dist = compute_wasserstein(X1, X2)
    non_negative = w_dist >= 0
    
    print(f"\nProperty 1: Wasserstein ≥ 0")
    print(f"Computed Wasserstein: {w_dist:.4f}")
    print(f"Result: {'✅ PASS' if non_negative else '❌ FAIL'}")
    
    # Property 2: Symmetry
    w_forward = compute_wasserstein(X1, X2)
    w_backward = compute_wasserstein(X2, X1)
    symmetric = abs(w_forward - w_backward) < 0.01
    
    print(f"\nProperty 2: Symmetry")
    print(f"W(A→B): {w_forward:.4f}")
    print(f"W(B→A): {w_backward:.4f}")
    print(f"Difference: {abs(w_forward - w_backward):.6f}")
    print(f"Result: {'✅ PASS' if symmetric else '❌ FAIL'}")
    
    # Property 3: Self-comparison should be near zero
    w_self = compute_wasserstein(X1, X1)
    self_zero = w_self < 0.01
    
    print(f"\nProperty 3: W(X, X) ≈ 0")
    print(f"W(X, X): {w_self:.6f}")
    print(f"Result: {'✅ PASS' if self_zero else '❌ FAIL'}")
    
    return non_negative and symmetric and self_zero


# ============================================================================
# MAIN VALIDATION RUNNER
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("COMPLETE METRICS VALIDATION SUITE")
    print("="*60)
    
    validator = MetricValidator()
    test_cases = generate_test_cases()
    
    # Test each metric
    metrics_to_test = [
        ("MMD", compute_mmd),
        ("JS Divergence", compute_js_divergence),
        ("Correlation Stability", compute_correlation_stability),
        ("KS Statistic", compute_ks_statistic),
        ("Wasserstein Distance", compute_wasserstein)
    ]
    
    all_passed = []
    for name, func in metrics_to_test:
        passed = validator.test_metric(name, func, test_cases)
        all_passed.append(passed)
    
    # Run specific property validations
    all_passed.append(test_mmd_validation())
    all_passed.append(test_js_divergence_validation())
    all_passed.append(test_correlation_stability_validation())
    all_passed.append(test_wasserstein_validation())
    
    # Final report
    print("\n" + "="*60)
    print("FINAL REPORT")
    print("="*60)
    
    for metric_name, results in validator.results.items():
        status = "✅" if results['success_rate'] == 100 else "⚠️"
        print(f"\n{status} {metric_name}:")
        print(f"  Tests Passed: {results['passed']}/{results['total']}")
        print(f"  Success Rate: {results['success_rate']:.1f}%")
    
    print("\n" + "="*60)
    if all(all_passed):
        print("🎉 ALL METRICS VALIDATED - Ready for production!")
    else:
        print("⚠️  SOME VALIDATIONS FAILED - Review implementations")
    print("="*60)