"""
MMD Validation Script
Tests Maximum Mean Discrepancy implementation against known cases
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Import your MMD function from master script
# Or copy it here

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
# TEST 1: Identical Distributions (MMD should be ~0)
# ============================================================================

def test_identical_distributions():
    """When source = target, MMD should be nearly 0"""
    print("TEST 1: Identical Distributions")
    print("-" * 50)
    
    np.random.seed(42)
    X_source = np.random.normal(0, 1, (100, 2))
    X_target = np.random.normal(0, 1, (100, 2))
    
    mmd = compute_mmd(X_source, X_target)
    
    print(f"MMD value: {mmd:.6f}")
    print(f"Expected: ~0.0 (close to zero)")
    
    # Test passes if MMD < 0.1 (reasonable threshold)
    passed = mmd < 0.1
    print(f"Result: {'✅ PASS' if passed else '❌ FAIL'}")
    
    return passed


# ============================================================================
# TEST 2: Very Different Distributions (MMD should be large)
# ============================================================================

def test_different_distributions():
    """When distributions are different, MMD should be large"""
    print("\nTEST 2: Different Distributions")
    print("-" * 50)
    
    np.random.seed(42)
    X_source = np.random.normal(0, 1, (100, 2))      # Mean=0
    X_target = np.random.normal(10, 1, (100, 2))     # Mean=10
    
    mmd = compute_mmd(X_source, X_target)
    
    print(f"MMD value: {mmd:.6f}")
    print(f"Expected: Large (>1.0)")
    
    passed = mmd > 1.0
    print(f"Result: {'✅ PASS' if passed else '❌ FAIL'}")
    
    return passed


# ============================================================================
# TEST 3: Gradually Shifting Distributions (MMD should increase monotonically)
# ============================================================================

def test_monotonic_increase():
    """As distributions shift apart, MMD should increase"""
    print("\nTEST 3: Monotonic Increase Property")
    print("-" * 50)
    
    np.random.seed(42)
    X_source = np.random.normal(0, 1, (100, 2))
    
    shifts = [0, 0.5, 1, 2, 5]
    mmd_values = []
    
    for shift in shifts:
        X_target = np.random.normal(shift, 1, (100, 2))
        mmd = compute_mmd(X_source, X_target)
        mmd_values.append(mmd)
        print(f"Shift={shift:3.1f} → MMD={mmd:.4f}")
    
    # Check if MMD increases with shift
    is_increasing = all(mmd_values[i] <= mmd_values[i+1] 
                       for i in range(len(mmd_values)-1))
    
    passed = is_increasing
    print(f"\nResult: {'✅ PASS' if passed else '❌ FAIL'}")
    
    # Visualization
    plt.figure(figsize=(8, 5))
    plt.plot(shifts, mmd_values, marker='o', linewidth=2)
    plt.xlabel('Distribution Shift (mean difference)')
    plt.ylabel('MMD Value')
    plt.title('MMD should increase as distributions diverge')
    plt.grid(True, alpha=0.3)
    plt.savefig('mmd_validation_plot.png', dpi=150, bbox_inches='tight')
    print("\n✓ Saved: mmd_validation_plot.png")
    
    return passed


# ============================================================================
# TEST 4: Symmetry Property (MMD(A,B) = MMD(B,A))
# ============================================================================

def test_symmetry():
    """MMD should be symmetric"""
    print("\nTEST 4: Symmetry Property")
    print("-" * 50)
    
    np.random.seed(42)
    X_source = np.random.normal(0, 1, (100, 2))
    X_target = np.random.normal(2, 1, (100, 2))
    
    mmd_forward = compute_mmd(X_source, X_target)
    mmd_backward = compute_mmd(X_target, X_source)
    
    print(f"MMD(Source→Target): {mmd_forward:.6f}")
    print(f"MMD(Target→Source): {mmd_backward:.6f}")
    print(f"Difference: {abs(mmd_forward - mmd_backward):.6f}")
    
    # Should be nearly identical (allowing tiny numerical error)
    passed = abs(mmd_forward - mmd_backward) < 0.001
    print(f"Result: {'✅ PASS' if passed else '❌ FAIL'}")
    
    return passed


# ============================================================================
# TEST 5: Different Variances (should detect variance differences)
# ============================================================================

def test_variance_sensitivity():
    """MMD should detect variance differences"""
    print("\nTEST 5: Variance Sensitivity")
    print("-" * 50)
    
    np.random.seed(42)
    X_source = np.random.normal(0, 1, (100, 2))    # std=1
    X_target = np.random.normal(0, 3, (100, 2))    # std=3
    
    mmd = compute_mmd(X_source, X_target)
    
    print(f"Source: mean=0, std=1")
    print(f"Target: mean=0, std=3")
    print(f"MMD value: {mmd:.6f}")
    print(f"Expected: Moderate (0.2-0.8)")
    
    passed = 0.2 < mmd < 0.8
    print(f"Result: {'✅ PASS' if passed else '❌ FAIL'}")
    
    return passed


# ============================================================================
# RUN ALL TESTS
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("MMD VALIDATION TEST SUITE")
    print("="*60)
    
    tests = [
        test_identical_distributions,
        test_different_distributions,
        test_monotonic_increase,
        test_symmetry,
        test_variance_sensitivity
    ]
    
    results = []
    for test in tests:
        try:
            passed = test()
            results.append(passed)
        except Exception as e:
            print(f"❌ ERROR: {e}")
            results.append(False)
    
    # Final summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Tests passed: {sum(results)}/{len(results)}")
    print(f"Success rate: {sum(results)/len(results)*100:.1f}%")
    
    if all(results):
        print("\n✅ ALL TESTS PASSED - MMD implementation is correct!")
    else:
        print("\n⚠️  SOME TESTS FAILED - Review MMD implementation")