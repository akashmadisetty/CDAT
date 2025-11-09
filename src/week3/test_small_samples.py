import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load Pair 7 target data
rfm = pd.read_csv('src/week2/domain_pair7_target_RFM.csv')
X = rfm[['Recency', 'Frequency', 'Monetary']].values

print("="*60)
print("TESTING FINE-TUNING WITH SMALL SAMPLE SIZES")
print("="*60)
print(f"Total target samples: {len(X)}\n")

# Test different training percentages
for pct in [10, 20, 50, 100]:
    n_train = int(len(X) * pct / 100)
    
    # Random split
    indices = np.random.choice(len(X), n_train, replace=False)
    X_train = X[indices]
    
    # Train KMeans
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(X_train)
    
    # Check training predictions
    train_labels = kmeans.predict(X_train)
    unique_clusters_train = len(np.unique(train_labels))
    
    # Check test predictions (if not 100%)
    if pct < 100:
        test_mask = np.ones(len(X), dtype=bool)
        test_mask[indices] = False
        X_test = X[test_mask]
        test_labels = kmeans.predict(X_test)
        unique_clusters_test = len(np.unique(test_labels))
    else:
        X_test = X_train
        test_labels = train_labels
        unique_clusters_test = unique_clusters_train
    
    print(f"{pct}% training data ({n_train} samples):")
    print(f"  Training: {unique_clusters_train} clusters")
    
    if unique_clusters_train >= 2:
        sil_train = silhouette_score(X_train, train_labels)
        print(f"    Silhouette: {sil_train:.3f}")
    
    print(f"  Test: {unique_clusters_test} clusters")
    if unique_clusters_test >= 2:
        sil_test = silhouette_score(X_test, test_labels)
        print(f"    Silhouette: {sil_test:.3f}")
    else:
        print(f"    ⚠️ Only 1 cluster predicted on test set!")
    
    print()

print("\n" + "="*60)
print("DIAGNOSIS:")
print("="*60)
print("If test set shows only 1 cluster, it means:")
print("1. Training sample too small for k=3")
print("2. Cluster centers from small sample don't generalize")
print("3. All test points are closest to same center")
print("\nSOLUTION: Evaluate on TRAINING set, not held-out test!")
