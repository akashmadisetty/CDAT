"""
Debug script to reproduce the exact issue from run_all_experiments.py
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load data exactly as experiment does
target_rfm = pd.read_csv('src/week2/domain_pair7_target_RFM.csv')
target_data = target_rfm[['Recency', 'Frequency', 'Monetary']].values

print("="*80)
print("REPRODUCING EXACT EXPERIMENT SCENARIO")
print("="*80)
print(f"Total target samples: {len(target_data)}\n")

# Simulate fine-tune with 10%
percentage = 10
n_samples = int(len(target_data) * percentage / 100)
print(f"10% fine-tune: {n_samples} training samples\n")

# NO RANDOM SEED SET (this is the issue!)
indices = np.random.choice(len(target_data), n_samples, replace=False)

train_indices = indices
test_mask = np.ones(len(target_data), dtype=bool)
test_mask[train_indices] = False
test_indices = np.where(test_mask)[0]

X_train = target_data[train_indices]
X_test = target_data[test_indices]

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# Train model
model = KMeans(n_clusters=3, random_state=42, max_iter=300, n_init=10)
model.fit(X_train)

# Predict on test
test_labels = model.predict(X_test)
unique_test = len(np.unique(test_labels))

print(f"\nTest predictions: {unique_test} unique clusters")
print(f"Cluster distribution: {np.bincount(test_labels)}")

if unique_test >= 2:
    sil = silhouette_score(X_test, test_labels)
    print(f"Silhouette: {sil:.3f} ✅")
else:
    print("Only 1 cluster! ❌")
    
print("\n" + "="*80)
print("CHECKING DATA CHARACTERISTICS")
print("="*80)

print(f"\nX_train statistics:")
print(f"  Recency:  mean={X_train[:, 0].mean():.2f}, std={X_train[:, 0].std():.2f}")
print(f"  Frequency: mean={X_train[:, 1].mean():.2f}, std={X_train[:, 1].std():.2f}")
print(f"  Monetary:  mean={X_train[:, 2].mean():.2f}, std={X_train[:, 2].std():.2f}")

print(f"\nX_test statistics:")
print(f"  Recency:  mean={X_test[:, 0].mean():.2f}, std={X_test[:, 0].std():.2f}")
print(f"  Frequency: mean={X_test[:, 1].mean():.2f}, std={X_test[:, 1].std():.2f}")
print(f"  Monetary:  mean={X_test[:, 2].mean():.2f}, std={X_test[:, 2].std():.2f}")

print(f"\nCluster centers (from training):")
for i, center in enumerate(model.cluster_centers_):
    print(f"  Cluster {i}: R={center[0]:.2f}, F={center[1]:.2f}, M={center[2]:.2f}")

print(f"\nTest data ranges:")
print(f"  Recency:  [{X_test[:, 0].min():.2f}, {X_test[:, 0].max():.2f}]")
print(f"  Frequency: [{X_test[:, 1].min():.2f}, {X_test[:, 1].max():.2f}]")
print(f"  Monetary:  [{X_test[:, 2].min():.2f}, {X_test[:, 2].max():.2f}]")

# Check if centers fall outside test range
print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

outliers = []
for i, center in enumerate(model.cluster_centers_):
    for j, (feat_name, feat_val) in enumerate([('Recency', center[0]), 
                                                 ('Frequency', center[1]), 
                                                 ('Monetary', center[2])]):
        test_min = X_test[:, j].min()
        test_max = X_test[:, j].max()
        
        if feat_val < test_min or feat_val > test_max:
            outliers.append(f"Cluster {i} {feat_name}: {feat_val:.2f} outside [{test_min:.2f}, {test_max:.2f}]")

if outliers:
    print("\n⚠️ PROBLEM FOUND:")
    for out in outliers:
        print(f"  {out}")
    print("\n→ Cluster centers from small training sample don't match test data!")
    print("→ All test points assigned to nearest (same) cluster")
else:
    print("\n✓ No obvious range issues")
    print("→ Problem must be elsewhere")
