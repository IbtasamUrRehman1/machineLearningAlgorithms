import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt

# load irirs
iris = load_iris()
X = iris.data
y = iris.target

# Standardize the linkage matrix
scalar = StandardScaler()
X_scaled = scalar.fit_transform(X)

# Calculate the linakge Matrix
Z = linkage(X_scaled, method='ward') # Using ward's method for hierarchical clustring

# plot dendogram
plt.figure(figsize=(10, 6))
dendrogram(Z, no_labels=True)
plt.title('Hierarchical Clustring Dendogram')
plt.xlabel('Data Points')
plt.ylabel('Distance')
plt.show()

# Perform Hierarchical clustring
n_clusters = 3
clustering = AgglomerativeClustering(n_clusters=n_clusters)
clustering.fit(X_scaled)

# Print clustring results
print("Clustering Results :")
for i in range(n_clusters):
    cluster_indices = np.where(clustering.labels_ == i)[0]
    print(f"Cluster { i + 1}: {cluster_indices}")









