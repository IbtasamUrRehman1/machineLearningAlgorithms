import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn import metrics

# Load the lungs cancer dataset
data = pd.read_csv('dataset/lung_cancer.csv')

# Extract the feautres and labels
X = data.iloc[:,1:-1] # Assumig the features are in columns 1 throught second-to-last
y_true = data.iloc[:,-1]  # Assuming the labels are in the last colummn

# Remoe rows with non-numeric values
X_numeric = X.select_dtypes(include=[np.number])
print(X_numeric)

# Standardizr tyhe numeric data for the beter performance of K means
scalar = StandardScaler()
X_standardized = scalar.fit_transform(X_numeric)

# Number of clusters
k = 2 # Assuming there are the two types of lung cancer

# Create KMeans instances
kmeans = KMeans(n_clusters=k, random_state=42)

# Fit the modal to the standdarize data
kmeans.fit(X_standardized)

# Get the cluster labels
y_pred = kmeans.labels_

# Evaluate the clustring using Adjusted Rand Index
ari = metrics.adjusted_rand_score(y_true, y_pred)
print("Adjusted Rand Index : " ,ari)

# Visuliaze the results ( using only 2 features for simplicity)
plt.scatter(X_standardized[:,0], X_standardized[:,1], c=y_pred, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], marker='X', s=200, color='red'),
plt.title('K-Means Clustring on lung Cancer')
plt.xlabel('Feature 1 ( Standardized)')
plt.ylabel('Feature 2 ( Standardized)')
plt.show()















