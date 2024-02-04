import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Create some random data for decomposition
np.random.seed(42)
data = np.random.rand(5, 3)  # 5 sample, 3 features

# Apply PCA with only 2 components
pca = PCA(n_components=2)
transformed_data = pca.fit_transform(data)

# Visualize the trasnformed data
plt.scatter(transformed_data[:, 0], transformed_data[:, 1])
plt.title("PCA Transformed Data")
plt.xlabel('Principle Component 1')
plt.ylabel('Principle Component 2')
plt.show()

# Print the percentage of variance explained by each components in terminal
print("Explained variance ratio: ", pca.explained_variance_ratio_)
