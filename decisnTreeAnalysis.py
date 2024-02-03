import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt

# Load Iris dataset
iris = load_iris()
X, y = iris.data[:, :2], iris.target

# Combine features and labels into one dataset
dataset = np.column_stack((X, y))

def calculate_entropy(labels):
    unique_labels, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))  # Added epsilon to avoid log(0)
    return entropy

def calculate_gini_index(labels):
    unique_labels, counts = np.unique(labels, return_counts=True)
    probabilities = counts / len(labels)
    gini_index = 1 - np.sum(probabilities**2)
    return gini_index

def calculate_information_gain(dataset, feature_index, threshold):
    total_entropy = calculate_entropy(dataset[:, -1])

    # Split the dataset based on the given feature and threshold
    left_subset = dataset[dataset[:, feature_index] <= threshold]
    right_subset = dataset[dataset[:, feature_index] > threshold]

    # Calculate weighted average of the entropy for the subsets
    left_entropy = calculate_entropy(left_subset[:, -1])
    right_entropy = calculate_entropy(right_subset[:, -1])
    subset_entropy = (len(left_subset) / len(dataset)) * left_entropy + (len(right_subset) / len(dataset)) * right_entropy

    # Calculate information gain
    information_gain = total_entropy - subset_entropy
    return information_gain

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Combine features and labels into one training dataset
train_dataset = np.column_stack((X_train, y_train))

# Test the functions with a specific feature and threshold
feature_index = 0
threshold = 5.0  # Adjust threshold based on the feature distribution

# Calculate Entropy, Gini Index, and Information Gain
entropy = calculate_entropy(train_dataset[:, -1])
gini_index = calculate_gini_index(train_dataset[:, -1])
info_gain = calculate_information_gain(train_dataset, feature_index, threshold)

print(f"Entropy: {entropy}")
print(f"Gini Index: {gini_index}")
print(f"Information Gain for feature {feature_index} at threshold {threshold}: {info_gain}")

# Train a decision tree classifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Display decision tree text representation
tree_text = export_text(clf, feature_names=iris.feature_names[:2])
print("Decision Tree:")
print(tree_text)

# Plot the decision tree
plt.figure(figsize=(12, 8))
tree_plot = plot_tree(clf, filled=True, feature_names=iris.feature_names[:2], class_names=iris.target_names, fontsize=8)
plt.show()
