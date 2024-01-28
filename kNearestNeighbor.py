# import all required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load iris datet
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create knn classifier with k=3
knn = KNeighborsClassifier(n_neighbors=3)

# Fit  the classifies on the training data
knn.fit(X_train, y_train)

# Make predication on the test data
y_pred = knn.predict(X_test)

# evaluate the performanace
accuracy = accuracy_score(y_test, y_pred)

# Display classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# confusion matrix
cm = confusion_matrix(y_pred, y_test)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted Label')
plt.ylabel(' True Label')
plt.title(' Confusion Matrix')
plt.show()


# pair plot
iris_df = pd.DataFrame(data=np.c_[iris['data'],iris['target']], columns=iris['feature_names'] + ['target'])
sns.pairplot(iris_df, hue='target', palette='viridis')
plt.suptitle('Pair Ploat of Iris Datasets', y=1.02)
plt.show()
