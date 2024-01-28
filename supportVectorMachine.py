# import all required libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_decision_regions

# import dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# User only the first two features for visualization
X_visualization = X[:, :2]

# Split the data into training and testing sets
X_train, X_test, y_train, y_Test = train_test_split(X_visualization, y, test_size=0.2, random_state=42)

# Standardize the featuresx
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create an SVM Model
svm_model = SVC(kernel='linear', C=1.0)

# Train the SVM Model
svm_model.fit(X_train, y_train)

# Make a prediction on the test det
y_pred = svm_model.predict(X_test)

# Evaluate the model's accuracy
accuracy = accuracy_score(y_Test, y_pred)
print("Accuracy : ", accuracy)

# Visualize decision regions
plt.figure(figsize=(10, 6))
plot_decision_regions(X_train, y_train, clf=svm_model, legend=2)

# Add axis labels and a title
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('SVM Decision Regions')

# Add a legend
plt.legend("Upper Left", "Lower Right")

# Show the plot
plt.show()
