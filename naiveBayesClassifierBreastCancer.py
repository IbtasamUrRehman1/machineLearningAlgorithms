import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Laod the breast cancer dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Gaussian Naives Bayes cLASSIFIER
gnb = GaussianNB()

# Train the classifier
gnb.fit(X_train, y_train)

# Predict on the training data
y_train_pred = gnb.predict(X_train)

# Calculate the training accuracy
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"Training Accuracy: {train_accuracy}")

# Predict on the test data
y_test_pred = gnb.predict(X_test)

# Calculate testing accuracy
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Testing Accuracy: {test_accuracy}")

# Generate the classification report for the testing data
print(f"Classification Report (Testing Data): ")
print(classification_report(y_test, y_test_pred))

# Generate the confusion matrix for the testing data
conf_matix = confusion_matrix(y_test, y_test_pred)
print("Confusion matrix (Testing Data): ")
print(conf_matix)

# Visualize confusion matrix for testing data
plt.figure(figsize=(8, 6))
plt.imshow(conf_matix, cmap=plt.cm.Blues)
plt.colorbar()
plt.title('Confusion matrix (Testing Data)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.xticks(ticks=[0, 1], labels=['Benign', 'Malignant'])
plt.yticks(ticks=[0, 1], labels=['Benign', 'Malignant'])
for i in range(conf_matix.shape[0]):
    for j in range(conf_matix.shape[1]):
        plt.text(j,i,conf_matix[i, j], ha='center', va='center', color='white' )
plt.show()





