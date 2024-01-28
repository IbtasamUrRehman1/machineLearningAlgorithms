import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Generate some example data
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = ( X > 5 ).astype(int).flatten()

# Split the data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Logistics Regression Modal
model = LogisticRegression()

# Train the modal on the training data
model.fit(X_train, y_train)

# Make a prediction on the testing data
y_pred = model.predict(X_test)

# Evaludate the modal
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("Accuracy :", accuracy)
print("Confusion  Matrix: \n", conf_matrix)


#  plot the decision boundry
plt.scatter(X_test, y_test, color='black', marker='.')
plt.scatter(X_test, y_pred, color='red', marker='x', linewidth=2)
plt.xlabel("Input Feature")
plt.ylabel("Class")
plt.title("Logistic Regression Decision Boundry")
plt.show()
