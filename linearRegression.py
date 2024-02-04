import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Load the California Housing dataset
data = fetch_california_housing()
X, y = data.data, data.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# create a linear regression modal
model = LinearRegression()

# Tarain the model
model.fit(X_train, y_train)

# Make a prediction on the test set
pred = model.predict(X_test)

# Evaluate the modelk
mse = mean_squared_error(y_test, pred)
print(f"Mean Squared Error: {mse}")

# Visualize the prediction vs Actual values
plt.scatter(y_test, pred)
plt.xlabel('Actual Values')
plt.ylabel('Predicted values')
plt.title('Linear Regression: Actual vs Predicted Values')
plt.show()
