# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset
url = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
titanic_data = pd.read_csv(url)

# Display the first few rows of the dataset to understand its structure
print(titanic_data.head())

# Select relevant features and target variable
features = ['Pclass', 'Sex', 'Age', 'Fare']
target = 'Survived'

# Preprocess the data (handle missing values, convert categorical variables, etc.)
titanic_data = titanic_data[features + [target]].dropna()
titanic_data['Sex'] = titanic_data['Sex'].map({'male': 0, 'female': 1})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(titanic_data[features], titanic_data[target], test_size=0.2, random_state=42)

# Initialize the Naive Bayes classifier
nb_classifier = GaussianNB()

# Train the classifier
nb_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = nb_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Visualize the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Survived', 'Survived'], yticklabels=['Not Survived', 'Survived'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
