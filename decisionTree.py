# import all required libraries
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

iris = load_iris()
X, y = iris.data, iris.target

# split the dataset into traning and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

# create decision tree classifier
clf = DecisionTreeClassifier()

# Train the classifie on traning data
clf.fit(X_train, y_train)

# make a prediction on testing data
prediction = clf.predict(X_test)
print(prediction)

# Evaluate the accuracy on the modal
accuracy = accuracy_score(y_test, prediction)
print(f"accuracy: {accuracy}")

# disaply  additional metrics
print(classification_report(y_test, prediction))

# Visualization of decision tree

plt.figure(figsize=(12, 8))
plot_tree(clf, feature_names=iris.feature_names, class_names =iris.target_names, filled=True, rounded=True)
plt.show()
