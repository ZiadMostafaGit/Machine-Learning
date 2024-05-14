# ChatGPT
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Generate a synthetic dataset with 6 classes
X, y = make_classification(n_samples=1000, n_features=20, n_informative=15,
                           n_redundant=5, n_classes=6, random_state=42)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=50, random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# Predict the classes for the test set
y_pred = clf.predict(X_test)

# Calculate the accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.2f}")

# Generate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Visualize the confusion matrix
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=range(6), yticklabels=range(6))
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()


# Normalize the confusion matrix
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

# Visualize the normalized confusion matrix
sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=range(6), yticklabels=range(6))
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.title("Normalized Confusion Matrix (Percentages)")
plt.show()