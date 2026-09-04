# Decision Tree Classifier on Iris Dataset
# Clean, well-commented, and slightly improved version

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Load dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print(f"Dataset shape: {X.shape}")
print(f"Classes: {target_names}")

# 2. Train-Test Split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Feature Scaling (optional for trees, included for completeness)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Train Decision Tree
clf = DecisionTreeClassifier(
    criterion='gini',
    max_depth=None,          # grow fully (you can limit it)
    random_state=42
)
clf.fit(X_train_scaled, y_train)
print("\nDecision Tree trained successfully!")

# 5. Predictions & Evaluation
y_pred = clf.predict(X_test_scaled)

print(f"\nAccuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=target_names))

# 6. Visualize the Tree
plt.figure(figsize=(16, 10))
plot_tree(
    clf,
    feature_names=feature_names,
    class_names=target_names,
    filled=True,
    rounded=True,
    fontsize=11
)
plt.title("Decision Tree – Iris Dataset (Gini Criterion)", fontsize=16)
plt.tight_layout()
plt.show()