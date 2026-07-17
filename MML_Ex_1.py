# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 09:54:38 2023

@author: chris
"""
# 1. Split the dataset into training and testing sets.
# 2. Train the following classifiers:
#   - A linear SVM.
#   - A polynomial kernel SVM (degree = 3).
#   - An RBF kernel SVM.
# 3. Plot the decision boundary of each classifier.
# 4. Compare the classifiers by discussing:
#   - Which classifier performs best?
# Why does the linear SVM struggle?
# How do kernel methods overcome this limitation?

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

# Set random seed for reproducibility
np.random.seed(42)

# Generate two circles not touching each other
X, y = make_circles(n_samples=1000, factor=0.5, noise=0.05, random_state=42)

# Augment the original features with (x₁² + x₂²)
X_augmented = np.c_[X, np.sum(X**2, axis=1)]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(X_augmented, y, test_size=0.2, random_state=42)

# Create a linear kernel SVM
linear_svm_classifier = SVC(kernel='linear', C=1.0, random_state=42)
linear_svm_classifier_augmented = SVC(kernel='linear', C=1.0, random_state=42)

# Create a kernelized SVM with an RBF kernel
svm_classifier = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)

# Create a kernelized SVM with a Polynomial kernel
poly_svm_classifier = SVC(kernel='poly', degree=3, C=1.0, random_state=42)

# Train the SVM
svm_classifier.fit(X_train, y_train)
linear_svm_classifier.fit(X_train, y_train)
poly_svm_classifier.fit(X_train, y_train)
linear_svm_classifier_augmented.fit(X_train_a, y_train_a)

# Make predictions on the test set
predictions = svm_classifier.predict(X_test)


# Plot the decision boundary
def plot_decision_boundary(model, X, y):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')
    
def plot_decision_boundary_augmented(model, X, y):
    h = .02  # step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # Augment the meshgrid data with (x₁² + x₂²)
    meshgrid_augmented = np.c_[xx.ravel(), yy.ravel(), np.sum(np.c_[xx.ravel(), yy.ravel()]**2, axis=1)]
    Z = model.predict(meshgrid_augmented)
    
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors='k')

# Plot decision boundary and data points
plt.figure(figsize=(15, 6))
plt.subplot(1,3,1)
plot_decision_boundary(linear_svm_classifier, X_test, y_test)
plt.title('SVM linear Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.subplot(1,3,2)
plot_decision_boundary(poly_svm_classifier, X_test, y_test)
plt.title('SVM polynomial Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.subplot(1,3,3)
plot_decision_boundary(svm_classifier, X_test, y_test)
plt.title('SVM Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')

plt.tight_layout()
plt.show()

plt.figure(figsize=(6,4))
plot_decision_boundary_augmented(linear_svm_classifier_augmented, X_test, y_test)
plt.title('Linear SVM Decision Boundary on Augmented Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()


