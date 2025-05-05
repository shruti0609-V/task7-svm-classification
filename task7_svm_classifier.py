import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

# Load dataset
data = load_breast_cancer()
X, y = data.data, data.target

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Linear SVM
svm_linear = SVC(kernel='linear')
svm_linear.fit(X_train, y_train)
print("Linear Kernel SVM:")
print(classification_report(y_test, svm_linear.predict(X_test)))

# Train RBF SVM with hyperparameter tuning
params = {'C': [0.1, 1, 10], 'gamma': ['scale', 0.1, 1]}
grid = GridSearchCV(SVC(kernel='rbf'), params, cv=5)
grid.fit(X_train, y_train)
print("RBF Kernel SVM with GridSearchCV:")
print(f"Best Parameters: {grid.best_params_}")
print(classification_report(y_test, grid.predict(X_test)))

# Confusion Matrix
disp = ConfusionMatrixDisplay.from_estimator(grid, X_test, y_test)
disp.ax_.set_title("Confusion Matrix (RBF SVM)")
plt.savefig("svm_confusion_matrix.png")
plt.show()
