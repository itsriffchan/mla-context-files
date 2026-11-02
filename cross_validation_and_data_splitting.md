# Cross-Validation and Data Splitting Coding Concepts

This document contains exact code snippets extracted from the provided notebooks, categorized by concept. These non-overlapping snippets represent the specific implementations and syntax used for dataset partitioning and validation.

Notebooks used in making this context file: `Hold_out_split1.ipynb`, `Hold_out_split2.ipynb`, `K-Fold_cv.ipynb`, `LOOCV.ipynb`, `Random_Sampling.ipynb`, `Stratified_CV.ipynb`

## 1. Dataset Loading and Preparation

**Loading the Iris Dataset**
```python
from sklearn import datasets

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
```

## 2. Hold-Out Data Splitting

**Simple Hold-Out Train-Test Split (80% / 20%)**
```python
from sklearn.model_selection import train_test_split

# Split the data into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Output the sizes of the splits to verify
print(f'Training set size: {X_train.shape[0]} samples')
print(f'Test set size: {X_test.shape[0]} samples')
```

**Training, Validation, and Test Split**
```python
from sklearn.model_selection import train_test_split

# First, split the data into training + validation and testing sets (80% - 20%)
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Then, split the training + validation set into training and validation sets 
# (80% - 20% of the remaining 80%)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

# Output the sizes of the splits to verify
print(f'Training set size: {X_train.shape[0]} samples')
print(f'Validation set size: {X_val.shape[0]} samples')
print(f'Test set size: {X_test.shape[0]} samples')
```

## 3. Random Sampling Partitioning

**Manual Random Permutation Split**
```python
import numpy as np

# Define the split ratio
test_ratio = 0.2
total_samples = X.shape[0]
test_size = int(total_samples * test_ratio)

# Generate random indices
np.random.seed(42)  # For reproducibility
indices = np.random.permutation(total_samples)

# Split the data
test_indices = indices[:test_size]
train_indices = indices[test_size:]

X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

# Output the sizes of the splits to verify
print(f'Training set size: {X_train.shape[0]} samples')
print(f'Test set size: {X_test.shape[0]} samples')
```

## 4. K-Fold Cross-Validation

**Standard K-Fold Cross-Validation**
```python
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Define the model
model = RandomForestClassifier(random_state=42)

# Define the K-Fold cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=42)  # 5-Fold cross-validation

# Perform cross-validation
scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')

# Output the scores for each fold and the mean score
print(f'Cross-validation scores: {scores}')
print(f'Mean cross-validation score: {scores.mean():.2f}')
```

**Stratified K-Fold Cross-Validation**
```python
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Define the model
model = RandomForestClassifier(random_state=42)

# Define the Stratified K-Fold cross-validator
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)  # 5-Fold stratified cross-validation

# Perform cross-validation
scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')

# Output the scores for each fold and the mean score
print(f'Stratified Cross-validation scores: {scores}')
print(f'Mean stratified cross-validation score: {scores.mean():.2f}')
```

## 5. Leave-One-Out Cross-Validation (LOOCV)

**Implementing LOOCV with `LeaveOneOut`**
```python
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Define the model
model = RandomForestClassifier(random_state=42)

# Define the Leave-One-Out cross-validator
loo = LeaveOneOut()

# Perform cross-validation
scores = cross_val_score(model, X, y, cv=loo, scoring='accuracy')

# Output the mean score
print(f'LOOCV mean accuracy score: {scores.mean():.2f}')
```