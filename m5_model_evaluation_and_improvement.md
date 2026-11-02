# Model Evaluation and Improvement Coding Concepts

This document contains exact code snippets extracted from the Model Evaluation and Improvement module PDF, categorized by concept. These non-overlapping snippets represent the specific implementations and syntax used across hyperparameter optimization, cross-validation, classification metrics, confusion matrices, and regression metrics.

## 1. Hyperparameter Optimization Techniques

**Manual Search Hyperparameter Optimization**

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = datasets.load_iris()
X = data.data
y = data.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define a range of hyperparameters to search over
n_estimators_options = [50, 100, 200]
max_depth_options = [None, 10, 20, 30]
min_samples_split_options = [2, 5, 10]

# Manually search through the hyperparameters
best_score = 0
best_params = {}

for n_estimators in n_estimators_options:
    for max_depth in max_depth_options:
        for min_samples_split in min_samples_split_options:
            # Initialize the model with the current hyperparameters
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                random_state=42
            )
            # Train the model
            model.fit(X_train, y_train)
            # Make predictions
            predictions = model.predict(X_test)
            # Evaluate the model
            score = accuracy_score(y_test, predictions)
            # Update the best parameters if the current score is higher
            if score > best_score:
                best_score = score
                best_params = {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'min_samples_split': min_samples_split
                }

print(f'Best Score: {best_score:.2f}')
print('Best Parameters:', best_params)

```

**Grid Search Hyperparameter Optimization**

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the model
model = RandomForestClassifier(random_state=42)

# Define the parameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                           cv=5, n_jobs=-1, scoring='accuracy')

# Fit the model
grid_search.fit(X_train, y_train)

# Print the best parameters and the corresponding score
print(f'Best Parameters: {grid_search.best_params_}')
print(f'Best Score: {grid_search.best_score_:.2f}')

# Evaluate the best model on the test set
best_model = grid_search.best_estimator_
test_predictions = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

```

**Randomized Search Hyperparameter Optimization**

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import randint

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the model
model = RandomForestClassifier(random_state=42)

# Define the parameter distribution
param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': randint(2, 11)
}

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, 
                                   n_iter=100, cv=5, n_jobs=-1, random_state=42, scoring='accuracy')

# Fit the model
random_search.fit(X_train, y_train)

# Print the best parameters and the corresponding score
print(f'Best Parameters: {random_search.best_params_}')
print(f'Best Score: {random_search.best_score_:.2f}')

# Evaluate the best model on the test set
best_model = random_search.best_estimator_
test_predictions = best_model.predict(X_test)
test_accuracy = accuracy_score(y_test, test_predictions)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

```

**Bayesian Optimization**

```python
from bayes_opt import BayesianOptimization
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the function to optimize
def rf_cv(n_estimators, max_depth, min_samples_split):
    model = RandomForestClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        min_samples_split=int(min_samples_split),
        random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    return accuracy

# Set the range for hyperparameters
param_bounds = {
    'n_estimators': (50, 200),
    'max_depth': (10, 30),
    'min_samples_split': (2, 10)
}

# Initialize Bayesian Optimization
optimizer = BayesianOptimization(
    f=rf_cv,
    pbounds=param_bounds,
    random_state=42
)

# Perform optimization
optimizer.maximize(
    init_points=5,
    n_iter=25
)

# Print the best parameters and the corresponding score
print(f"Best Parameters: {optimizer.max['params']}")
print(f"Best Score: {optimizer.max['target']:.2f}")

```

## 2. Data Splitting and Validation Techniques

**Hold-Out Training and Test Split**

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the data into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Output the sizes of the splits to verify
print(f'Training set size: {X_train.shape[0]} samples')
print(f'Test set size: {X_test.shape[0]} samples')

```

**Training, Validation, and Test Split**

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

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

**Random Sampling Partitioning**

```python
import numpy as np
from sklearn import datasets

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

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

**K-Fold Cross-Validation**

```python
from sklearn import datasets
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

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

**Stratified Cross-Validation**

```python
from sklearn import datasets
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

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

**Leave-One-Out Cross-Validation (LOOCV)**

```python
from sklearn import datasets
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Define the model
model = RandomForestClassifier(random_state=42)

# Define the Leave-One-Out cross-validator
loo = LeaveOneOut()

# Perform cross-validation
scores = cross_val_score(model, X, y, cv=loo, scoring='accuracy')

# Output the mean score
print(f'LOOCV mean accuracy score: {scores.mean():.2f}')

```

**Shuffle-Split Cross-Validation**

```python
from sklearn.model_selection import ShuffleSplit
shuffle_split = ShuffleSplit(test_size=.5, train_size=.5, n_splits=10)
scores = cross_val_score(logreg, iris.data, iris.target, cv=shuffle_split)
print("Cross-validation scores: \n{}".format(scores))

```

## 3. Classification Metrics & Confusion Matrices

**Calculating Macro-Averaged Precision, Recall, and F1-Score**

```python
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train the classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

precision = precision_score(y_test, predictions, average='macro')
recall = recall_score(y_test, predictions, average='macro')
f1 = f1_score(y_test, predictions, average='macro')

print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')

```

**Generating Confusion Matrix and Visualization**

```python
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10, criterion="entropy")
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.show()

```

**Generating Classification Report**

```python
from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred, target_names=iris.target_names)
print(report)

```

**Calculating Specificity**

```python
from sklearn.metrics import confusion_matrix

# True Labels
y_true = [0, 0, 1, 1, 0, 1, 0, 0, 1, 0]

# Predicted Labels
y_pred = [0, 0, 1, 0, 0, 1, 0, 1, 1, 0]

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)
print(f'Confusion Matrix:\n{cm}')

# Extract TN and FP from the confusion matrix
tn, fp, fn, tp = cm.ravel()

# Calculate specificity
specificity = tn / (tn + fp)
print(f'Specificity: {specificity:.2f}')

```

**ROC-AUC Calculation and Plotting**

```python
from sklearn.metrics import roc_auc_score, roc_curve

# Predict probabilities
y_probs = model.predict_proba(X_test)[:, 1]

# Compute AUC
auc_score = roc_auc_score(y_test, y_probs)
print(f'AUC: {auc_score:.2f}')

# Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_probs)
plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc='lower right')
plt.show()

```

## 4. Regression Metrics

**Mean Absolute Error (MAE)**

```python
from sklearn.metrics import mean_absolute_error

true_values = [2.5, 3.7, 1.8, 4.0, 5.2]
predicted_values = [2.1, 3.9, 1.7, 3.8, 5.0]

mae = mean_absolute_error(true_values, predicted_values)
print('Mean Absolute Error: ', round(mae, 2))

```

**Mean Squared Error (MSE)**

```python
from sklearn.metrics import mean_squared_error

true_values = [2.5, 3.7, 1.8, 4.0, 5.2]
predicted_values = [2.1, 3.9, 1.7, 3.8, 5.0]

mse = mean_squared_error(true_values, predicted_values)
print('Mean Squared Error: ', round(mse, 2))

```

**Root Mean Squared Error (RMSE)**

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np

# Sample data
true_prices = np.array([250000, 300000, 200000, 400000, 350000])
predicted_prices = np.array([240000, 310000, 210000, 380000, 340000])

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(true_prices, predicted_prices))
print('Root Mean Squared Error: ', round(rmse, 2))

```

**R-Squared ($R^2$) Score**

```python
from sklearn.metrics import r2_score

true_values = [2.5, 3.7, 1.8, 4.0, 5.2]
predicted_values = [2.1, 3.9, 1.7, 3.8, 5.0]

r2 = r2_score(true_values, predicted_values)
print('R-squared (R²) Score: ', round(r2, 2))

```