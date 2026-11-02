# Hyperparameter Tuning Coding Concepts

This document contains exact code snippets extracted from the provided notebooks, categorized by concept. These non-overlapping snippets represent the specific implementations and syntax used for hyperparameter optimization and evaluation.

Notebooks used in making this context file: `Manual_Search.ipynb`, `Random_Search.ipynb`, `Bayesian_optimization.ipynb`, `Grid_Search.ipynb`

## 1. Dataset Loading and Preparation

**Loading the Iris Dataset and Splitting Data**
```python
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

## 2. Manual Hyperparameter Search

**Defining Hyperparameter Options and Iterating**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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

## 3. Grid Search Optimization

**Configuring and Fitting GridSearchCV**
```python
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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

## 4. Randomized Search Optimization

**Configuring and Fitting RandomizedSearchCV**
```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from scipy.stats import randint

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

## 5. Bayesian Optimization

**Setting Up and Executing BayesianOptimization**
```python
from bayes_opt import BayesianOptimization
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

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