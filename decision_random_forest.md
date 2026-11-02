# Decision Tree and Random Forest Coding Concepts

This document contains exact code snippets extracted from the provided notebooks, categorized by concept. These non-overlapping snippets represent the specific implementations and syntax used across the coursework.

Notebooks used in making this context file: `DecisionTree_cart_classification_1.ipynb`, `DecisionTree_cart_classification_2.ipynb`, `DecisionTree_cart_regression.ipynb`, `RandomForest.ipynb`, `RandomForest_3.ipynb`, `RandomForest-Regression.ipynb`


## 1. Dataset Loading, Generation, and Preprocessing

**Importing Required Libraries for Decision Tree & Regression**
```python
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
```

**Loading Built-in Iris Dataset**
```python
iris = load_iris()
```

**Configuring Predictors and Continuous Target Variable**
```Python
# Use first 3 features as predictors
X = iris.data[:, :3]

# Predict Petal Width (continuous value)
y = iris.data[:, 3]
```

**Extracting and Inspecting Feature Names**
```Python
# Feature Names
feature_names = iris.feature_names[:3]
print(feature_names)
```

**Generating Synthetic Classification Data for Random Forest Experiments**
```Python
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, n_features=20, n_informative=5, n_redundant=2,
                           n_repeated=0, n_classes=2, random_state=42)
```

**Converting Synthetic Data to Pandas DataFrame and Splitting**
```Python

feature_names = [f'feature_{i}' for i in range(X.shape[1])]
X = pd.DataFrame(X, columns=feature_names)
y = pd.Series(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

**Splitting Data for Decision Tree Regressor**
```Python
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
```

## 2. Model Training and Fitting

**Creating and Fitting the CART Decision Tree Regressor**
```Python
cart_reg_model = DecisionTreeRegressor(criterion="squared_error", max_depth=4, min_samples_split=5, min_samples_leaf=2,random_state=42)
cart_reg_model.fit(X_train,y_train)
```

**Training an Initial Random Forest Classifier (Before Feature Selection)**
```Python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=10, random_state=42)
rf.fit(X_train, y_train)
```

**Training a Random Forest Classifier with Selected Features***
```Python
rf_selected = RandomForestClassifier(n_estimators=100, random_state=42)
rf_selected.fit(X_train_selected, y_train)
```

**Training a Random Forest Regressor on CSV Data (e.g., Boston Dataset)**
```Python
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(random_state=0)
rf.fit(X_train,y_train)
```

## 3. Model Evaluation and Performance Metrics

**Generating Test Set Predictions (Decision Tree Regressor)**
```Python
y_pred = cart_reg_model.predict(X_test)
```

**Calculating Regression Metrics (MSE, MAE, R²)**
```Python
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Performance")
print("-------------------------")
print(f"MSE : {mse:.4f}")
print(f"MAE : {mae:.4f}")
print(f"R²  : {r2:.4f}")
```

**Evaluating Random Forest Accuracy Before and After Feature Selection**
```Python
accuracy_before = rf.score(X_test, y_test)
print(f'Accuracy before feature selection: {accuracy_before:.2f}')

accuracy_after = rf_selected.score(X_test_selected, y_test)
print(f'Accuracy after feature selection: {accuracy_after:.2f}')
```

## 4. Feature Importance and Advanced Selection (RFECV / Random Forest)

**Extracting and Ranking Feature Importances from Random Forest**
```Python
importances = rf.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df)
```

**Selecting Top Features Based on Importance**
```Python
top_features = feature_importance_df['Feature'][:10].values
X_train_selected = X_train[top_features]
X_test_selected = X_test[top_features]
```

**Recursive Feature Elimination with Cross-Validation (RFECV)**
```Python
from sklearn.feature_selection import RFECV

rfe = RFECV(rf, cv=5, scoring="neg_mean_squared_error")
rfe.fit(X_train, y_train)

selected_features_mask = rfe.support_
selected_features = features[selected_features_mask]
print(selected_features)
```

## 5. Inference with New Unseen Data

**Making Predictions with a Decision Tree Regressor on New Input**
```Python
# Format: sepal_length, sepal_width, petal_length
new_flower = [[5.1, 3.5, 1.4]]

predicted_petal_width = cart_reg_model.predict(new_flower)

print("\nPrediction")
print("-------------------------")
print("Input:", new_flower)
print("Predicted Petal Width:", round(predicted_petal_width[0], 3))
```

## 6. Tree Visualization

**Plotting and Rendering a Decision Tree Structure**
```Python
%matplotlib inline
plt.figure(figsize=(20, 8))    # width=20 inches, height=8 inches
plot_tree(cart_reg_model, filled=True, feature_names=feature_names, rounded=True, fontsize=9)
plt.show()
```