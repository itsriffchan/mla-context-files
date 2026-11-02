# Logistic Regression Coding Concepts

This document contains exact code snippets extracted from the provided notebooks, categorized by concept. These non-overlapping snippets represent the specific implementations and syntax used across the coursework.

## 1. Dataset Loading and Inspection

**Importing Libraries and Loading Breast Cancer Dataset**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.datasets import load_breast_cancer

df_cancer = load_breast_cancer()
```

**Inspecting Dataset Dictionary Keys**
```python
print(df_cancer.keys())
```

**Converting Dataset to a Pandas DataFrame**
```python
df = pd.DataFrame(df_cancer.data, columns=df_cancer.feature_names)
df.head()
```

**Checking Unique Targets and Target Names**
```python
pd.unique(df_cancer.target)
list(df_cancer.target_names)
```

**Listing Feature Names and Inspecting a Row**
```python
list(df_cancer.feature_names)
print(df.iloc[0])
```

## 2. Data Preparation and Train-Test Split

**Splitting Data into Training and Testing Sets**
```python
from sklearn.model_selection import train_test_split

X = df_cancer.data
y = df_cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
```

## 3. Model Training and Evaluation

**Building and Fitting the Logistic Regression Classifier**
```python
from sklearn.linear_model import LogisticRegression

logistic = LogisticRegression(max_iter=100)
logistic.fit(X_train, y_train)
```

**Evaluating Training Accuracy**
```python
print("Model Accuracy - Training : {:.2f}".format(logistic.score(X_train, y_train)))
```

**Evaluating Testing Accuracy**
```python
print("Model Accuracy - Testing: {:.2f}".format(logistic.score(X_test, y_test)))
```

## 4. Inference with New Unseen Data

**Generating Predictions for Test Data**
```python
y_predict = logistic.predict(X_test)
print('Predicted Labels for Y:', y_predict)
```

## 5. Data Scaling for Logistic Regression

**Standardizing Feature Data using StandardScaler**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
```

**Splitting Scaled Data and Training Logistic Regression**
```python
X = scaled_df
y = df_cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

logistic = LogisticRegression(max_iter=100)
logistic.fit(X_train, y_train)

print("Model Accuracy - Training (Scaled): {:.2f}".format(logistic.score(X_train, y_train)))
print("Model Accuracy - Testing (Scaled): {:.2f}".format(logistic.score(X_test, y_test)))
```