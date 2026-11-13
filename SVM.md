# Support Vector Machine Coding Concepts

This document contains exact code snippets extracted from the provided notebooks, categorized by concept. These non-overlapping snippets represent the specific implementations and syntax used across the coursework.

Notebooks used in making this context file: `SVM_code2.ipynb`, `Kernel_view_code1.ipynb`, `SVM_Para_Tuning.ipynb`

## 1. Library Imports and Dataset Loading

**Importing Libraries for Breast Cancer SVM Classification**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
```

**Loading the Built-in Breast Cancer Dataset**
```python
data = load_breast_cancer()
```

**Importing Libraries for Iris SVM and Kernel Visualization**
```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib qt
```

**Importing Libraries for SVM Parameter Tuning**
```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
```

## 2. Dataset Inspection and Data Preparation

**Inspecting Breast Cancer Target Class Names**
```python
list(data.target_names)
```

**Inspecting Breast Cancer Feature Names**
```python
list(data.feature_names)
```

**Selecting Features and Target, Splitting Breast Cancer Data, and Training Linear SVM**
```python
from sklearn import svm, metrics
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
classify = svm.SVC(kernel='linear')
classify.fit(X_train, y_train)
```

**Selecting Two Iris Features and Splitting Data for Visualization**
```python
iris = datasets.load_iris()
X = iris.data[:,:2] #only 2 features is used for visualization
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

## 3. SVM Kernel Selection and Model Training

**Configuring Linear, RBF, or Polynomial Kernels and Fitting the SVC**
```python
#gamma = 'scale' - automatic setting
#When you set gamma='scale', scikit-learn automatically calculates the value of gamma as:
# 1/(number of features * variance of the features)
#C-set = [0.1, 1, 10, 100, 1000]
#gammas = [0.1, 1, 10, 100]

model = SVC(kernel='linear', C=1.0, gamma='scale') 
#model = SVC(kernel='rbf', C=1.0, gamma='scale')
#model = SVC(kernel='poly', C=1.0, gamma='scale')

model.fit(X_train,y_train)
```

## 4. Kernel Decision Boundary and Support Vector Visualization

**Creating a Prediction Mesh, Plotting the Decision Boundary, and Highlighting Support Vectors**
```python
#Create a mesh to plot the decision boundary
h = .02  # Step size in the mesh
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Plot the decision boundary by assigning a color to each point in the mesh
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)

# Plot the training points
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('SVM Decision Boundary with Linear Kernel')

# Plot the support vectors
plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100, facecolors='none', edgecolors='k')

plt.show()
```

## 5. Classification Predictions and Evaluation

**Evaluating Breast Cancer SVM with Accuracy, Precision, and Recall**
```python
y_predict = classify.predict(X_test)
print('Model Accuracy:', round(metrics.accuracy_score(y_test, y_predict),4))
print('Precision:', round(metrics.precision_score(y_test, y_predict), 4))
print('Recall:', round(metrics.recall_score(y_test, y_predict), 4))
```

**Evaluating the Iris SVM with Prediction Accuracy**
```python
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')
```

## 6. SVM Hyperparameter Tuning with GridSearchCV

**Preparing Two Iris Features and a Train-Test Split for Parameter Tuning**
```python
iris = datasets.load_iris()
X = iris.data[:,:2] #only 2 features is used for visualization
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

**Defining Candidate C, Gamma, and Kernel Values and Finding the Best SVC**
```python
cSetG = [0.1, 1, 10]
gammaG = [0.1, 1, 10]
kernelsG = ['linear', 'rbf', 'poly']

from sklearn.model_selection import GridSearchCV
model = SVC()
gridSearch = GridSearchCV(estimator=model,param_grid={'kernel':kernelsG,'C':cSetG,'gamma':gammaG},cv=3)
gridSearch.fit(X,y)
print(f'The best parameter is {gridSearch.best_params_}')
print(f'The best score is {round(gridSearch.best_score_,2)}')
print(f'The best estimator is {gridSearch.best_estimator_}')
```
