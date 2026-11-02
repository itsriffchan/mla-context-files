# Linear Regression, Ridge, and Lasso Coding Concepts

This document contains exact code snippets extracted from the provided coursework notebooks, categorized by concept. These non-overlapping snippets represent the specific implementations and syntax used across single linear regression, multiple linear regression, and regularized regression models.

## 1. Data Loading and Preparation

**Loading the Iris Dataset using Seaborn**

```
import seaborn as sns
iris = sns.load_dataset("iris")
iris

```

**Filtering Specific Columns for Single Linear Regression**

```
iris = iris[['petal_length', 'petal_width']]
iris

```

**Defining Predictors and Target for Single Linear Regression**

```
X = iris[['petal_length']]
y = iris[['petal_width']]

```

**Plotting a Scatter Plot**

```
import matplotlib.pyplot as plt
plt.scatter(X, y)
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Scatter Plot of Petal Length vs. Petal Width')

```

## 2. Train-Test Splitting and Array Reshaping

**Splitting Data into Training and Testing Sets**

```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=23)

```

**Reshaping Train and Test Arrays into 2D Column Vectors**

```
import numpy as np
X_train = np.array(X_train).reshape(-1,1)
X_test = np.array(X_test).reshape(-1,1)

```

## 3. Model Training and Prediction

**Fitting Single Linear Regression Model**

```
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)

```

**Extracting Model Intercept and Coefficients**

```
c = lr.intercept_
m = lr.coef_

```

**Computing Predictions Manually vs via Predict Method**

```
Y_pred_train = m*X_train + c
Y_pred_train.flatten()

Y_pred_train1 = lr.predict(X_train)
Y_pred_train1.flatten()

```

**Visualizing the Best-Fit Line on Training Data**

```
import matplotlib.pyplot as plt
plt.scatter(X_train, y_train, color='blue')
plt.plot(X_train, Y_pred_train1, color='red')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.show()

```

**Predicting on Testing Data and Plotting Best-Fit Line**

```
Y_pred_test1 = lr.predict(X_test)

import matplotlib.pyplot as plt
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, Y_pred_test1, color='red')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.show()

```

## 4. Model Evaluation Metrics

**Calculating Training and Testing R² Scores**

```
from sklearn.metrics import r2_score
r2_train = r2_score(y_train, Y_pred_train1)
print('R2 train scores: ', r2_train)

r2_test = r2_score(y_test, Y_pred_test1)
print('R2 train scores: ', r2_test)

```

## 5. Multiple Linear Regression Pipeline

**Loading Boston Housing Data, Defining Features, and Fitting Multiple Linear Regression**

```
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

iris = sns.load_dataset("iris")
X = iris[["sepal_width", "petal_length", "petal_width"]]
y = iris["sepal_length"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)

print("Coefficients:", lr.coef_)
print("Intercept:", lr.intercept_)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

```

**Making Predictions Using New Data with Multiple Features**

```
new_data = pd.DataFrame({
    "sepal_width": [3.5],
    "petal_length": [4.8],
    "petal_width": [1.8]
})

new_pred = lr.predict(new_data)
print("Predicted sepal length : {:.2f}".format(new_pred[0]))

```

## 6. Ridge and Lasso Regularized Regression

**Loading and Cleaning Boston Housing CSV Data**

```
import pandas as pd
import numpy as np

boston = pd.read_csv('/Users/Fanny/Desktop/CS0075_codes_data/Boston.csv')
boston.dropna(inplace = True)
df = pd.DataFrame(boston)

X = df.iloc[:, 1:14]
y = df['medv']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

```

**Fitting Ridge Regression and Extracting Coefficients/Scores**

```
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0).fit(X_train, y_train)

print("Score : {:.2f}".format(ridge.score(X_train, y_train)))
print("Features Used : {}".format(np.sum(ridge.coef_ != 0)))

```

**Fitting Lasso Regression and Evaluating Feature Sparsity**

```
from sklearn.linear_model import Lasso
lasso = Lasso(alpha=1.0).fit(X_train, y_train)

print("Score : {:.2f}".format(lasso.score(X_train, y_train)))
print("Features Used : {}".format(np.sum(lasso.coef_ != 0)))

```