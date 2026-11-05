# Gradient Boosting Coding Concepts

This document contains code snippets extracted from the provided Gradient Boosting notebooks, categorized by concept. The snippets are non-redundant and preserve the notebooks' terminology, parameter choices, and implementation details. Classification and regression workflows are separated wherever their datasets, estimators, evaluation methods, or tuning steps differ.

Notebooks used in making this context file: `GB-Class-Code1.ipynb`, `GB-Class-code2 1.ipynb`, `GB-Regression-code2 2.ipynb`

## 1. Classification Imports and Estimator

**Importing the Tools Used in the Wine Classification Workflow**

```python
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
```

**Importing the Tools Used in the Income Classification Workflow**

```python
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import preprocessing
#import warnings
#warnings.filterwarnings("ignore")
```

The classification notebooks use `GradientBoostingClassifier`. The Wine workflow also uses `cross_val_score`, while the Income workflow evaluates train/test scores and a confusion matrix before tuning.

## 2. Classification Dataset Loading and Inspection

### 2.1 Wine Dataset

**Loading the Built-in Wine Dataset as DataFrames**

```python
wine = datasets.load_wine(as_frame=True)
```

**Accessing and Inspecting Feature and Target DataFrames**

```python
# Access the DataFrame containing the feature data 
wine_data = wine.data
# Access the DataFrame containing the target data 
wine_target = wine.target
# Display the first few rows of the feature DataFrame 
print(wine_data.head())
# Display the first few rows of the target DataFrame 
print(wine_target.head())
```

**Accessing Wine Feature Names**

```python
wine_features = wine.feature_names
wine_features
```

### 2.2 Income Evaluation Dataset

**Loading the Income CSV File and Previewing It**

```python
df = pd.read_csv('/Users/Fanny/Desktop/CS0075_codes_data/gradient boosting/income_evaluation.csv')
df.head()
```

**Inspecting Column Names and Dataset Information**

```python
print(df.columns)
df.info()
```

The notebook notes that the column names have a leading space that needs to be handled. The subsequent code uses names such as `workclass`, `occupation`, `native_country`, `native-country`, and `income` exactly as shown below.

## 3. Classification Data Preprocessing and Preparation

### 3.1 Wine Feature and Target Selection

**Selecting the Wine Data and Target from the Dataset Bunch**

```python
X = wine['data']
y = wine['target']
```

**Creating the Wine Training and Testing Sets**

```python
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=17)
```

### 3.2 Income Missing-Value Handling and Label Encoding

**Replacing Question-Mark Values with `NaN`**

```python
#deal with the '?' values
df.loc[df['workclass']=='?', 'workclass']= np.NaN
df.loc[df['occupation']=='?', 'occupation']= np.NaN
df.loc[df['native_country']=='?', 'native_country']= np.NaN
```

**Creating Separate Label Encoders for Income Dataset Columns**

```python
from sklearn.preprocessing import LabelEncoder
le_workclass = LabelEncoder()      
le_fnlwgt = LabelEncoder()         
le_education = LabelEncoder() 
le_marital_status =LabelEncoder()
le_occupation = LabelEncoder()
le_relationship = LabelEncoder()
le_race = LabelEncoder()
le_sex = LabelEncoder()
le_native_country = LabelEncoder()
le_income = LabelEncoder()
```

**Encoding the Income Dataset Columns**

```python
df['workclass'] = le_workclass.fit_transform(df['workclass'])
df['fnlwgt'] = le_fnlwgt.fit_transform(df['fnlwgt'])
df['education'] = le_education.fit_transform(df['education'])
df['marital-status'] = le_marital_status.fit_transform(df['marital-status'])
df['occupation'] = le_occupation.fit_transform(df['occupation'])
df['relationship'] = le_relationship.fit_transform(df['relationship'])
df['race'] = le_race.fit_transform(df['race'])
df['sex'] = le_sex.fit_transform(df['sex'])
df['native-country'] = le_native_country.fit_transform(df['native-country'])
df['income'] = le_native_country.fit_transform(df['income'])
```

The final line above intentionally preserves the notebook's implementation: `income` is transformed with `le_native_country`, even though `le_income` is created in the preceding snippet.

**Inspecting the Encoded Income Data**

```python
print(df)
df.info()
```

**Separating Income Features and Target**

```python
y = df['income']
X = df.drop('income', axis=1)
```

**Creating the Income Training and Testing Sets**

```python
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state = 0)
```

## 4. Classification Model Training and Evaluation

### 4.1 Default Classifier with Cross-Validation on Wine Data

**Building and Fitting the Gradient Boosting Classifier**

```python
gbc = GradientBoostingClassifier()
gbc.fit(X_train,y_train)
```

**Calculating the Mean Three-Fold Cross-Validation Score**

```python
cross_val_score(gbc,X_train,y_train,cv=3, n_jobs=-1).mean()
```

### 4.2 Configured Classifier with Train/Test Scoring on Income Data

**Defining the Classifier with Explicit Hyperparameters**

```python
# Define Gradient Boosting Classifier with hyperparameters

gbc=GradientBoostingClassifier(n_estimators=100,learning_rate=0.05,random_state=100,max_features=5 )
```

**Fitting the Classifier**

```python
# Fit train data to GBC
gbc.fit(X_train,y_train)
```

**Calculating Training and Testing Scores**

```python
gbc.score(X_train,y_train)
gbc.score(X_test,y_test)
```

**Displaying the Confusion Matrix**

```python
# Confusion matrix will give number of correct and incorrect classifications

print(confusion_matrix(y_test, gbc.predict(X_test)))
```

**Calculating Accuracy Manually from the Confusion-Matrix Counts**

```python
cm_accuracy = (4687+874)/(4687+231+721+874)
print(cm_accuracy)
```

## 5. Classification Hyperparameter Tuning

### 5.1 Joint Grid Search on Wine Data

**Defining a Grid for Estimators, Learning Rate, and Tree Depth**

```python
#Parameter Tuning

param_grid = {
    'n_estimators': [10,50,100,500],
    'learning_rate':[0.0001,0.001,0.01,0.1,1.0],
    'max_depth': [3,5,7,9] 
}
```

**Running Three-Fold Grid Search in Parallel**

```python
gbc2 = GridSearchCV(gbc,param_grid, cv=3, n_jobs=-1)
gbc2.fit(X_train, y_train)
```

**Displaying the Best Parameter Values and Cross-Validation Score**

```python
#Diplay the best parameter value
gbc2.best_params_

#Display the model score using best parameter values
gbc2.best_score_
```

### 5.2 Staged Grid Search on Income Data

**Tuning `learning_rate` and `n_estimators` First**

```python
#Tuning n_estimators and Learning rate

from sklearn.model_selection import GridSearchCV
grid = {
    'learning_rate':[0.01,0.05,0.1],
    'n_estimators':np.arange(100,500,100)        #resulting array will be: [100, 200, 300, 400]
}
```

**Running Four-Fold Grid Search and Reporting Train/Test Results**

```python
gb = GradientBoostingClassifier()
gb_cv = GridSearchCV(gb, grid, cv = 4)
gb_cv.fit(X_train,y_train)

print("Best Parameters:",gb_cv.best_params_)
print("Train Score:",gb_cv.best_score_)
print("Test Score:",gb_cv.score(X_test,y_test))
```

Here, the notebook labels `gb_cv.best_score_` as `Train Score`; the value is the best mean cross-validation score produced by `GridSearchCV` on the training data.

**Tuning `max_depth` with the Selected Learning Rate and Estimator Count**

```python
#tune max_depth parameter
grid = {'max_depth':[2,3,4,5,6,7] }
gb = GradientBoostingClassifier(learning_rate=0.1,n_estimators=400)
gb_cv = GridSearchCV(gb, grid, cv = 4)
gb_cv.fit(X_train,y_train)
```

**Reporting the Best Depth and Final Train/Test Results**

```python
print("Best Parameters:",gb_cv.best_params_)
print("Train Score:",gb_cv.best_score_)
print("Test Score:",gb_cv.score(X_test,y_test))
```

## 6. Regression Imports and Estimator

**Importing the Tools Used in the Diabetes Regression Workflow**

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.datasets import load_diabetes
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```

Unlike the classification workflows, this notebook uses `GradientBoostingRegressor` and evaluates continuous predictions with mean absolute error.

## 7. Regression Dataset Loading and Preparation

**Loading the Built-in Diabetes Dataset**

```python
data = load_diabetes()
```

**Creating a DataFrame and Adding the Continuous Target**

```python
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = pd.Series(data.target)       #create new col target
```

**Inspecting Data Types and Missing Values**

```python
#simple data preprocessing

df.info()        #check the info about the dataset
               #no need for value encoding since all features is float data type 
               #no missing value as well
```

**Checking for Duplicate Records**

```python
df[df.duplicated()]    #no duplicate records as well
```

**Separating Regression Features and Target**

```python
#Data Preparation

X = df.drop('target',axis=1)
y = df['target']
```

**Creating the Regression Training and Testing Sets**

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
```

## 8. Regression Model Training and Evaluation

**Building a 500-Estimator Regressor and Generating Predictions**

```python
#Create GBR Model and predict
gbr = GradientBoostingRegressor(n_estimators = 500)
gbr.fit(X_train, y_train)

y_pred = gbr.predict(X_test)
print(y_pred)
```

**Evaluating Predictions with Mean Absolute Error**

```python
mae = mean_absolute_error(y_test,y_pred)
#mse = mean_squared_error(y_test,y_pred)
#RMSE = mse ** (1/2)

print(mae)	     
```

The notebook identifies hyperparameter tuning as the next step for minimizing the error, but it does not include a `GradientBoostingRegressor` tuning implementation.

## 9. Regression Feature Importance

**Creating a Descending Feature-Importance Series**

```python
# Feature Importance

feature_scores = pd.Series(gbr.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print(feature_scores)
```

**Plotting Feature Importance with Seaborn**

```python
# Plot the feature importance

sns.barplot(x=feature_scores, y=feature_scores.index)
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
```

## 10. Classification and Regression Workflow Differences

- Classification uses `GradientBoostingClassifier`; regression uses `GradientBoostingRegressor`.
- The Wine classifier is validated with mean three-fold `cross_val_score` and a joint `GridSearchCV` over `n_estimators`, `learning_rate`, and `max_depth`.
- The Income classifier is evaluated with train/test `.score()`, a confusion matrix, and manually calculated accuracy before staged four-fold grid searches.
- The Diabetes regressor predicts continuous target values and is evaluated with `mean_absolute_error`.
- Feature-importance extraction and plotting are implemented only in the regression notebook.
- Hyperparameter tuning code is implemented in both classification notebooks, while the regression notebook only identifies tuning as a proposed next step.
