# Gradient Boosting Coding Concepts

This document contains exact code snippets extracted from the provided notebooks, categorized by concept. These non-overlapping snippets represent the specific implementations and syntax used across the coursework.

Notebooks used in making this context file: `GB-Regression-code2 2.ipynb`, `GB-Class-Code1.ipynb`, `GB-Class-code2 1.ipynb`


## 1. Data Loading and Inspection

**Loading Built-in Datasets & Accessing the `Bunch` Object**
```python
from sklearn.datasets import load_iris
iris_dataset = load_iris()
print("Keys of iris_dataset:\n", iris_dataset.keys())
print("Target :",iris_dataset['target'])
print("Target names :",iris_dataset['target_names'])
```

**Creating a Frequency Dictionary (Using `zip` and `bincount`)**
```python
print("Sample counts per class:\n{}".
      format({n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))
```

**Parsing Raw Text Data into a Dictionary**
```python
raw_data = """
mean radius \t20.570000
mean texture \t17.770000
...
"""
# Convert to dictionary
data_dict = dict(line.rsplit(maxsplit=1) for line in raw_data.strip().split('\n'))
data_dict = {k: float(v) for k, v in data_dict.items()}
new_data = pd.DataFrame([data_dict])
```

## 2. Data Preprocessing

**Handling Missing Values (Replacing '?' with NaN)**
```python
df.loc[df['workclass']=='?', 'workclass']= np.NaN
df.loc[df['occupation']=='?', 'occupation']= np.NaN
df.loc[df['native_country']=='?', 'native_country']= np.NaN
```

**Label Encoding Categorical Features**
```python
from sklearn.preprocessing import LabelEncoder
le_color = LabelEncoder()
df['Color'] = le_color.fit_transform(df['Color'])
```

**Feature Scaling (Standardization)**
```python
from sklearn.preprocessing import StandardScaler  
sc = StandardScaler()  
x_train = sc.fit_transform(x_train)  
x_test = sc.transform(x_test)
```

**Train-Test Split (Standard & Stratified)**
```python
from sklearn.model_selection import train_test_split
# Standard Split
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

# Stratified Split (maintains class proportions)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=66)
```

## 3. K-Nearest Neighbors (KNN)

**Model Complexity Curve (Finding optimal K)**
```python
train_accuracies = []
test_accuracies = []
# try n_neighbors from 1 to 9
neighbors = range(1, 10)
for n in neighbors:
    # build the model
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(X_train, y_train)
    train_accuracies.append(accuracy_score(y_train, knn.predict(X_train)))
    test_accuracies.append(accuracy_score(y_test, knn.predict(X_test)))

plt.plot(neighbors, train_accuracies, label='Training Accuracy')
plt.plot(neighbors, test_accuracies, label='Testing Accuracy')
```

**Manual Accuracy Calculation (Boolean Averaging)**
```python
y_pred = knn.predict(X_test)
print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))
```

## 4. Naive Bayes 

**Categorical Naive Bayes & Frequency Tables (`pd.crosstab`)**
```python
from sklearn.naive_bayes import CategoricalNB
cnb = CategoricalNB()
cnb.fit(X, y)

frequency_table = pd.DataFrame()
for feature in X.columns:
    feature_freq = pd.crosstab(X[feature], y, rownames=[feature], colnames=['Class'])
    feature_freq.columns = le_fruit.inverse_transform(feature_freq.columns)
    frequency_table = pd.concat([frequency_table, feature_freq], axis=1)
```

**Gaussian Naive Bayes: Extracting Parameters & Manual Probabilities**
```python
from sklearn.naive_bayes import GaussianNB  
gnb = GaussianNB()  
gnb.fit(x_train, y_train)

#calculate mean and variance for each feature
means = gnb.theta_
variances = gnb.var_

#Function to calculate Gaussian probability density
def gaussian_probability(x, mean, var):
    exponent = np.exp(-((x - mean) ** 2) / (2 * var))
    return (1 / np.sqrt(2 * np.pi * var)) * exponent

# Create a likelihood table (snippet of inner loop)
        feature_likelihood.append({
            'Class': gnb.classes_[class_idx],
            'Mean': means[class_idx][feature_idx],
            'Variance': variances[class_idx][feature_idx],
            'Likelihood': gaussian_probability(X_test[feature_name], means[class_idx][feature_idx], 
                                               variances[class_idx][feature_idx])
        })
```

## 5. Gradient Boosting

**Gradient Boosting Regressor & Feature Importance**
```python
from sklearn.ensemble import GradientBoostingRegressor
gbr = GradientBoostingRegressor(n_estimators = 500)
gbr.fit(X_train, y_train)

# Feature Importance
feature_scores = pd.Series(gbr.feature_importances_, index=X_train.columns).sort_values(ascending=False)
sns.barplot(x=feature_scores, y=feature_scores.index)
```

**Gradient Boosting Classifier & Hyperparameter Tuning (`GridSearchCV`)**
```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

gbc = GradientBoostingClassifier()
param_grid = {
    'n_estimators': [10,50,100,500],
    'learning_rate':[0.0001,0.001,0.01,0.1,1.0],
    'max_depth': [3,5,7,9] 
}

gbc2 = GridSearchCV(gbc,param_grid, cv=3, n_jobs=-1)
gbc2.fit(X_train, y_train)

#Display the best parameter value and score
print(gbc2.best_params_)
print(gbc2.best_score_)
```

## 6. Model Evaluation Metrics

**Cross-Validation Score**
```python
from sklearn.model_selection import cross_val_score
cross_val_score(gbc,X_train,y_train,cv=3, n_jobs=-1).mean()
```

**Mean Absolute Error (Regression)**
```python
from sklearn.metrics import mean_absolute_error
mae = mean_absolute_error(y_test,y_pred)
```

**Confusion Matrix & Accuracy Score (Classification)**
```python
from sklearn.metrics import accuracy_score, confusion_matrix
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(cm)
```