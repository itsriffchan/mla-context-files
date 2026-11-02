# Naive Bayes Coding Concepts

This document contains exact code snippets extracted from the provided notebooks, categorized by concept. These non-overlapping snippets represent the specific implementations and syntax used across the coursework.

Notebooks used in making this context file: `CategoricalNB-code2.ipynb`, `GaussianNB-code1.ipynb`, `GaussianNB-Code2.ipynb`
## 1. Data Loading and Inspection

**Loading Data from CSV and Checking Shape**
```python
df = pd.read_csv("/Users/Fanny/Desktop/CS0075_codes_data/diabetes.csv")
df.head()
shape = df.shape
shape
```

**Loading User Data from CSV**
```python
df = pd.read_csv('/Users/Fanny/Desktop/CS0075_codes_data/user_data.csv')  
print(df)
```

## 2. Data Preprocessing and Splitting

**Encoding Categorical Variables with LabelEncoder**
```python
le_color = LabelEncoder()
le_size = LabelEncoder()
le_fruit = LabelEncoder()
df['Color'] = le_color.fit_transform(df['Color'])
df['Size'] = le_size.fit_transform(df['Size'])
df['Fruit'] = le_fruit.fit_transform(df['Fruit'])
```

**Splitting Data into Features and Target (CategoricalNB)**
```python
X = df[['Color', 'Size']]
y = df['Fruit']
```

**Splitting Data into Features and Target (Diabetes Dataset)**
```python
X = df[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
y = df['Outcome']
```

**Splitting Data into Training and Testing Sets with Train-Test Split**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
```

**Extracting Specific Feature Columns via iloc**
```python
X = df.iloc[:, [2, 3]].values  
y = df.iloc[:, 4].values
```

**Splitting Data for StandardScaler Pipeline**
```python
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
```

**Applying Standard Scaler for Feature Scaling**
```python
sc = StandardScaler()  
x_train = sc.fit_transform(x_train)  
x_test = sc.transform(x_test)
```

## 3. Model Training

**Initializing and Fitting Categorical Naive Bayes**
```python
cnb = CategoricalNB()
cnb.fit(X, y)
```

**Initializing and Fitting Gaussian Naive Bayes (Diabetes Dataset)**
```python
gnb = GaussianNB()
gnb.fit(X_train, y_train)
```

**Fitting Gaussian Naive Bayes to Scaled Training Data**
```python
from sklearn.naive_bayes import GaussianNB  
gnb = GaussianNB()  
gnb.fit(x_train, y_train)
```

## 4. Feature Analysis and Probability Calculations

**Creating a Frequency Table via Crosstab**
```python
frequency_table = pd.DataFrame()

for feature in X.columns:
    feature_freq = pd.crosstab(X[feature], y, rownames=[feature], colnames=['Class'])
    feature_freq.columns = le_fruit.inverse_transform(feature_freq.columns)
    frequency_table = pd.concat([frequency_table, feature_freq], axis=1)

print(frequency_table)
```

**Extracting Means and Variances from Gaussian Naive Bayes**
```python
means = gnb.theta_
variances = gnb.var_
```

**Calculating Gaussian Probability Density**
```python
def gaussian_probability(x, mean, var):
    exponent = np.exp(-((x - mean) ** 2) / (2 * var))
    return (1 / np.sqrt(2 * np.pi * var)) * exponent
```

**Creating a Likelihood Table Dictionary**
```python
likelihood_table = {}
for feature_idx, feature_name in enumerate(X.columns):
    feature_likelihood = []
    for class_idx in range(len(gnb.classes_)):
        feature_likelihood.append({
            'Class': gnb.classes_[class_idx],
            'Mean': means[class_idx][feature_idx],
            'Variance': variances[class_idx][feature_idx],
            'Likelihood': gaussian_probability(X_test[feature_name], means[class_idx][feature_idx], 
                                               variances[class_idx][feature_idx])
        })
    likelihood_table[feature_name] = feature_likelihood
```

**Converting Likelihood Table to DataFrame**
```python
likelihood_df = pd.DataFrame({
    (feature, class_info['Class']): class_info['Likelihood']
    for feature, class_likelihoods in likelihood_table.items()
    for class_info in class_likelihoods
})
```

## 5. Model Prediction and Evaluation

**Transforming New Categorical Data for Prediction**
```python
new_data = pd.DataFrame({
    'Color': le_color.transform(['Red']),
    'Size': le_size.transform(['Large'])
})
```

**Making Predictions and Inverse Transforming Results**
```python
prediction = cnb.predict(new_data)
predicted_fruit = le_fruit.inverse_transform(prediction)

print(predicted_fruit)
```

**Generating Model Predictions on Test Data**
```python
y_pred = gnb.predict(X_test)
```

**Measuring Model Accuracy Score**
```python
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
```

**Making and Printing the Confusion Matrix**
```python
from sklearn.metrics import confusion_matrix  
cm = confusion_matrix(y_test, y_pred)  
print(cm)
```

**Predicting with New Unseen Data Points (Diabetes)**
```python
x_new = [[5,183,80,0,90,21.5,0.672,45]]
x_new = pd.DataFrame(x_new, columns=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI',
                                     'DiabetesPedigreeFunction','Age'])
prediction = gnb.predict(x_new)
print("With Diabetes : ", prediction)
```

**Predicting with New Unseen Data Points (User Data)**
```python
x_new = [[45,20000]]
x_pred = gnb.predict(x_new)
print(x_pred)
```