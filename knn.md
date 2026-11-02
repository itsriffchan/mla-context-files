# K-Nearest Neighbors Coding Concepts

This document contains exact code snippets extracted from the provided notebooks, categorized by concept. These non-overlapping snippets represent the specific implementations and syntax used across the coursework.

Notebooks used in making this context file: `KNN1.ipynb`, `KNN2.ipynb`
## 1. Dataset Loading and Inspection

**Loading Built-in Breast Cancer Dataset & Accessing Keys**
```python
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
print("cancer.keys(): \n{}".format(cancer.keys()))
```

**Checking the Type of the Dataset Bunch Object**
```python
print(type(cancer))
```

**Printing Dataset Description and Characteristics**
```python
print(cancer.DESCR)
```

**Inspecting Shape of Dataset Features**
```python
print("Shape of cancer data: {}".format(cancer.data.shape))
```

**Printing Feature Names (Predictor Columns)**
```python
print(cancer.feature_names)
```

**Inspecting Target Array and Target Classes**
```python
print(cancer.target)
print(cancer.target_names)
```

**Counting Samples Per Class**
```python
print("Sample counts per class:\n{}".
      format({n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}))
```

## 2. Data Preparation and DataFrame Conversion

**Creating a Pandas DataFrame from Feature Data and Target**
```python
df = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
df['target'] = cancer.target
print(df.head())
```

**Extracting a Transposed Row as a DataFrame**
```python
row_train_tuple = pd.DataFrame(df.iloc[1,0:30]).T
row_train_tuple
```

**Splitting Data into Training and Testing Sets with Stratification**
```python
from sklearn.model_selection import train_test_split

X = df.drop(columns='target')
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=66)
```

**Verifying Split Dimensions**
```python
print(X_train.shape, y_train.shape)
```

## 3. Model Training and Evaluation

**Building and Fitting the K-Nearest Neighbors Classifier**
```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=6)
knn.fit(X_train, y_train)
```

**Evaluating Training Predictions and Accuracy**
```python
print("Train test predictions: {}".format(knn.predict(X_train)))
print("Train test accuracy: {}".format(knn.score(X_train,y_train)))
print("Train test accuracy: {:.2f}".format(knn.score(X_train,y_train)))
```

**Evaluating Testing Predictions and Accuracy**
```python
print("Test set predictions: {}".format(knn.predict(X_test)))
print("Test set accuracy: {}".format(knn.score(X_test, y_test)))
print("Test set accuracy: {:.2f}".format(knn.score(X_test, y_test)))
```

## 4. Inference with New Unseen Data

**Parsing Raw Text Data into a Dictionary**
```python
raw_data = """
mean radius 	20.570000
mean texture 	17.770000
mean perimeter 	132.900000
mean area 	1326.000000
mean smoothness 	0.084740
mean compactness 	0.078640
mean concavity 	0.086900
mean concave points 	0.070170
mean symmetry 	0.181200
mean fractal dimension 	0.056670
radius error 	0.543500
texture error 	0.733900
perimeter error 	3.398000
area error 	74.080000
smoothness error 	0.005225
compactness error 	0.013080
concavity error 	0.018600
concave points error 	0.013400
symmetry error 	0.013890
fractal dimension error 	0.003532
worst radius 	24.990000
worst texture 	23.410000
worst perimeter 	158.800000
worst area 	1956.000000
worst smoothness 	0.123800
worst compactness 	0.186600
worst concavity 	0.241600
worst concave points 	0.186000
worst symmetry 	0.275000
worst fractal dimension 	0.089020
"""

data_dict = dict(line.rsplit(maxsplit=1) for line in raw_data.strip().split('\n'))
data_dict = {k: float(v) for k, v in data_dict.items()}
```

**Converting New Data Dictionary to a DataFrame**
```python
new_data = pd.DataFrame([data_dict])
print(new_data)
```

**Making Predictions on New Data**
```python
prediction = knn.predict(new_data)
print("Prediction: {}".format(prediction))
target_names = ['malignant', 'benign']
predicted_label = target_names[prediction[0]]
print("Predicted target name:", predicted_label)
```