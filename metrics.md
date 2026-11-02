# Machine Learning Notebook Coding Concepts

This document contains exact code snippets extracted from the provided notebooks, categorized by concept. These non-overlapping snippets represent the specific implementations and syntax used across evaluation metrics, model comparison, regression metrics, and specificity calculations.

Notebooks used in making this context file: `Confusion_Matrix2.ipynb`, `MAE_MSE_RMSE_R2.ipynb`, `Specificity.ipynb`, `AUC_ROC_3.ipynb`, `AUC_ROC-2.ipynb`, `Classification_report.ipynb`, `Confusion_Matrix1.ipynb`

## 1. Dataset Loading and Model Training

**Loading the Iris Dataset and Splitting Data**
```python
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

**Training the Model and Making Predictions**
```python
# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
```

## 2. Evaluation Metrics & Confusion Matrices

**Computing and Displaying the Confusion Matrix**
```python
# Compute the confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm = confusion_matrix(y_test, y_pred)
print(f'Confusion Matrix:\n{cm}')

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap=plt.cm.Blues)
plt.show()
```

**Calculating Specificity from Confusion Matrix**
```python
from sklearn.metrics import confusion_matrix

# True labels
y_true = [0, 0, 1, 1, 0, 1, 0, 0, 1, 0]

# Predicted labels
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

## 3. Regression Metrics (MAE, MSE, RMSE, R-squared)

**Mean Absolute Error (MAE)**
```python
from sklearn.metrics import mean_absolute_error

true_values = [2.5, 3.7, 1.8, 4.0, 5.2]
predicted_values = [2.1, 3.9, 1.7, 3.8, 5.0]

mae = mean_absolute_error(true_values, predicted_values)
print('Mean Absolute Error : ', round(mae,2))
```

**Mean Squared Error (MSE)**
```python
from sklearn.metrics import mean_squared_error

true_values = [2.5, 3.7, 1.8, 4.0, 5.2]
predicted_values = [2.1, 3.9, 1.7, 3.8, 5.0]

mse = mean_squared_error(true_values, predicted_values)
print('Mean Squared Error : ', round(mse,2))
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
print('Root Mean Squared Error : ', round(rmse,2))
```

**R-squared ($R^2$) Score**
```python
from sklearn.metrics import r2_score

true_values = [2.5, 3.7, 1.8, 4.0, 5.2]
predicted_values = [2.1, 3.9, 1.7, 3.8, 5.0]

r2 = r2_score(true_values, predicted_values)
print('R-squared (R²) Score: ', round(r2,2))
```

## 4. Classification Reports and Performance Metrics

**Generating Classification Reports (Precision, Recall, F1-Score)**
```python
# Generate the classification report
from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred, target_names=iris.target_names)
print(report)
```

**Calculating Macro-Averaged Metrics**
```python
from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_test, predictions, average='macro')
recall = recall_score(y_test, predictions, average='macro')
f1 = f1_score(y_test, predictions, average='macro')

print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
```

## 5. ROC Curves and AUC Comparison

**Comparing Multiple Models with ROC Curves and AUC Scores**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt

# Define the models
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "K-Nearest Neighbors": KNeighborsClassifier()
}

# Initialize a dictionary to store AUC - ROC scores
roc_auc_scores = {}

# Generate a binary classification dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Plot the ROC curves
plt.figure(figsize=(10, 8))

for name, model in models.items():
    model.fit(X_train, y_train)
    y_probs = model.predict_proba(X_test)[:, 1]
    roc_auc = roc_auc_score(y_test, y_probs)
    roc_auc_scores[name] = roc_auc
    
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc="lower right")
plt.show()

# Print the AUC - ROC scores for each model
for name, score in roc_auc_scores.items():
    print(f'{name}: AUC - ROC = {score:.2f}')
```