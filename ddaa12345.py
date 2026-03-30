#!/usr/bin/env python
# coding: utf-8

# In[ ]:


df = pd.DataFrame(iris.data, columns=iris.feature_names)
feature_names = iris.feature_names
target_names = iris.target_names
df['target'] = iris.target
X = df.drop("target", axis=1)
y = df["target"]
# 2. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# 3. Train Naive Bayes Model
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# 4. Prediction
pred = nb_model.predict(X_test)

# 5. Evaluation
print("Test Accuracy:", accuracy_score(y_test, pred))
print("\nClassification Report:\n", classification_report(y_test, pred))

# 6. Confusion Matrix
sns.heatmap(confusion_matrix(y_test, pred), annot=True, cmap='Blues')from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# 1. Load the Iris dataset
iris = datasets.load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
feature_names = iris.feature_names
target_names = iris.target_names
df['target'] = iris.target
X = df.drop("target", axis=1)
y = df["target"]
# 2. Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# 3. Train Naive Bayes Model

nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# 4. Prediction
pred = nb_model.predict(X_test)

# 5. Evaluation
print("Test Accuracy:", accuracy_score(y_test, pred))
print("\nClassification Report:\n", classification_report(y_test, pred))
# 6. Confusion Matrix
sns.heatmap(confusion_matrix(y_test, pred), annot=True, cmap='Blues')
plt.title("Confusion Matrix (Naive Bayes)")
plt.title( Confusion Matrix (Naive Bayes) )
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 7. Hyperparameter Grid
param_grid = {
    'var_smoothing': np.logspace(0, -9, num=100)
}
# 8. GridSearchCV with 5-fold Cross Validation
cv_nb = GridSearchCV(
    nb_model,
    param_grid,
    cv=5,
    scoring='accuracy'
)
cv_nb.fit(X_train, y_train)
print("Best Parameter:", cv_nb.best_params_)
# 9. Use Best Model
best_nb = cv_nb.best_estimator_
pred = best_nb.predict(X_test)
# 10. Evaluation
print("\nOptimized Accuracy:", accuracy_score(y_test, pred))

