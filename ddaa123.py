#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import necessary libraries
from sklearn import datasets
import pandas as pd
import sklearn.model_selection as skms
from  sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

# Load the Iris dataset
iris = datasets.load_iris()
# Create a DataFrame for the dataset
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target
df.head()

X = df.drop("target", axis=1)
y = df["target"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.4, random_state=42)
y_train.size

print("Train features shape:", X_train.shape)
print("Test features shape:", X_test.shape)

# default n_neighbors = 3
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
pred = model.predict(X_test)
# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, pred)
print("Accuracy:", accuracy)

# Display Confusion Matrix
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, pred))
# Display Classification Report
from sklearn.metrics import classification_report
print(classification_report(y_test, pred))

import seaborn as sns
sns.heatmap(confusion_matrix(y_test, pred), annot=True)

#Grid Search CV with Cross Validation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
cv_knn = GridSearchCV(
KNeighborsClassifier(),
{
'n_neighbors': range(1, 21, 2),
'weights': ['uniform', 'distance']
},
cv=10
).fit(X_train, y_train)
print("Best parameters:", cv_knn.best_params_)
preds = cv_knn.best_estimator_.predict(X_test)
# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, preds)
print("New Accuracy:", accuracy)

