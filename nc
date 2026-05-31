# Exercise 5 : Naivebayes Classifier 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split, cross_val_score 
from sklearn.naive_bayes import GaussianNB 
from sklearn.metrics import accuracy_score, classification_report, 
confusion_matrix 
# 1. Load your local titanic file 
# Ensure 'titanic.csv' is in the same folder as your script 
df = pd.read_csv('titanic.csv') 
# 2. Preprocessing based on your columns 
# Drop high-cardinality/unique string columns that don't help the model 
generalize 
cols_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin'] 
df = df.drop(columns=cols_to_drop) 
# Handle Missing Values (Crucial for Titanic) 
df['Age'] = df['Age'].fillna(df['Age'].median()) 
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0]) 
# Encode Categorical Data into numbers 
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1}) 
# Convert Embarked (C, Q, S) into dummy variables 
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True) 
# Define Features (X) and Target (y) 
X = df.drop('Survived', axis=1) 
y = df['Survived'] 
# 3. Split the data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
random_state=42) 
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
plt.xlabel("Predicted") 
plt.ylabel("Actual") 
plt.show() 
# 7. Cross Validation (5-Fold) 
cv_scores = cross_val_score(nb_model, X, y, cv=5) 
print("\nCross Validation Scores:", cv_scores) 
print("Average CV Accuracy:", cv_scores.mean())
