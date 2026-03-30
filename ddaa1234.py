#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 3. Decision Tree with Entropy (Information Gain)
# We set criterion='entropy' here
dt_model = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt_model.fit(X_train, y_train)
# 4. Initial Evaluation

pred = dt_model.predict(X_test)
print("Initial Accuracy (Entropy):", accuracy_score(y_test, pred))
print("\nClassification Report:\n", classification_report(y_test, pred))
# 5. Visualization
sns.heatmap(confusion_matrix(y_test, pred), annot=True, cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

# 5.b Decision tree visualization
plt.figure(figsize=(12, 6))
plot_tree(dt_model, feature_names=feature_names, class_names=target_names, filled=True)
plt.title("Decision Tree (Iris - Entropy, max_depth=3)")
plt.show()

# 6. Grid Search CV for Decision Tree
# We tune 'max_depth' and 'min_samples_split' for better generalization
param_grid = {
'criterion': ['entropy'], # Keeping it focused on entropy as requested
'max_depth': [3, 4, 5, 6, None],
'min_samples_split': [2, 5, 10],
'min_samples_leaf': [1, 2, 4]
}
cv_dt = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=10)
cv_dt.fit(X_train, y_train)
print("Best parameters:", cv_dt.best_params_)

# 7. Final Evaluation with Best Model
best_preds = cv_dt.best_estimator_.predict(X_test)
print("Optimized Accuracy:", accuracy_score(y_test, best_preds))
# 8. Visualization

sns.heatmap(confusion_matrix(y_test, best_preds), annot=True, cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

df = pd.read_csv('titanic.csv')

# 2. Preprocessing based on your columns
# Drop high-cardinality/unique string columns that don't help the model generalize
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# 4. Decision Tree with Entropy
# Initial model to see baseline performance
dt_model = DecisionTreeClassifier(criterion='entropy', random_state=42)
dt_model.fit(X_train, y_train)

# 5. Initial Evaluation
pred = dt model.predict(X test)
pred  
dt_model.predict(X_test)
print("Initial Accuracy (Entropy):", accuracy_score(y_test, pred))
print("\nClassification Report:\n", classification_report(y_test, pred))
# 6. Visualization

sns.heatmap(confusion_matrix(y_test, pred), annot=True, cmap='Blues')
plt.title("Confusion Matrix")
plt.show()
# 6.b Decision tree visualization

plt.figure(figsize=(12, 6))
# Create a list of string labels for the target classes
class_names_labels = ['Did not Survive', 'Survived']
plot_tree(dt_model, feature_names=X.columns, class_names=class_names_labels, filled=True)
plt.title("Decision Tree (Titanic - Entropy)")
plt.show()

# 7. Grid Search CV for Optimization
# We focus on Entropy while tuning depth to prevent overfitting
param_grid = {
'criterion': ['entropy'],
'max_depth': [3, 4, 5, 6, 8],
'min_samples_split': [2, 5, 10],
'min_samples_leaf': [1, 2, 4]
}
cv_dt = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5)
cv_dt.fit(X_train, y_train)
print(f"Best Parameters found: {cv_dt.best_params_}")

# 8. Evaluation
best_model = cv_dt.best_estimator_
preds = best_model.predict(X_test)
print(f"Final Accuracy: {accuracy_score(y_test, preds):.2%}")
print("\nDetailed Report:\n", classification_report(y_test, preds))

# 7. Visualize Results
sns.heatmap(confusion_matrix(y_test, preds), annot=True, fmt='d', cmap='Blues')
plt.title("Titanic Survival Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

