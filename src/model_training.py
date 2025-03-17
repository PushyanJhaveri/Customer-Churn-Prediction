# Define features and target variable
X = data.drop('Churn', axis=1)  # Assuming 'Churn' is the target variable
y = data['Churn']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Define a function to evaluate models
def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"Model: {model.__class__.__name__}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC Score: {roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]):.2f}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}\n")

# 1. Train and evaluate a Random Forest model
rf_model = RandomForestClassifier(random_state=42)
evaluate_model(rf_model, X_train, y_train, X_test, y_test)

# 2. Train and evaluate a Logistic Regression model
lr_model = LogisticRegression(max_iter=1000)
evaluate_model(lr_model, X_train, y_train, X_test, y_test)

# 3. Train and evaluate a Support Vector Classifier model
svc_model = SVC(probability=True)
evaluate_model(svc_model, X_train, y_train, X_test, y_test)

# 4. Hyperparameter Tuning for Random Forest
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, 
                           scoring='roc_auc', cv=StratifiedKFold(n_splits=5), n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)

# Best parameters and score
print("Best Parameters for Random Forest:", grid_search.best_params_)
print("Best ROC AUC Score from Grid Search:", grid_search.best_score_)

# Evaluate the best model from grid search
best_rf_model = grid_search.best_estimator_
evaluate_model(best_rf_model, X_train, y_train, X_test, y_test)

# Save the best model using joblib
import joblib
joblib.dump(best_rf_model, 'best_random_forest_model.pkl')
print("Best Random Forest model saved as 'best_random_forest_model.pkl'.")
