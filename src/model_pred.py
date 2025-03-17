import joblib

# Load the best trained model
model = joblib.load('best_random_forest_model.pkl')

# Load the feature-engineered dataset (or new data for prediction)
# For demonstration, let's assume we are using the same dataset for predictions
data = pd.read_csv('feature_engineered_data.csv')

# Prepare the data for prediction
# Assuming 'Churn' is the target variable and we want to predict it
X = data.drop('Churn', axis=1)  # Drop the target variable

# Make predictions
predictions = model.predict(X)
predicted_probabilities = model.predict_proba(X)[:, 1]  # Probability of the positive class

# Add predictions to the original DataFrame
data['Predicted_Churn'] = predictions
data['Predicted_Probability'] = predicted_probabilities

# Display the first few rows of the DataFrame with predictions
print(data[['Predicted_Churn', 'Predicted_Probability']].head())

# Save the predictions to a new CSV file
data.to_csv('predictions_with_churn.csv', index=False)
print("Predictions saved to 'predictions_with_churn.csv'.")
