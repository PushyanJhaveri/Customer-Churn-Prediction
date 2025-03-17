# 1. Create tenure in months
data['tenure_months'] = data['tenure'] * 12

# 2. Create a feature for total charges per month
data['charges_per_month'] = data['TotalCharges'] / data['tenure_months']

# 3. Create binary features for categorical variables
data['is_male'] = (data['gender'] == 'Male').astype(int)
data['has_partner'] = (data['Partner'] == 'Yes').astype(int)
data['has_dependents'] = (data['Dependents'] == 'Yes').astype(int)
data['has_internet_service'] = (data['InternetService'] != 'No').astype(int)

# 4. One-hot encoding for categorical variables
data = pd.get_dummies(data, columns=['Contract', 'PaymentMethod', 'InternetService'], drop_first=True)

# 5. Create interaction features
data['tenure_charges_interaction'] = data['tenure_months'] * data['charges_per_month']

# 6. Normalize numerical features
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
numerical_features = ['tenure_months', 'charges_per_month', 'TotalCharges']
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# 7. Drop original columns that are no longer needed
data.drop(['gender', 'Partner', 'Dependents', 'InternetService', 'TotalCharges', 'tenure'], axis=1, inplace=True)

# 8. Final check of the engineered data
print("\nFeature Engineered Data Summary:")
print(data.info())

# Save the feature-engineered data to a new CSV file
data.to_csv('feature_engineered_data.csv', index=False)
print("\nFeature engineered data saved to 'feature_engineered_data.csv'.")
