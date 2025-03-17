# Display the first few rows of the dataset
print("Initial Data:")
print(data.head())

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Check for duplicates
duplicates = data.duplicated().sum()
print(f"\nNumber of duplicate rows: {duplicates}")

# Drop duplicates if any
if duplicates > 0:
    data.drop_duplicates(inplace=True)
    print(f"Duplicates removed. New total rows: {data.shape[0]}")

# Data type conversion
# Convert 'TotalCharges' to numeric, handling errors
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')

# Check for any remaining missing values after conversion
print("\nMissing Values After Conversion:")
print(data.isnull().sum())

# Fill missing values
# For 'TotalCharges', fill with the median
data['TotalCharges'].fillna(data['TotalCharges'].median(), inplace=True)

# Check for categorical variables and convert them to category type
categorical_cols = data.select_dtypes(include=['object']).columns
for col in categorical_cols:
    data[col] = data[col].astype('category')

# Display data types after conversion
print("\nData Types After Conversion:")
print(data.dtypes)

# Check for unique values in categorical columns
print("\nUnique Values in Categorical Columns:")
for col in categorical_cols:
    print(f"{col}: {data[col].unique()}")

# Drop unnecessary columns (if any)
# For example, if 'customerID' is not needed for analysis
data.drop(['customerID'], axis=1, inplace=True)

# Final check of the cleaned data
print("\nCleaned Data Summary:")
print(data.info())

# Save the cleaned data to a new CSV file
data.to_csv('cleaned_customer_data.csv', index=False)
print("\nCleaned data saved to 'cleaned_customer_data.csv'.")
