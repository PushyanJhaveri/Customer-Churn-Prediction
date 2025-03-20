# Customer-Churn-Prediction-Model-and-Retention-Strategies

## Project Structure
```
Customer-Churn-Prediction/
│
├── data/
│   ├── Customer-Churn.csv                # Original dataset
│   ├── cleaned_customer_data.csv         # Cleaned dataset for Tableau
│
├── notebooks/
│   ├── 01_data_exploration.ipynb         # Python script for data exploration and cleaning
│   ├── 02_model_development.ipynb        # Python script for model development and evaluation
│
├── sql/
│   ├── create_customers_table.sql        # SQL script to create the customers table
│   ├── churn_analysis_queries.sql        # SQL queries for churn analysis
│
├── src/
│   ├── data_cleaning.py                   # Python script for data cleaning
│   ├── feature_engineering.py             # Python script for feature engineering
│   ├── model_training.py                  # Python script for model training and evaluation
│   ├── hyperparameter_tuning.py           # Python script for hyperparameter tuning
│
├── visualizations/
│   ├── churn_visualizations.py            # Python script for visualizations (optional)
│   ├── Tableau_Dashboard_Screenshots/     # Folder for screenshots of Tableau dashboards
│   ├── app.py                             # Dash app visualization
│   ├── composite_dashboard.py             # Dasboard in png
│
├── requirements.txt                       # Python dependencies
├── README.md                              # Project overview and instructions
└── LICENSE                                 # License file (License)
```


The **Customer Churn Prediction** project aims to develop a machine learning model that predicts customer churn for one of our client of Fingertips. This model was developed back in 2021. By analyzing customer data, the model identifies customers who are likely to leave, enabling the company to implement proactive retention strategies.
This project leverages data cleaning, exploratory data analysis (EDA), feature engineering, and machine learning techniques to achieve its objectives. The aim of the model was to reduce the churn rate and retain the customers back for our client.

## Dataset
The dataset used in this project is one our clients at Fingertips, which contains information about customers, including demographics, account details, and churn status. This data is of October 2020.

## Project Structure
The project is organized into the following directories:

- **data/**: Contains the original and cleaned datasets used in the analysis.
- **notebooks/**: Jupyter notebooks for data exploration, cleaning, and model development.
- **sql/**: SQL scripts for creating the database schema and performing churn analysis.
- **src/**: Python scripts for data cleaning, feature engineering, model training, and hyperparameter tuning.
- **visualizations/**: Contains visualizations and Tableau-related files, including screenshots of dashboards.

## Installation
To run this project, you need to have Python installed along with the required libraries. You can install the dependencies using:
```bash
pip install -r requirements.txt
```

## Usage
- **Data Exploration/**: Explore the data in the Jupyter notebooks located in the notebooks/ directory.
- **Data Cleaning/**: Run the Python scripts in the src/ directory for data cleaning and preprocessing.
- **Model Development/**: Train the machine learning model using the scripts in the src/ directory.
- **SQL Analysis/**: Use the SQL scripts in the sql/ directory to analyze churn data.
- **Visualization/**: Visualize the results in Tableau and refer to the screenshots in the visualizations/ directory.
- **Model Deployment/**: AWS.

## Results
Results & Performance
Churn Rate Reduction
After implementing the predictive model and retention strategies:

Predicted Churn Rate: Reduced by 22%
Increase in Retained Customers: 15% higher than the control group
ROC-AUC Curve (Best Model)
The Random Forest Classifier provided the best predictive performance with an AUC score of 0.91, outperforming other models.

Confusion Matrix Results

|  Metrics  |    Value   |
| ----------| ---------- |
| Accuracy  |    87%     |
| Precision |    85%     |
|  Recall   |    81%     |
| F1-Score  |    83%     |

Customer Retention Success
By identifying at-risk customers early and applying targeted retention strategies:

Customer loyalty programs increased retention by 18%.
Personalized discounts helped reduce churn by 12%.
Improved customer service engagement lowered dissatisfaction and complaints by 35%.


## Acknowledgments
Scikit-learn for machine learning tools.<br />
Pandas for data manipulation and analysis.<br />
Seaborn and Matplotlib for data visualization.<br />
Tableau for creating interactive dashboards.<br />
AWS for model deployment
