# Customer-Churn-Prediction

## Project Structure
```
Customer-Churn-Prediction/ │ ├── data/ │ ├── Telco-Customer-Churn.csv # Original dataset │ ├── cleaned_customer_data.csv # Cleaned dataset for Tableau │ ├── notebooks/ │ ├── 01_data_exploration.ipynb # Jupyter Notebook for data exploration and cleaning │ ├── 02_model_development.ipynb # Jupyter Notebook for model development and evaluation │ ├── sql/ │ ├── create_customers_table.sql # SQL script to create the customers table │ ├── churn_analysis_queries.sql # SQL queries for churn analysis │ ├── src/ │ ├── data_cleaning.py # Python script for data cleaning │ ├── feature_engineering.py # Python script for feature engineering │ ├── model_training.py # Python script for model training and evaluation │ ├── hyperparameter_tuning.py # Python script for hyperparameter tuning │ ├── visualizations/ │ ├── churn_visualizations.py # Python script for visualizations (optional) │ ├── Tableau_Dashboard_Screenshots/ # Folder for screenshots of Tableau dashboards │ ├── requirements.txt # Python dependencies ├── README.md # Project overview and instructions └── LICENSE # License file (e.g., MIT License)
```


The **Customer Churn Prediction** project aims to develop a machine learning model that predicts customer churn for one of our client at Fingertips. This model was developed back in 2021. By analyzing customer data, the model identifies customers who are likely to leave, enabling the company to implement proactive retention strategies.
This project leverages data cleaning, exploratory data analysis (EDA), feature engineering, and machine learning techniques to achieve its objectives.

## Dataset
The dataset used in this project is the **Telco Customer Churn** dataset, which contains information about customers, including demographics, account details, and churn status. The dataset is publicly available on [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

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
