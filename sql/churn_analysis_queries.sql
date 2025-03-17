-- Calculate churn rate by gender
SELECT gender, 
       COUNT(*) AS total_customers, 
       SUM(CASE WHEN churn = 1 THEN 1 ELSE 0 END) AS churned_customers,
       (SUM(CASE WHEN churn = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*)) AS churn_rate
FROM customers
GROUP BY gender;

-- Analyze churn by contract type
SELECT contract, 
       COUNT(*) AS total_customers, 
       SUM(CASE WHEN churn = 1 THEN 1 ELSE 0 END) AS churned_customers,
       (SUM(CASE WHEN churn = 1 THEN 1 ELSE 0 END) * 100.0 / COUNT(*)) AS churn_rate
FROM customers
GROUP BY contract;
