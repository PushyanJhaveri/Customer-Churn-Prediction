import matplotlib.pyplot as plt

# Bar chart for churn distribution
churn_counts = data['Churn'].value_counts()
plt.bar(churn_counts.index, churn_counts.values, color=['lightblue', 'blue'])
plt.title('Churn Distribution')
plt.xlabel('Churn Status')
plt.ylabel('Number of Customers')
plt.xticks([0, 1], ['No', 'Yes'])
plt.show()
