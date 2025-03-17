import seaborn as sns

# Example: Count plot for churn by gender
sns.countplot(data=data, x='gender', hue='Churn', palette='Set2')
plt.title('Churn by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.legend(title='Churn', loc='upper right', labels=['No', 'Yes'])
plt.show()
