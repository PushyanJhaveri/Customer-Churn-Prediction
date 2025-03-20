"""
Composite Dashboard Image Generator
-------------------------------------
This script creates a comprehensive composite image combining multiple visualizations,
including bar charts, scatter plots, heatmaps, pie charts, and more using Plotly subplots.
The final image is saved as a PNG file and can be embedded on your website.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# File paths
TRAIN_DATA = "data/customer_churn_dataset-training-master.csv"
TEST_DATA = "data/customer_churn_dataset-testing-master.csv"

# Load datasets
train_df = pd.read_csv(TRAIN_DATA).dropna()
test_df = pd.read_csv(TEST_DATA).dropna()

# 1. Churn Distribution (Training)
churn_counts_train = train_df["Churn"].value_counts().sort_index()
churn_fig_train = px.bar(
    x=churn_counts_train.index.astype(str),
    y=churn_counts_train.values,
    labels={'x': 'Churn (0 = No, 1 = Yes)', 'y': 'Count'},
    title="Churn Dist. (Training)"
)

# 2. Churn Distribution (Testing)
churn_counts_test = test_df["Churn"].value_counts().sort_index()
churn_fig_test = px.bar(
    x=churn_counts_test.index.astype(str),
    y=churn_counts_test.values,
    labels={'x': 'Churn (0 = No, 1 = Yes)', 'y': 'Count'},
    title="Churn Dist. (Testing)"
)

# 3. Scatter Plot: Tenure vs. Total Spend by Churn (Training)
scatter_fig_churn = px.scatter(
    train_df,
    x="Tenure",
    y="Total Spend",
    color="Churn",
    title="Tenure vs. Total Spend by Churn",
    labels={"Tenure": "Tenure", "Total Spend": "Total Spend"}
)

# 4. Correlation Heatmap (Training)
numeric_train = train_df.select_dtypes(include=["number"]).drop(columns=["CustomerID"])
corr_train = numeric_train.corr()
heatmap_fig_train = go.Figure(data=go.Heatmap(
    z=corr_train.values,
    x=corr_train.columns,
    y=corr_train.columns,
    colorscale='Viridis'
))
heatmap_fig_train.update_layout(title="Corr. Heatmap (Training)")

# 5. Correlation Heatmap (Testing)
numeric_test = test_df.select_dtypes(include=["number"]).drop(columns=["CustomerID"])
corr_test = numeric_test.corr()
heatmap_fig_test = go.Figure(data=go.Heatmap(
    z=corr_test.values,
    x=corr_test.columns,
    y=corr_test.columns,
    colorscale='Viridis'
))
heatmap_fig_test.update_layout(title="Corr. Heatmap (Testing)")

# 6. Customer Segmentation using KMeans (Training)
seg_data = train_df[["Total Spend", "Support Calls"]].copy()
scaler = StandardScaler()
seg_scaled = scaler.fit_transform(seg_data)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(seg_scaled)
seg_data["Cluster"] = clusters.astype(str)
seg_data["CustomerID"] = train_df["CustomerID"]
segmentation_fig = px.scatter(
    seg_data,
    x="Total Spend",
    y="Support Calls",
    color="Cluster",
    hover_data=["CustomerID"],
    title="Customer Segmentation"
)

# 7. Feature Importance (Simulated)
features = ["Support Calls", "Contract Length", "Total Spend", "Payment Delay", "Age"]
importances = [0.35, 0.30, 0.20, 0.10, 0.05]
feature_imp_fig = px.bar(
    x=features,
    y=importances,
    labels={'x': 'Feature', 'y': 'Importance'},
    title="Feature Importance",
    text=importances
)
feature_imp_fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')

# 8. Pie Chart: Gender Distribution (Training)
gender_counts_train = train_df["Gender"].value_counts()
pie_fig_gender_train = px.pie(
    values=gender_counts_train.values,
    names=gender_counts_train.index,
    title="Gender Distribution"
)

# 9. Pie Chart: Subscription Type Distribution (Training)
subscription_counts_train = train_df["Subscription Type"].value_counts()
pie_fig_subscription_train = px.pie(
    values=subscription_counts_train.values,
    names=subscription_counts_train.index,
    title="Subscription Type Distribution"
)

# -------------------------------
# Create Composite Figure with Subplots
# -------------------------------
# Define subplot specs with type "domain" for pie charts.
specs = [
    [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
    [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}],
    [{"type": "xy"}, {"type": "domain"}, {"type": "domain"}]
]

fig = make_subplots(
    rows=3, cols=3,
    specs=specs,
    subplot_titles=[
        "Churn Dist. (Training)",
        "Churn Dist. (Testing)",
        "Tenure vs. Total Spend",
        "Corr. Heatmap (Training)",
        "Corr. Heatmap (Testing)",
        "Customer Segmentation",
        "Feature Importance",
        "Gender Distribution",
        "Subscription Type Distribution"
    ],
    vertical_spacing=0.1,
    horizontal_spacing=0.05
)

# Populate subplots with traces
for trace in churn_fig_train.data:
    fig.add_trace(trace, row=1, col=1)
for trace in churn_fig_test.data:
    fig.add_trace(trace, row=1, col=2)
for trace in scatter_fig_churn.data:
    fig.add_trace(trace, row=1, col=3)
for trace in heatmap_fig_train.data:
    fig.add_trace(trace, row=2, col=1)
for trace in heatmap_fig_test.data:
    fig.add_trace(trace, row=2, col=2)
for trace in segmentation_fig.data:
    fig.add_trace(trace, row=2, col=3)
for trace in feature_imp_fig.data:
    fig.add_trace(trace, row=3, col=1)
for trace in pie_fig_gender_train.data:
    fig.add_trace(trace, row=3, col=2)
for trace in pie_fig_subscription_train.data:
    fig.add_trace(trace, row=3, col=3)

fig.update_layout(
    height=1200,
    width=1600,
    title_text="Comprehensive Customer Churn Analysis Dashboard",
    showlegend=False
)

# Save composite image (requires Kaleido)
fig.write_image("comprehensive_dashboard_with_scatter.png")
print("Composite dashboard image saved as 'comprehensive_dashboard_with_scatter.png'.")
