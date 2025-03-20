"""
Customer Churn Analysis Dashboard
----------------------------------
This Dash application provides an interactive dashboard with multiple tabs to analyze
customer churn data, view model insights, and explore retention strategies.
"""

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# File paths (adjust if needed)
TRAIN_DATA = "data/customer_churn_dataset-training-master.csv"
TEST_DATA = "data/customer_churn_dataset-testing-master.csv"

# Load datasets
train_df = pd.read_csv(TRAIN_DATA).dropna()
test_df = pd.read_csv(TEST_DATA).dropna()

# -------------------------------
# Create Visualizations for Training Data
# -------------------------------
# Churn Distribution (Training)
churn_counts_train = train_df["Churn"].value_counts().sort_index()
churn_fig_train = px.bar(
    x=churn_counts_train.index.astype(str),
    y=churn_counts_train.values,
    labels={'x': 'Churn (0 = No, 1 = Yes)', 'y': 'Count'},
    title="Churn Distribution (Training)"
)

# Correlation Heatmap (Training) - numeric columns only
numeric_train = train_df.select_dtypes(include=["number"]).drop(columns=["CustomerID"])
corr_train = numeric_train.corr()
heatmap_fig_train = go.Figure(data=go.Heatmap(
    z=corr_train.values,
    x=corr_train.columns,
    y=corr_train.columns,
    colorscale='Viridis'
))
heatmap_fig_train.update_layout(title="Correlation Heatmap (Training)")

# -------------------------------
# Create Visualizations for Testing Data
# -------------------------------
# Churn Distribution (Testing)
churn_counts_test = test_df["Churn"].value_counts().sort_index()
churn_fig_test = px.bar(
    x=churn_counts_test.index.astype(str),
    y=churn_counts_test.values,
    labels={'x': 'Churn (0 = No, 1 = Yes)', 'y': 'Count'},
    title="Churn Distribution (Testing)"
)

# Correlation Heatmap (Testing)
numeric_test = test_df.select_dtypes(include=["number"]).drop(columns=["CustomerID"])
corr_test = numeric_test.corr()
heatmap_fig_test = go.Figure(data=go.Heatmap(
    z=corr_test.values,
    x=corr_test.columns,
    y=corr_test.columns,
    colorscale='Viridis'
))
heatmap_fig_test.update_layout(title="Correlation Heatmap (Testing)")

# -------------------------------
# Additional Visualizations (Training Data)
# -------------------------------
# Scatter Plot: Tenure vs. Total Spend by Churn (Training)
scatter_fig_churn = px.scatter(
    train_df,
    x="Tenure",
    y="Total Spend",
    color="Churn",
    title="Tenure vs. Total Spend by Churn (Training)",
    labels={"Tenure": "Tenure", "Total Spend": "Total Spend"}
)

# -------------------------------
# Simulated Feature Importance (Model Insights)
# -------------------------------
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

# -------------------------------
# Additional Pie Charts (Training Data)
# -------------------------------
# Gender Distribution
gender_counts_train = train_df["Gender"].value_counts()
pie_fig_gender_train = px.pie(
    values=gender_counts_train.values,
    names=gender_counts_train.index,
    title="Gender Distribution (Training)"
)

# Subscription Type Distribution
subscription_counts_train = train_df["Subscription Type"].value_counts()
pie_fig_subscription_train = px.pie(
    values=subscription_counts_train.values,
    names=subscription_counts_train.index,
    title="Subscription Type Distribution (Training)"
)

# -------------------------------
# Customer Segmentation using KMeans (Training Data)
# -------------------------------
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

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

# -------------------------------
# Build the Dash Dashboard Layout
# -------------------------------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Customer Churn Analysis Dashboard", className="text-center mb-4"), width=12)
    ]),
    dbc.Tabs([
        dbc.Tab(label="Overview", tab_id="overview"),
        dbc.Tab(label="Training Data Analysis", tab_id="training"),
        dbc.Tab(label="Testing Data Analysis", tab_id="testing"),
        dbc.Tab(label="Model Insights", tab_id="model"),
        dbc.Tab(label="Additional Visualizations", tab_id="additional"),
        dbc.Tab(label="Customer Segmentation", tab_id="segmentation"),
        dbc.Tab(label="Retention Strategies", tab_id="retention"),
    ], id="tabs", active_tab="overview"),
    html.Div(id="tab-content", className="p-4")
], fluid=True)

@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab")
)
def render_tab_content(active_tab):
    if active_tab == "overview":
        return [
            html.H3("Dashboard Overview"),
            dcc.Markdown("""
                **Customer Churn Analysis Dashboard** offers interactive insights into customer behavior.
                Navigate through the tabs to review training/testing data analyses, model insights, segmentation, 
                additional visualizations, and retention strategies.
            """)
        ]
    elif active_tab == "training":
        return [
            html.H3("Training Data Analysis"),
            dcc.Graph(figure=churn_fig_train),
            dcc.Graph(figure=heatmap_fig_train),
            dcc.Markdown("""
                **Insights (Training Data):**  
                - The churn distribution reveals the proportion of churned vs. retained customers.  
                - The correlation heatmap identifies key relationships among features.
            """)
        ]
    elif active_tab == "testing":
        return [
            html.H3("Testing Data Analysis"),
            dcc.Graph(figure=churn_fig_test),
            dcc.Graph(figure=heatmap_fig_test),
            dcc.Markdown("""
                **Insights (Testing Data):**  
                - Analyze the churn distribution and feature correlations in the testing set.
                - Compare these insights with training data to validate patterns.
            """)
        ]
    elif active_tab == "model":
        return [
            html.H3("Model Insights"),
            dcc.Markdown("""
                Different models (e.g., Logistic Regression, Random Forest, XGBoost) were applied for churn prediction.
                The bar chart below represents simulated feature importance scores that indicate which factors most influence churn.
            """),
            dcc.Graph(figure=feature_imp_fig)
        ]
    elif active_tab == "additional":
        return [
            html.H3("Additional Visualizations"),
            dcc.Graph(figure=pie_fig_gender_train),
            dcc.Graph(figure=pie_fig_subscription_train),
            dcc.Graph(figure=scatter_fig_churn),
            dcc.Markdown("""
                **Additional Insights:**  
                - **Gender Distribution:** Overview of gender makeup in the training data.  
                - **Subscription Type Distribution:** Frequencies of different subscription types.  
                - **Tenure vs. Total Spend:** Scatter plot illustrating customer spending and tenure by churn.
            """)
        ]
    elif active_tab == "segmentation":
        return [
            html.H3("Customer Segmentation"),
            dcc.Graph(figure=segmentation_fig),
            dcc.Markdown("""
                **Customer Segmentation Analysis:**  
                This scatter plot segments customers based on their Total Spend and Support Calls.
                Clusters are identified using KMeans clustering (with 3 clusters in this example). 
                Hover over the points to view the CustomerID.
            """)
        ]
    elif active_tab == "retention":
        return [
            html.H3("Retention Strategies"),
            dcc.Markdown("""
                **Actionable Strategies Based on Analysis:**  
                - **High-Risk Customers:** Identified by frequent support calls, delayed payments, and monthly contracts.  
                  *Retention Tactics:* Personalized outreach, improved support, and flexible payment options.
                
                - **Medium-Risk Customers:** Display early signs of disengagement (e.g., reduced usage).  
                  *Retention Tactics:* Re-engagement campaigns and incentives for longer-term contracts.
                
                - **Low-Risk Customers:** Loyal customers on annual contracts.  
                  *Retention Tactics:* Reward programs, referrals, and proactive monitoring.
            """)
        ]
    return html.P("This tab is under construction.")

if __name__ == '__main__':
    # For Dash versions > 2.0, use app.run() instead of app.run_server()
    app.run(debug=True)
