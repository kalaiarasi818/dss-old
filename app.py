import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

@st.cache_resource # Use this decorator to train the models only once
def train_dss_models(data_path="dss sample data.csv"):
    """Loads data, preprocesses, and trains the Regression and Classification models."""
    try:
        data = pd.read_csv(data_path)
    except FileNotFoundError:
        st.error(f"FATAL ERROR: The file '{data_path}' was not found. Please ensure it is in the same directory.")
        return None, None, None
    except Exception as e:
        st.error(f"FATAL ERROR: Could not load data. Details: {e}")
        return None, None, None

    numerical_features = ['Distance_km', 'Avg_Speed', 'Road_Gradient', 'Fuel_Efficiency_kmpl']
    categorical_features = ['Traffic_Level', 'Vehicle_Type']
    target_regression = 'Carbon_Emission_kg'
    target_classification = 'Sustainable'

    required_cols = numerical_features + categorical_features + [target_regression, target_classification]
    if not all(col in data.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in data.columns]
        st.error(f"FATAL ERROR: Required columns are missing: {', '.join(missing_cols)}")
        return None, None, None

    feature_cols = [col for col in data.columns if col not in ['Route_ID', target_regression, target_classification]]
    X = data[feature_cols]
    y_reg = data[target_regression]
    y_cls = data[target_classification]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='drop' # Ensures only defined features are passed
    )
    reg_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    reg_pipeline.fit(X, y_reg)

    cls_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    cls_pipeline.fit(X, y_cls)

    st.success("Machine Learning Models trained successfully.")
    return reg_pipeline, cls_pipeline, data

st.set_page_config(page_title="DSS Routing Prototype", layout="wide")

st.title("Sustainability-Focused Routing DSS Prototype")
st.markdown("---")

reg_model, cls_model, data = train_dss_models()

if reg_model is None:
    st.stop() # Stop if training failed due to file/column errors

st.sidebar.header("Route Input Module")
st.sidebar.markdown("Define a new route to get a real-time sustainability assessment.")
try:
    distance = st.sidebar.slider("Distance (km)", min_value=5.0, max_value=100.0, value=30.0, step=0.5)
    speed = st.sidebar.slider("Average Speed (km/h)", min_value=20.0, max_value=120.0, value=50.0, step=1.0)
    traffic_options = data['Traffic_Level'].unique().tolist()
    traffic = st.sidebar.selectbox("Traffic Level", traffic_options)
    gradient = st.sidebar.slider("Road Gradient (%)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
    vehicle_options = data['Vehicle_Type'].unique().tolist()
    vehicle_type = st.sidebar.selectbox("Vehicle Type", vehicle_options)
    fuel_efficiency = st.sidebar.slider("Fuel Efficiency (kmpl)", min_value=5.0, max_value=30.0, value=15.0, step=0.1)

    new_route_input = pd.DataFrame({
        'Distance_km': [distance],
        'Avg_Speed': [speed],
        'Traffic_Level': [traffic],
        'Road_Gradient': [gradient],
        'Vehicle_Type': [vehicle_type],
        'Fuel_Efficiency_kmpl': [fuel_efficiency]
    })
    predicted_emissions = reg_model.predict(new_route_input)[0]
    predicted_sustainability = cls_model.predict(new_route_input)[0]

except Exception as e:
    st.error(f"Input or Prediction Error: Please check the sidebar inputs. Details: {e}")
    st.stop()

st.header("1. DSS Decision & Route Flagging (Alert System)")
col1, col2 = st.columns(2)
with col1:
    st.metric(
        label="Predicted Carbon Footprint (kg)",
        value=f"{predicted_emissions:.2f} kg CO2"
    )
with col2:
    if predicted_sustainability == 1:
        st.success(" ROUTE STATUS: SUSTAINABLE (No Flag)")
        st.markdown("This route is estimated to be environmentally efficient.")
    else:
        st.error("ROUTE STATUS: ENVIRONMENTALLY COSTLY (FLAGGED)")
        st.markdown("ALERT: This route is flagged for further review due to high predicted emissions.")

st.markdown("---")
st.header("2. DSS Visualization Dashboard")

tab1, tab2 = st.tabs(["Feature Importance", "Emissions vs. Route Factors"])

with tab1:
    st.subheader("Top Drivers for Carbon Emission")
    feature_names = reg_model.named_steps['preprocessor'].get_feature_names_out()
    importances = reg_model.named_steps['regressor'].feature_importances_

    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)

    feature_importance_df['Feature'] = feature_importance_df['Feature'].str.replace(r'^(num__)|(cat__Traffic_Level_)|(cat__Vehicle_Type_)', '', regex=True)

    fig_feat_imp, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10), palette='viridis', ax=ax)
    ax.set_title('Top 10 Feature Importance for Carbon Emission Prediction')
    ax.set_xlabel('Importance Score')
    ax.set_ylabel('Feature')
    st.pyplot(fig_feat_imp)

with tab2:
    st.subheader("Emissions Distribution by Vehicle Type")

    fig_dist, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='Vehicle_Type', y='Carbon_Emission_kg', data=data, palette='Set2', ax=ax)
    ax.set_title('Carbon Emissions by Vehicle Type (from Training Data)')
    ax.set_xlabel('Vehicle Type')
    ax.set_ylabel('Carbon Emission (kg)')
    plt.xticks(rotation=15)
    st.pyplot(fig_dist)
st.sidebar.markdown("---")
st.sidebar.header("Human Feedback Module")

if st.sidebar.button("Log DSS Decision to Database"):
    st.sidebar.success("✅ **DSS Decision Logged!** (Simulated)")
    st.sidebar.caption("Route data and prediction stored in a simulated log.")
