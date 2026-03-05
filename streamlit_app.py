import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import LabelEncoder
import numpy as np

# Set page config
st.set_page_config(
    page_title="Supply Chain Disruption Predictor",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("🏭 Supply Chain Disruption Prediction")
st.markdown("---")

# Load model
@st.cache_resource
def load_model():
    return joblib.load("model/supply_chain_model.pkl")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("data/supply_chain.csv")

model = load_model()
df = load_data()

# Create disruption label
df['disruption'] = (
    (df['Availability'] < 20) |
    (df['Stock levels'] < 20) |
    (df['Lead time'] > 20) |
    (df['Defect rates'] > 3) |
    (df['Shipping times'] > 7)
).astype(int)

df.drop(['SKU','Customer demographics','Inspection results'], axis=1, inplace=True)

# Encode categorical columns
cat_cols = [
    'Product type',
    'Shipping carriers',
    'Supplier name',
    'Location',
    'Transportation modes',
    'Routes'
]

le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

X = df.drop('disruption', axis=1)

# Generate predictions
df['Predicted Risk'] = model.predict(X)

# Define recommendation function
def recommendation(row):
    if row['Predicted Risk'] == 1:
        if row['Stock levels'] < 30:
            return "🚨 Increase inventory and reorder early"
        elif row['Lead time'] > 20:
            return "⚠️ Switch to alternate supplier or route"
        else:
            return "👁️ Monitor supplier closely"
    else:
        return "✅ No immediate action required"

df['Recommendation'] = df.apply(recommendation, axis=1)

# Display metrics
col1, col2, col3 = st.columns(3)

with col1:
    risk_count = (df['Predicted Risk'] == 1).sum()
    st.metric("High Risk Records", risk_count, f"out of {len(df)}")

with col2:
    safe_count = (df['Predicted Risk'] == 0).sum()
    st.metric("Safe Records", safe_count, f"out of {len(df)}")

with col3:
    risk_percentage = (risk_count / len(df)) * 100
    st.metric("Risk Percentage", f"{risk_percentage:.1f}%")

st.markdown("---")

# Display predictions table
st.subheader("📋 Detailed Predictions")
display_df = df[['Supplier name', 'Product type', 'Stock levels', 'Lead time', 'Predicted Risk', 'Recommendation']].copy()

# Color code the predictions
def style_risk(val):
    if val == 1:
        return 'background-color: #ffcccc'
    return 'background-color: #ccffcc'

styled_df = display_df.style.map(style_risk, subset=['Predicted Risk'])
st.dataframe(styled_df, use_container_width=True)

# Summary statistics
st.markdown("---")
st.subheader("📊 Summary Statistics")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.write("**Avg Stock Levels (High Risk)**")
    high_risk_stock = df[df['Predicted Risk'] == 1]['Stock levels'].mean()
    st.write(f"{high_risk_stock:.1f}")

with col2:
    st.write("**Avg Lead Time (High Risk)**")
    high_risk_lead = df[df['Predicted Risk'] == 1]['Lead time'].mean()
    st.write(f"{high_risk_lead:.1f}")

with col3:
    st.write("**Avg Defect Rate (High Risk)**")
    high_risk_defect = df[df['Predicted Risk'] == 1]['Defect rates'].mean()
    st.write(f"{high_risk_defect:.2f}%")

with col4:
    st.write("**Avg Shipping Time (High Risk)**")
    high_risk_shipping = df[df['Predicted Risk'] == 1]['Shipping times'].mean()
    st.write(f"{high_risk_shipping:.1f}")

# Distribution chart
st.markdown("---")
st.subheader("📈 Risk Distribution")

col1, col2 = st.columns(2)

with col1:
    risk_dist = df['Predicted Risk'].value_counts()
    st.bar_chart(risk_dist)

with col2:
    # Recommendations breakdown
    rec_counts = df['Recommendation'].value_counts()
    st.bar_chart(rec_counts)

st.markdown("---")
st.info("💡 This model predicts supply chain disruptions based on key metrics like availability, stock levels, lead time, defect rates, and shipping times.")
