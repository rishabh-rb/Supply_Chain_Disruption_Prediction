# Supply Chain Disruption Prediction System
# Built for predicting supply chain risks and disruptions

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Configure the page
st.set_page_config(
    page_title="Supply Chain Risk Platform",
    page_icon="🏭",
    layout="wide"
)

# Add some custom styling
st.markdown("""
    <style>
    .big-title {
        font-size:42px !important;
        font-weight: bold;
        color: #1f77b4;
    }
    </style>
    """, unsafe_allow_html=True)

# Main title
st.markdown('<p class="big-title">🏭 Supply Chain Disruption Prediction Platform</p>', unsafe_allow_html=True)
st.write("Predict and prevent supply chain disruptions using machine learning")
st.markdown("---")

# Load the trained model and other artifacts
@st.cache_resource
def load_model_stuff():
    """Load model, scaler, encoders etc."""
    model = joblib.load("model/supply_chain_model_advanced.pkl")
    scaler = joblib.load("model/scaler.pkl")
    encoders = joblib.load("model/encoders.pkl")
    
    with open("model/feature_columns.json", "r") as f:
        features = json.load(f)
    
    with open("model/training_metrics.json", "r") as f:
        metrics = json.load(f)
    
    return model, scaler, encoders, features, metrics

@st.cache_data
def load_dataset():
    """Load the supply chain data"""
    return pd.read_csv("data/supply_chain_advanced.csv")

# Load everything
try:
    model, scaler, encoders, feature_cols, train_metrics = load_model_stuff()
    data = load_dataset()
except Exception as e:
    st.error(f"Error loading files: {str(e)}")
    st.stop()

# Sidebar for navigation
st.sidebar.title("Navigation")
view = st.sidebar.radio(
    "Choose View:",
    ["Dashboard", "Make Predictions", "Model Performance", "Data Analytics"]
)

st.sidebar.markdown("---")
st.sidebar.info(f"""
**Model Info:**
- Type: {train_metrics['best_model']}
- Accuracy: {train_metrics['models_performance'][train_metrics['best_model']]['accuracy']:.1%}
- Training Date: {train_metrics['training_date'][:10]}
- Records: {train_metrics['dataset_size']}
""")

# ================================================================
# DASHBOARD VIEW
# ================================================================
if view == "Dashboard":
    st.header("📊 Executive Dashboard")
    st.write("Overview of supply chain risk across your network")
    
    # Show some key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    high_risk = (data['Disruption_Risk'] == 1).sum()
    low_risk = (data['Disruption_Risk'] == 0).sum()
    avg_prob = data['Disruption_Probability'].mean()
    critical = len(data[data['Production_Dependency_Level'] == 'Critical'])
    
    col1.metric("🚨 High Risk", high_risk, f"{high_risk/len(data)*100:.0f}%")
    col2.metric("✅ Low Risk", low_risk, f"{low_risk/len(data)*100:.0f}%")
    col3.metric("📊 Avg Risk Prob", f"{avg_prob:.1f}%")
    col4.metric("⚠️ Critical Deps", critical)
    
    st.markdown("---")
    
    # Regional risk analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk by Region")
        region_risk = data.groupby('Supplier_Region')['Disruption_Risk'].agg(['sum', 'count'])
        region_risk['pct'] = (region_risk['sum'] / region_risk['count'] * 100).round(1)
        
        fig = px.bar(region_risk.reset_index(), 
                     x='Supplier_Region', 
                     y='pct',
                     title="High Risk % by Region",
                     labels={'pct': 'High Risk %'},
                     color='pct',
                     color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Risk by Product Category")
        product_risk = data.groupby('Product_Category')['Disruption_Risk'].agg(['sum', 'count'])
        product_risk['pct'] = (product_risk['sum'] / product_risk['count'] * 100).round(1)
        
        fig = px.bar(product_risk.reset_index(), 
                     x='Product_Category', 
                     y='pct',
                     title="High Risk % by Product",
                     labels={'pct': 'High Risk %'},
                     color='pct',
                     color_continuous_scale='Oranges')
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Top risky suppliers
    st.subheader("🔴 Top 10 Riskiest Suppliers")
    risky = data[data['Disruption_Risk'] == 1].nlargest(10, 'Disruption_Probability')
    display_cols = ['Supplier_Name', 'Supplier_Country', 'Product_Category', 
                    'Disruption_Probability', 'On_Time_Delivery_Rate', 'Financial_Stability_Score']
    st.dataframe(risky[display_cols], use_container_width=True, hide_index=True)
    
    # Geographic map
    st.markdown("---")
    st.subheader("🗺️ Global Risk Map")
    country_data = data.groupby('Supplier_Country').agg({
        'Disruption_Risk': 'sum',
        'Record_ID': 'count'
    }).reset_index()
    country_data.columns = ['Country', 'High_Risk', 'Total']
    country_data['Risk_Pct'] = (country_data['High_Risk'] / country_data['Total'] * 100).round(1)
    
    fig = px.scatter_geo(country_data,
                         locations='Country',
                         locationmode='country names',
                         size='Total',
                         color='Risk_Pct',
                         hover_name='Country',
                         title='Supply Chain Risk by Country',
                         color_continuous_scale='RdYlGn_r')
    st.plotly_chart(fig, use_container_width=True)

# ================================================================
# PREDICTION VIEW
# ================================================================
elif view == "Make Predictions":
    st.header("🔮 Predict Disruption Risk")
    st.write("Enter details about a supplier to predict disruption risk")
    
    tab1, tab2 = st.tabs(["Single Prediction", "Batch Mode"])
    
    with tab1:
        st.subheader("Enter Supplier Details")
        
        # Organize inputs into columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Location & Product**")
            country = st.selectbox("Country", data['Supplier_Country'].unique())
            region = st.selectbox("Region", data['Supplier_Region'].unique())
            category = st.selectbox("Product Category", data['Product_Category'].unique())
            prod_type = st.selectbox("Product Type", data['Product_Type'].unique())
            
        with col2:
            st.markdown("**Operational Metrics**")
            lead_time = st.slider("Lead Time (days)", 5, 60, 25)
            stock = st.slider("Stock Level", 0, 5000, 1500)
            reorder = st.slider("Reorder Point", 200, 2000, 800)
            safety = st.slider("Safety Stock", 100, 1500, 500)
            qty = st.slider("Order Quantity", 100, 10000, 2000)
            
        with col3:
            st.markdown("**Performance**")
            on_time = st.slider("On-Time Delivery %", 65.0, 100.0, 85.0)
            quality = st.slider("Quality Rating %", 65.0, 100.0, 85.0)
            defects = st.slider("Defect Rate %", 0.0, 8.0, 2.0)
            fin_stability = st.slider("Financial Stability", 40.0, 100.0, 75.0)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Risk Factors**")
            geo_risk = st.slider("Geopolitical Risk", 0.0, 10.0, 3.0)
            weather_risk = st.slider("Weather Risk", 0.0, 10.0, 3.0)
            demand_vol = st.slider("Demand Volatility", 0.0, 10.0, 4.0)
            supply_vol = st.slider("Supply Volatility", 0.0, 10.0, 4.0)
            
        with col2:
            st.markdown("**Logistics**")
            transport = st.selectbox("Transport Mode", data['Transportation_Mode'].unique())
            ship_time = st.slider("Shipping Days", 1, 45, 15)
            customs = st.slider("Customs Days", 0, 10, 3)
            port_cong = st.slider("Port Congestion", 0.0, 10.0, 4.0)
            
        with col3:
            st.markdown("**Dependencies**")
            single_source = st.selectbox("Single Source?", ['Yes', 'No'])
            alt_suppliers = st.slider("Alternative Suppliers", 0, 5, 2)
            prod_dep = st.selectbox("Production Dependency", 
                                   ['Critical', 'High', 'Medium', 'Low'])
            hist_disruptions = st.slider("Past Disruptions", 0, 15, 3)
        
        # More inputs
        col1, col2 = st.columns(2)
        with col1:
            inv_turnover = st.slider("Inventory Turnover", 2.0, 12.0, 6.0, 0.1)
            responsiveness = st.slider("Responsiveness Score", 60.0, 100.0, 80.0)
            capacity = st.slider("Capacity Utilization %", 50.0, 100.0, 75.0)
            
        with col2:
            avg_delay = st.slider("Avg Delay (days)", 0.0, 20.0, 5.0, 0.5)
            compliance = st.slider("Contract Compliance %", 70.0, 100.0, 90.0)
            response_time = st.slider("Response Time (hrs)", 1.0, 72.0, 24.0)
        
        col1, col2 = st.columns(2)
        with col1:
            payment = st.selectbox("Payment Terms", [30, 45, 60, 90])
            price_vol = st.slider("Price Volatility %", 0.0, 25.0, 5.0)
            
        with col2:
            dep_score = st.slider("Dependency Score", 0.0, 10.0, 5.0)
            impact = st.slider("Customer Impact", 0.0, 10.0, 5.0)
        
        if st.button("Predict Risk", type="primary"):
            # Prepare the input
            input_dict = {
                'Lead_Time_Days': lead_time,
                'Order_Quantity': qty,
                'Stock_Level': stock,
                'Reorder_Point': reorder,
                'Safety_Stock': safety,
                'Inventory_Turnover': inv_turnover,
                'On_Time_Delivery_Rate': on_time,
                'Quality_Rating': quality,
                'Defect_Rate_Percent': defects,
                'Supplier_Responsiveness_Score': responsiveness,
                'Supplier_Capacity_Utilization': capacity,
                'Financial_Stability_Score': fin_stability,
                'Shipping_Time_Days': ship_time,
                'Customs_Clearance_Time_Days': customs,
                'Geopolitical_Risk_Score': geo_risk,
                'Weather_Risk_Score': weather_risk,
                'Demand_Volatility': demand_vol,
                'Supply_Volatility': supply_vol,
                'Port_Congestion_Level': port_cong,
                'Supplier_Dependency_Score': dep_score,
                'Alternative_Suppliers_Available': alt_suppliers,
                'Historical_Disruptions_Count': hist_disruptions,
                'Average_Delay_Days': avg_delay,
                'Contract_Compliance_Rate': compliance,
                'Communication_Response_Time_Hours': response_time,
                'Payment_Terms_Days': payment,
                'Price_Volatility_Percent': price_vol,
                'Customer_Impact_Score': impact,
                'Supplier_Country_Encoded': encoders['Supplier_Country'].transform([country])[0],
                'Supplier_Region_Encoded': encoders['Supplier_Region'].transform([region])[0],
                'Product_Category_Encoded': encoders['Product_Category'].transform([category])[0],
                'Product_Type_Encoded': encoders['Product_Type'].transform([prod_type])[0],
                'Transportation_Mode_Encoded': encoders['Transportation_Mode'].transform([transport])[0],
                'Single_Source_Risk_Encoded': encoders['Single_Source_Risk'].transform([single_source])[0],
                'Production_Dependency_Level_Encoded': encoders['Production_Dependency_Level'].transform([prod_dep])[0]
            }
            
            # Create dataframe with proper order
            input_df = pd.DataFrame([input_dict])[feature_cols]
            
            # Scale and predict
            input_scaled = scaler.transform(input_df)
            pred = model.predict(input_scaled)[0]
            prob = model.predict_proba(input_scaled)[0][1]
            
            # Show results
            st.markdown("---")
            st.subheader("Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if pred == 1:
                    st.error("### 🚨 HIGH RISK")
                    st.write("**Disruption Likely**")
                else:
                    st.success("### ✅ LOW RISK")
                    st.write("**Stable Supplier**")
            
            with col2:
                st.metric("Probability", f"{prob*100:.1f}%")
                
            with col3:
                level = "Critical" if prob > 0.8 else "High" if prob > 0.6 else "Medium" if prob > 0.4 else "Low"
                st.metric("Risk Level", level)
            
            # Recommendations
            st.markdown("---")
            st.subheader("Recommendations")
            
            if pred == 1:
                recs = []
                if stock < reorder:
                    recs.append("🔴 **Critical**: Stock below reorder point - increase inventory now")
                if on_time < 85:
                    recs.append("🟠 **Important**: Poor delivery performance - find backup suppliers")
                if fin_stability < 70:
                    recs.append("🟠 **Monitor**: Weak financial health - review supplier viability")
                if single_source == 'Yes':
                    recs.append("🟡 **Suggest**: Single source risk - identify alternatives")
                if geo_risk > 7:
                    recs.append("🔴 **Urgent**: High geopolitical risk - diversify sourcing")
                if lead_time > 40:
                    recs.append("🟡 **Review**: Long lead times - explore faster options")
                
                if not recs:
                    recs.append("👁️ Monitor closely and prepare backup plans")
                
                for rec in recs:
                    st.markdown(rec)
            else:
                st.success("✅ No immediate action needed. Continue monitoring.")
    
    with tab2:
        st.subheader("Batch Predictions")
        st.write("Run predictions on multiple records at once")
        
        if st.button("Run Batch Prediction"):
            with st.spinner("Processing..."):
                # Prep data
                df_batch = data.copy()
                
                # Encode categorical vars
                for col in ['Supplier_Country', 'Supplier_Region', 'Product_Category',
                           'Product_Type', 'Transportation_Mode', 'Single_Source_Risk',
                           'Production_Dependency_Level']:
                    df_batch[col + '_Encoded'] = encoders[col].transform(df_batch[col])
                
                X = df_batch[feature_cols]
                X_scaled = scaler.transform(X)
                
                preds = model.predict(X_scaled)
                probs = model.predict_proba(X_scaled)[:, 1]
                
                df_batch['Predicted_Risk'] = preds
                df_batch['Predicted_Prob'] = (probs * 100).round(2)
                
                # Show results
                st.success("Done!")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total", len(df_batch))
                col2.metric("High Risk", (preds == 1).sum())
                col3.metric("Low Risk", (preds == 0).sum())
                
                st.markdown("### Results")
                results = df_batch[['Supplier_Name', 'Supplier_Country', 'Product_Category',
                                   'Predicted_Risk', 'Predicted_Prob', 
                                   'On_Time_Delivery_Rate', 'Financial_Stability_Score']].copy()
                
                results['Label'] = results['Predicted_Risk'].map({0: '✅ Low', 1: '🚨 High'})
                
                filter_risk = st.multiselect("Filter by Risk", 
                                            options=['✅ Low', '🚨 High'],
                                            default=['✅ Low', '🚨 High'])
                
                filtered = results[results['Label'].isin(filter_risk)]
                st.dataframe(filtered, use_container_width=True, hide_index=True)
                
                # Download option
                csv = filtered.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    csv,
                    f"predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv"
                )

# ================================================================
# MODEL PERFORMANCE VIEW
# ================================================================
elif view == "Model Performance":
    st.header("🎯 Model Performance")
    st.write("Detailed model evaluation metrics and analysis")
    
    # Basic info
    st.subheader("Model Info")
    col1, col2, col3, col4 = st.columns(4)
    
    col1.metric("Model Type", train_metrics['best_model'])
    col2.metric("Training Date", train_metrics['training_date'][:10])
    col3.metric("Dataset Size", f"{train_metrics['dataset_size']}")
    col4.metric("Features", train_metrics['num_features'])
    
    st.markdown("---")
    
    # Model comparison
    st.subheader("Model Comparison")
    
    models_df = pd.DataFrame(train_metrics['models_performance']).T
    models_df = models_df[['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']]
    models_df.columns = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    models_df = (models_df * 100).round(2)
    
    st.dataframe(models_df, use_container_width=True)
    
    # Chart
    fig = go.Figure()
    
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']:
        fig.add_trace(go.Bar(
            name=metric,
            x=models_df.index,
            y=models_df[metric],
            text=[f"{v:.1f}" for v in models_df[metric]],
            textposition='auto'
        ))
    
    fig.update_layout(
        title="Performance Comparison",
        xaxis_title="Model",
        yaxis_title="Score (%)",
        barmode='group',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Confusion matrix
    st.subheader("Confusion Matrix")
    
    best = train_metrics['best_model']
    conf_mat = np.array(train_metrics['models_performance'][best]['confusion_matrix'])
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Metrics")
        m = train_metrics['models_performance'][best]
        st.metric("Accuracy", f"{m['accuracy']*100:.2f}%")
        st.metric("Precision", f"{m['precision']*100:.2f}%")
        st.metric("Recall", f"{m['recall']*100:.2f}%")
        st.metric("F1-Score", f"{m['f1_score']*100:.2f}%")
        st.metric("ROC-AUC", f"{m['roc_auc']*100:.2f}%")
    
    with col2:
        fig = go.Figure(data=go.Heatmap(
            z=conf_mat,
            x=['Predicted Low', 'Predicted High'],
            y=['Actual Low', 'Actual High'],
            text=conf_mat,
            texttemplate='%{text}',
            textfont={"size": 20},
            colorscale='Blues'
        ))
        
        fig.update_layout(
            title=f"Confusion Matrix - {best}",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Class distribution
    st.subheader("Data Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Training Stats")
        st.json({
            "Train Size": train_metrics['train_size'],
            "Test Size": train_metrics['test_size'],
            "Features": train_metrics['num_features'],
            "Model": train_metrics['best_model']
        })
    
    with col2:
        st.markdown("### Class Balance")
        class_counts = data['Disruption_Risk'].value_counts()
        fig = px.pie(values=class_counts.values, 
                     names=['High Risk', 'Low Risk'],
                     title='Target Distribution',
                     color_discrete_map={'High Risk': 'red', 'Low Risk': 'green'})
        st.plotly_chart(fig, use_container_width=True)

# ================================================================
# ANALYTICS VIEW
# ================================================================
elif view == "Data Analytics":
    st.header("📈 Data Analytics")
    st.write("Deep dive into supply chain risk patterns")
    
    # Time trends
    st.subheader("Risk Trends Over Time")
    data['Date'] = pd.to_datetime(data['Date'])
    data_sorted = data.sort_values('Date')
    
    monthly = data_sorted.groupby(data_sorted['Date'].dt.to_period('M')).agg({
        'Disruption_Risk': 'mean',
        'Disruption_Probability': 'mean'
    }).reset_index()
    monthly['Date'] = monthly['Date'].astype(str)
    monthly['Disruption_Risk'] = (monthly['Disruption_Risk'] * 100).round(2)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly['Date'], y=monthly['Disruption_Risk'],
                            mode='lines+markers', name='Risk Rate %',
                            line=dict(color='red', width=3)))
    fig.add_trace(go.Scatter(x=monthly['Date'], y=monthly['Disruption_Probability'],
                            mode='lines+markers', name='Avg Prob %',
                            line=dict(color='orange', width=3)))
    
    fig.update_layout(title='Monthly Trends', 
                     xaxis_title='Month', 
                     yaxis_title='%',
                     height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Correlation heatmap
    st.subheader("Feature Correlations")
    
    corr_features = [
        'Lead_Time_Days', 'Stock_Level', 'On_Time_Delivery_Rate',
        'Defect_Rate_Percent', 'Financial_Stability_Score',
        'Geopolitical_Risk_Score', 'Supply_Volatility',
        'Historical_Disruptions_Count', 'Disruption_Risk'
    ]
    
    corr = data[corr_features].corr()
    
    fig = px.imshow(corr,
                    labels=dict(color="Correlation"),
                    x=corr.columns,
                    y=corr.columns,
                    color_continuous_scale='RdBu_r',
                    title='Correlation Matrix')
    
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Supplier and transport analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Top Suppliers")
        supplier_counts = data['Supplier_Name'].value_counts().head(10)
        fig = px.bar(x=supplier_counts.index, y=supplier_counts.values,
                     labels={'x': 'Supplier', 'y': 'Count'},
                     title='Top 10 Suppliers by Volume')
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Transport Modes")
        transport = data['Transportation_Mode'].value_counts()
        fig = px.pie(values=transport.values, 
                     names=transport.index,
                     title='Transportation Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Risk factor distributions
    st.subheader("Risk Factors")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig = px.histogram(data, x='Geopolitical_Risk_Score', 
                          color='Disruption_Risk',
                          title='Geopolitical Risk',
                          color_discrete_map={0: 'green', 1: 'red'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(data, x='Financial_Stability_Score',
                          color='Disruption_Risk',
                          title='Financial Stability',
                          color_discrete_map={0: 'green', 1: 'red'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        fig = px.histogram(data, x='On_Time_Delivery_Rate',
                          color='Disruption_Risk',
                          title='On-Time Delivery',
                          color_discrete_map={0: 'green', 1: 'red'})
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Summary stats
    st.subheader("Summary Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### High Risk Suppliers")
        high_stats = data[data['Disruption_Risk'] == 1].describe()[
            ['Lead_Time_Days', 'On_Time_Delivery_Rate', 'Defect_Rate_Percent',
             'Financial_Stability_Score']
        ].round(2)
        st.dataframe(high_stats, use_container_width=True)
    
    with col2:
        st.markdown("### Low Risk Suppliers")
        low_stats = data[data['Disruption_Risk'] == 0].describe()[
            ['Lead_Time_Days', 'On_Time_Delivery_Rate', 'Defect_Rate_Percent',
             'Financial_Stability_Score']
        ].round(2)
        st.dataframe(low_stats, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; padding: 20px;'>
        <p>Supply Chain Disruption Prediction Platform</p>
        <p>Machine Learning • Predictive Analytics • Risk Intelligence</p>
    </div>
    """, unsafe_allow_html=True)
