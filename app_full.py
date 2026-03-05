import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

# Page configuration
st.set_page_config(
    page_title="Supply Chain Resilience Platform",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .big-font {
        font-size:40px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<p class="big-font">🏭 AI-Powered Supply Chain Resilience Platform</p>', unsafe_allow_html=True)
st.markdown("**Enterprise-Grade Supply Chain Disruption Prediction & Intelligence System**")
st.markdown("---")

# Load model and artifacts
@st.cache_resource
def load_model_artifacts():
    model = joblib.load("model/supply_chain_model_advanced.pkl")
    scaler = joblib.load("model/scaler.pkl")
    encoders = joblib.load("model/encoders.pkl")
    
    with open("model/feature_columns.json", "r") as f:
        feature_columns = json.load(f)
    
    with open("model/training_metrics.json", "r") as f:
        metrics = json.load(f)
    
    return model, scaler, encoders, feature_columns, metrics

@st.cache_data
def load_data():
    return pd.read_csv("data/supply_chain_advanced.csv")

try:
    model, scaler, encoders, feature_columns, training_metrics = load_model_artifacts()
    df = load_data()
except Exception as e:
    st.error(f"Error loading model artifacts: {str(e)}")
    st.stop()

# Sidebar Navigation
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Select View",
    ["🏠 Dashboard", "👤 User: Make Predictions", "👨‍💼 Admin: Model Performance", "📈 Analytics & Insights"]
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**Model**: {training_metrics['best_model']}")
st.sidebar.markdown(f"**Training Date**: {training_metrics['training_date'][:10]}")
st.sidebar.markdown(f"**Dataset Size**: {training_metrics['dataset_size']} records")
st.sidebar.markdown(f"**Model Accuracy**: {training_metrics['models_performance'][training_metrics['best_model']]['accuracy']:.2%}")

# ====================================================================
# PAGE 1: DASHBOARD
# ====================================================================
if page == "🏠 Dashboard":
    st.header("📊 Executive Dashboard")
    st.markdown("Real-time supply chain risk monitoring and key performance indicators")
    
    # Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    high_risk_count = (df['Disruption_Risk'] == 1).sum()
    low_risk_count = (df['Disruption_Risk'] == 0).sum()
    avg_risk_prob = df['Disruption_Probability'].mean()
    critical_suppliers = len(df[df['Production_Dependency_Level'] == 'Critical'])
    
    with col1:
        st.metric("🚨 High Risk Suppliers", high_risk_count, 
                  f"{(high_risk_count/len(df)*100):.1f}%")
    
    with col2:
        st.metric("✅ Low Risk Suppliers", low_risk_count,
                  f"{(low_risk_count/len(df)*100):.1f}%")
    
    with col3:
        st.metric("📊 Avg Disruption Probability", f"{avg_risk_prob:.1f}%")
    
    with col4:
        st.metric("⚠️ Critical Dependencies", critical_suppliers)
    
    st.markdown("---")
    
    # Risk Distribution Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk Distribution by Region")
        risk_by_region = df.groupby('Supplier_Region')['Disruption_Risk'].agg(['sum', 'count'])
        risk_by_region['percentage'] = (risk_by_region['sum'] / risk_by_region['count'] * 100).round(1)
        
        fig = px.bar(risk_by_region.reset_index(), 
                     x='Supplier_Region', 
                     y='percentage',
                     title="High Risk Percentage by Region",
                     labels={'percentage': 'High Risk %', 'Supplier_Region': 'Region'},
                     color='percentage',
                     color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Risk Distribution by Product Category")
        risk_by_product = df.groupby('Product_Category')['Disruption_Risk'].agg(['sum', 'count'])
        risk_by_product['percentage'] = (risk_by_product['sum'] / risk_by_product['count'] * 100).round(1)
        
        fig = px.bar(risk_by_product.reset_index(), 
                     x='Product_Category', 
                     y='percentage',
                     title="High Risk Percentage by Product",
                     labels={'percentage': 'High Risk %', 'Product_Category': 'Category'},
                     color='percentage',
                     color_continuous_scale='Oranges')
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Top Risk Suppliers
    st.subheader("🔴 Top 10 High-Risk Suppliers")
    high_risk_suppliers = df[df['Disruption_Risk'] == 1].nlargest(10, 'Disruption_Probability')[
        ['Supplier_Name', 'Supplier_Country', 'Product_Category', 
         'Disruption_Probability', 'On_Time_Delivery_Rate', 'Financial_Stability_Score']
    ]
    high_risk_suppliers['Disruption_Probability'] = high_risk_suppliers['Disruption_Probability'].round(2)
    st.dataframe(high_risk_suppliers, use_container_width=True, hide_index=True)
    
    # Geographic Risk Map
    st.markdown("---")
    st.subheader("🗺️ Geographic Risk Distribution")
    country_risk = df.groupby('Supplier_Country').agg({
        'Disruption_Risk': 'sum',
        'Record_ID': 'count'
    }).reset_index()
    country_risk.columns = ['Country', 'High_Risk_Count', 'Total_Count']
    country_risk['Risk_Percentage'] = (country_risk['High_Risk_Count'] / country_risk['Total_Count'] * 100).round(1)
    
    fig = px.scatter_geo(country_risk,
                         locations='Country',
                         locationmode='country names',
                         size='Total_Count',
                         color='Risk_Percentage',
                         hover_name='Country',
                         hover_data={'High_Risk_Count': True, 'Total_Count': True, 'Risk_Percentage': ':.1f'},
                         title='Global Supply Chain Risk Map',
                         color_continuous_scale='RdYlGn_r')
    st.plotly_chart(fig, use_container_width=True)

# ====================================================================
# PAGE 2: USER PREDICTION INTERFACE
# ====================================================================
elif page == "👤 User: Make Predictions":
    st.header("🔮 Supply Chain Disruption Prediction")
    st.markdown("Enter supplier and operational data to predict disruption risk")
    
    tab1, tab2 = st.tabs(["📝 Manual Input", "📊 Batch Prediction"])
    
    with tab1:
        st.subheader("Enter Supplier Information")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Geographic Information**")
            supplier_country = st.selectbox("Supplier Country", df['Supplier_Country'].unique())
            supplier_region = st.selectbox("Supplier Region", df['Supplier_Region'].unique())
            product_category = st.selectbox("Product Category", df['Product_Category'].unique())
            product_type = st.selectbox("Product Type", df['Product_Type'].unique())
            
        with col2:
            st.markdown("**Operational Metrics**")
            lead_time = st.slider("Lead Time (Days)", 5, 60, 25)
            stock_level = st.slider("Stock Level", 0, 5000, 1500)
            reorder_point = st.slider("Reorder Point", 200, 2000, 800)
            safety_stock = st.slider("Safety Stock", 100, 1500, 500)
            order_quantity = st.slider("Order Quantity", 100, 10000, 2000)
            
        with col3:
            st.markdown("**Performance Metrics**")
            on_time_delivery = st.slider("On-Time Delivery Rate (%)", 65.0, 100.0, 85.0)
            quality_rating = st.slider("Quality Rating (%)", 65.0, 100.0, 85.0)
            defect_rate = st.slider("Defect Rate (%)", 0.0, 8.0, 2.0)
            financial_stability = st.slider("Financial Stability Score", 40.0, 100.0, 75.0)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Risk Factors**")
            geopolitical_risk = st.slider("Geopolitical Risk Score", 0.0, 10.0, 3.0)
            weather_risk = st.slider("Weather Risk Score", 0.0, 10.0, 3.0)
            demand_volatility = st.slider("Demand Volatility", 0.0, 10.0, 4.0)
            supply_volatility = st.slider("Supply Volatility", 0.0, 10.0, 4.0)
            
        with col2:
            st.markdown("**Logistics**")
            transportation_mode = st.selectbox("Transportation Mode", df['Transportation_Mode'].unique())
            shipping_time = st.slider("Shipping Time (Days)", 1, 45, 15)
            customs_clearance = st.slider("Customs Clearance (Days)", 0, 10, 3)
            port_congestion = st.slider("Port Congestion Level", 0.0, 10.0, 4.0)
            
        with col3:
            st.markdown("**Dependency Metrics**")
            single_source_risk = st.selectbox("Single Source Risk", ['Yes', 'No'])
            alternative_suppliers = st.slider("Alternative Suppliers Available", 0, 5, 2)
            production_dependency = st.selectbox("Production Dependency Level", 
                                                 ['Critical', 'High', 'Medium', 'Low'])
            historical_disruptions = st.slider("Historical Disruptions Count", 0, 15, 3)
        
        # Additional metrics
        col1, col2 = st.columns(2)
        with col1:
            inventory_turnover = st.slider("Inventory Turnover", 2.0, 12.0, 6.0, 0.1)
            supplier_responsiveness = st.slider("Supplier Responsiveness Score", 60.0, 100.0, 80.0)
            supplier_capacity = st.slider("Supplier Capacity Utilization (%)", 50.0, 100.0, 75.0)
            
        with col2:
            avg_delay_days = st.slider("Average Delay (Days)", 0.0, 20.0, 5.0, 0.5)
            contract_compliance = st.slider("Contract Compliance Rate (%)", 70.0, 100.0, 90.0)
            communication_response_time = st.slider("Communication Response Time (Hours)", 1.0, 72.0, 24.0)
        
        col1, col2 = st.columns(2)
        with col1:
            payment_terms = st.selectbox("Payment Terms (Days)", [30, 45, 60, 90])
            price_volatility = st.slider("Price Volatility (%)", 0.0, 25.0, 5.0)
            
        with col2:
            supplier_dependency_score = st.slider("Supplier Dependency Score", 0.0, 10.0, 5.0)
            customer_impact_score = st.slider("Customer Impact Score", 0.0, 10.0, 5.0)
        
        if st.button("🔮 Predict Disruption Risk", type="primary"):
            # Prepare input data
            input_data = {
                'Lead_Time_Days': lead_time,
                'Order_Quantity': order_quantity,
                'Stock_Level': stock_level,
                'Reorder_Point': reorder_point,
                'Safety_Stock': safety_stock,
                'Inventory_Turnover': inventory_turnover,
                'On_Time_Delivery_Rate': on_time_delivery,
                'Quality_Rating': quality_rating,
                'Defect_Rate_Percent': defect_rate,
                'Supplier_Responsiveness_Score': supplier_responsiveness,
                'Supplier_Capacity_Utilization': supplier_capacity,
                'Financial_Stability_Score': financial_stability,
                'Shipping_Time_Days': shipping_time,
                'Customs_Clearance_Time_Days': customs_clearance,
                'Geopolitical_Risk_Score': geopolitical_risk,
                'Weather_Risk_Score': weather_risk,
                'Demand_Volatility': demand_volatility,
                'Supply_Volatility': supply_volatility,
                'Port_Congestion_Level': port_congestion,
                'Supplier_Dependency_Score': supplier_dependency_score,
                'Alternative_Suppliers_Available': alternative_suppliers,
                'Historical_Disruptions_Count': historical_disruptions,
                'Average_Delay_Days': avg_delay_days,
                'Contract_Compliance_Rate': contract_compliance,
                'Communication_Response_Time_Hours': communication_response_time,
                'Payment_Terms_Days': payment_terms,
                'Price_Volatility_Percent': price_volatility,
                'Customer_Impact_Score': customer_impact_score,
                'Supplier_Country_Encoded': encoders['Supplier_Country'].transform([supplier_country])[0],
                'Supplier_Region_Encoded': encoders['Supplier_Region'].transform([supplier_region])[0],
                'Product_Category_Encoded': encoders['Product_Category'].transform([product_category])[0],
                'Product_Type_Encoded': encoders['Product_Type'].transform([product_type])[0],
                'Transportation_Mode_Encoded': encoders['Transportation_Mode'].transform([transportation_mode])[0],
                'Single_Source_Risk_Encoded': encoders['Single_Source_Risk'].transform([single_source_risk])[0],
                'Production_Dependency_Level_Encoded': encoders['Production_Dependency_Level'].transform([production_dependency])[0]
            }
            
            # Create DataFrame with correct feature order
            input_df = pd.DataFrame([input_data])[feature_columns]
            
            # Scale input
            input_scaled = scaler.transform(input_df)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0][1]
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction == 1:
                    st.error("### 🚨 HIGH RISK")
                    st.markdown("**Disruption Likely**")
                else:
                    st.success("### ✅ LOW RISK")
                    st.markdown("**No Immediate Threat**")
            
            with col2:
                st.metric("Disruption Probability", f"{prediction_proba*100:.1f}%")
                
            with col3:
                risk_level = "Critical" if prediction_proba > 0.8 else "High" if prediction_proba > 0.6 else "Medium" if prediction_proba > 0.4 else "Low"
                st.metric("Risk Level", risk_level)
            
            # Recommendations
            st.markdown("---")
            st.subheader("💡 Recommended Actions")
            
            if prediction == 1:
                recommendations = []
                if stock_level < reorder_point:
                    recommendations.append("🔴 **Critical**: Stock level below reorder point - Increase inventory immediately")
                if on_time_delivery < 85:
                    recommendations.append("🟠 **Important**: Poor delivery performance - Consider alternative suppliers")
                if financial_stability < 70:
                    recommendations.append("🟠 **Monitor**: Low financial stability - Review supplier financial health")
                if single_source_risk == 'Yes':
                    recommendations.append("🟡 **Suggest**: Single source dependency - Identify backup suppliers")
                if geopolitical_risk > 7:
                    recommendations.append("🔴 **Urgent**: High geopolitical risk - Diversify geographic sourcing")
                if lead_time > 40:
                    recommendations.append("🟡 **Review**: Extended lead times - Explore faster logistics options")
                
                if not recommendations:
                    recommendations.append("👁️ Monitor supplier closely and prepare contingency plans")
                
                for rec in recommendations:
                    st.markdown(rec)
            else:
                st.success("✅ No immediate action required. Continue routine monitoring.")
    
    with tab2:
        st.subheader("📊 Batch Prediction from Dataset")
        st.markdown("Upload a CSV file or use the existing dataset for batch predictions")
        
        if st.button("Run Batch Prediction on Current Dataset"):
            with st.spinner("Processing predictions..."):
                # Prepare features
                df_pred = df.copy()
                
                # Encode categorical variables
                for col in ['Supplier_Country', 'Supplier_Region', 'Product_Category',
                           'Product_Type', 'Transportation_Mode', 'Single_Source_Risk',
                           'Production_Dependency_Level']:
                    df_pred[col + '_Encoded'] = encoders[col].transform(df_pred[col])
                
                X_pred = df_pred[feature_columns]
                X_pred_scaled = scaler.transform(X_pred)
                
                predictions = model.predict(X_pred_scaled)
                predictions_proba = model.predict_proba(X_pred_scaled)[:, 1]
                
                df_pred['Predicted_Risk'] = predictions
                df_pred['Predicted_Probability'] = (predictions_proba * 100).round(2)
                
                # Display results summary
                st.success("✅ Batch prediction complete!")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Records", len(df_pred))
                with col2:
                    st.metric("High Risk Predicted", (predictions == 1).sum())
                with col3:
                    st.metric("Low Risk Predicted", (predictions == 0).sum())
                
                # Show results table
                st.markdown("### Prediction Results")
                result_df = df_pred[['Supplier_Name', 'Supplier_Country', 'Product_Category',
                                     'Predicted_Risk', 'Predicted_Probability', 
                                     'On_Time_Delivery_Rate', 'Financial_Stability_Score']].copy()
                
                # Add risk label
                result_df['Risk_Label'] = result_df['Predicted_Risk'].map({0: '✅ Low Risk', 1: '🚨 High Risk'})
                
                # Filter options
                risk_filter = st.multiselect("Filter by Risk Level", 
                                            options=['✅ Low Risk', '🚨 High Risk'],
                                            default=['✅ Low Risk', '🚨 High Risk'])
                
                filtered_df = result_df[result_df['Risk_Label'].isin(risk_filter)]
                st.dataframe(filtered_df, use_container_width=True, hide_index=True)
                
                # Download button
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions as CSV",
                    data=csv,
                    file_name=f"supply_chain_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )

# ====================================================================
# PAGE 3: ADMIN - MODEL PERFORMANCE
# ====================================================================
elif page == "👨‍💼 Admin: Model Performance":
    st.header("🎯 Model Performance Dashboard")
    st.markdown("Comprehensive model evaluation and performance metrics")
    
    # Model Information
    st.subheader("🤖 Model Information")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Best Model", training_metrics['best_model'])
    with col2:
        st.metric("Training Date", training_metrics['training_date'][:10])
    with col3:
        st.metric("Dataset Size", f"{training_metrics['dataset_size']} records")
    with col4:
        st.metric("Features Used", training_metrics['num_features'])
    
    st.markdown("---")
    
    # Model Comparison
    st.subheader("📊 Model Comparison")
    
    models_df = pd.DataFrame(training_metrics['models_performance']).T
    models_df = models_df[['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']]
    models_df.columns = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    models_df = (models_df * 100).round(2)
    
    st.dataframe(models_df, use_container_width=True)
    
    # Visualize model comparison
    fig = go.Figure()
    
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    
    for metric in metrics_to_plot:
        fig.add_trace(go.Bar(
            name=metric,
            x=models_df.index,
            y=models_df[metric],
            text=models_df[metric].astype(float).round(1),
            textposition='auto'
        ))
    
    fig.update_layout(
        title="Model Performance Comparison",
        xaxis_title="Model",
        yaxis_title="Score (%)",
        barmode='group',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Confusion Matrix
    st.subheader("🎯 Confusion Matrix")
    
    best_model_name = training_metrics['best_model']
    conf_matrix = np.array(training_metrics['models_performance'][best_model_name]['confusion_matrix'])
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Performance Metrics")
        metrics_data = training_metrics['models_performance'][best_model_name]
        st.metric("Accuracy", f"{metrics_data['accuracy']*100:.2f}%")
        st.metric("Precision", f"{metrics_data['precision']*100:.2f}%")
        st.metric("Recall", f"{metrics_data['recall']*100:.2f}%")
        st.metric("F1-Score", f"{metrics_data['f1_score']*100:.2f}%")
        st.metric("ROC-AUC", f"{metrics_data['roc_auc']*100:.2f}%")
    
    with col2:
        fig = go.Figure(data=go.Heatmap(
            z=conf_matrix,
            x=['Predicted Low', 'Predicted High'],
            y=['Actual Low', 'Actual High'],
            text=conf_matrix,
            texttemplate='%{text}',
            textfont={"size": 20},
            colorscale='Blues'
        ))
        
        fig.update_layout(
            title=f"Confusion Matrix - {best_model_name}",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Feature Importance
    st.subheader("📈 Feature Importance Analysis")
    
    try:
        feature_importance_df = pd.read_csv("model/feature_importance.csv")
        
        fig = px.bar(feature_importance_df.head(15), 
                     x='importance', 
                     y='feature',
                     orientation='h',
                     title='Top 15 Most Important Features',
                     labels={'importance': 'Importance Score', 'feature': 'Feature'},
                     color='importance',
                     color_continuous_scale='Viridis')
        
        fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("📊 View Full Feature Importance Table"):
            st.dataframe(feature_importance_df, use_container_width=True, hide_index=True)
            
    except FileNotFoundError:
        st.info("Feature importance analysis not available for Logistic Regression model.")
    
    st.markdown("---")
    
    # Model Diagnostics
    st.subheader("🔬 Model Diagnostics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Training Configuration")
        st.json({
            "Train Size": training_metrics['train_size'],
            "Test Size": training_metrics['test_size'],
            "Features": training_metrics['num_features'],
            "Best Model": training_metrics['best_model']
        })
    
    with col2:
        st.markdown("### Class Distribution")
        class_dist = df['Disruption_Risk'].value_counts()
        fig = px.pie(values=class_dist.values, 
                     names=['High Risk', 'Low Risk'],
                     title='Target Variable Distribution',
                     color_discrete_map={'High Risk': 'red', 'Low Risk': 'green'})
        st.plotly_chart(fig, use_container_width=True)

# ====================================================================
# PAGE 4: ANALYTICS & INSIGHTS
# ====================================================================
elif page == "📈 Analytics & Insights":
    st.header("📈 Advanced Analytics & Insights")
    st.markdown("Deep-dive analysis of supply chain risks and patterns")
    
    # Risk Trends
    st.subheader("📅 Risk Trends Over Time")
    df['Date'] = pd.to_datetime(df['Date'])
    df_sorted = df.sort_values('Date')
    
    monthly_risk = df_sorted.groupby(df_sorted['Date'].dt.to_period('M')).agg({
        'Disruption_Risk': 'mean',
        'Disruption_Probability': 'mean'
    }).reset_index()
    monthly_risk['Date'] = monthly_risk['Date'].astype(str)
    monthly_risk['Disruption_Risk'] = (monthly_risk['Disruption_Risk'] * 100).round(2)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly_risk['Date'], y=monthly_risk['Disruption_Risk'],
                            mode='lines+markers', name='Risk Rate (%)',
                            line=dict(color='red', width=3)))
    fig.add_trace(go.Scatter(x=monthly_risk['Date'], y=monthly_risk['Disruption_Probability'],
                            mode='lines+markers', name='Avg Probability (%)',
                            line=dict(color='orange', width=3)))
    
    fig.update_layout(title='Monthly Risk Trends', 
                     xaxis_title='Month', 
                     yaxis_title='Percentage (%)',
                     height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Correlation Analysis
    st.subheader("🔗 Risk Factor Correlation Analysis")
    
    correlation_features = [
        'Lead_Time_Days', 'Stock_Level', 'On_Time_Delivery_Rate',
        'Defect_Rate_Percent', 'Financial_Stability_Score',
        'Geopolitical_Risk_Score', 'Supply_Volatility',
        'Historical_Disruptions_Count', 'Disruption_Risk'
    ]
    
    corr_matrix = df[correlation_features].corr()
    
    fig = px.imshow(corr_matrix,
                    labels=dict(color="Correlation"),
                    x=corr_matrix.columns,
                    y=corr_matrix.columns,
                    color_continuous_scale='RdBu_r',
                    aspect="auto",
                    title='Feature Correlation Heatmap')
    
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Supplier Analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏢 Top Suppliers by Volume")
        supplier_counts = df['Supplier_Name'].value_counts().head(10)
        fig = px.bar(x=supplier_counts.index, y=supplier_counts.values,
                     labels={'x': 'Supplier', 'y': 'Number of Records'},
                     title='Top 10 Suppliers')
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🚚 Transportation Mode Usage")
        transport_counts = df['Transportation_Mode'].value_counts()
        fig = px.pie(values=transport_counts.values, 
                     names=transport_counts.index,
                     title='Transportation Mode Distribution')
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Risk Factors Analysis
    st.subheader("⚠️ Key Risk Factors Distribution")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig = px.histogram(df, x='Geopolitical_Risk_Score', 
                          color='Disruption_Risk',
                          title='Geopolitical Risk Distribution',
                          labels={'Disruption_Risk': 'Risk Level'},
                          color_discrete_map={0: 'green', 1: 'red'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = px.histogram(df, x='Financial_Stability_Score',
                          color='Disruption_Risk',
                          title='Financial Stability Distribution',
                          labels={'Disruption_Risk': 'Risk Level'},
                          color_discrete_map={0: 'green', 1: 'red'})
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        fig = px.histogram(df, x='On_Time_Delivery_Rate',
                          color='Disruption_Risk',
                          title='On-Time Delivery Distribution',
                          labels={'Disruption_Risk': 'Risk Level'},
                          color_discrete_map={0: 'green', 1: 'red'})
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Summary Statistics
    st.subheader("📊 Summary Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### High Risk Suppliers")
        high_risk_stats = df[df['Disruption_Risk'] == 1].describe()[
            ['Lead_Time_Days', 'On_Time_Delivery_Rate', 'Defect_Rate_Percent',
             'Financial_Stability_Score']
        ].round(2)
        st.dataframe(high_risk_stats, use_container_width=True)
    
    with col2:
        st.markdown("### Low Risk Suppliers")
        low_risk_stats = df[df['Disruption_Risk'] == 0].describe()[
            ['Lead_Time_Days', 'On_Time_Delivery_Rate', 'Defect_Rate_Percent',
             'Financial_Stability_Score']
        ].round(2)
        st.dataframe(low_risk_stats, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        <p>🏭 AI-Powered Supply Chain Resilience Platform | Enterprise Edition</p>
        <p>Powered by Machine Learning • Real-time Risk Intelligence • Predictive Analytics</p>
    </div>
    """, unsafe_allow_html=True)
