import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json

# Page config
st.set_page_config(
    page_title="Supply Chain Disruption Predictor",
    page_icon="🚨",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main-title {
        font-size: 40px;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 10px;
    }
    .subtitle {
        font-size: 18px;
        color: #7f8c8d;
        margin-bottom: 30px;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<p class="main-title">🚨 Supply Chain Disruption Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict disruption risks from supply chain events</p>', unsafe_allow_html=True)
st.markdown("---")

# Load model and data
@st.cache_resource
def load_model_artifacts():
    try:
        model = joblib.load("model/disruption_model.pkl")
        scaler = joblib.load("model/scaler.pkl")
        encoders = joblib.load("model/encoders.pkl")
        features = joblib.load("model/features.pkl")
        
        with open("model/metrics.json", "r") as f:
            metrics = json.load(f)
        
        return model, scaler, encoders, features, metrics
    except:
        return None, None, None, None, None

@st.cache_data
def load_dataset():
    try:
        return pd.read_csv("data/supply_chain_events.csv")
    except:
        return None

model, scaler, encoders, features, metrics = load_model_artifacts()
data = load_dataset()

if model is None or data is None:
    st.error("⚠️ Please run the setup first:")
    st.code("python generate_data.py\npython train_model.py")
    st.stop()

# Sidebar
st.sidebar.title("📊 Navigation")
page = st.sidebar.radio(
    "Select View:",
    ["🏠 Dashboard", "🔮 Make Prediction", "📈 Model Performance", "📊 Analytics"]
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
**Model Info:**
- Type: {metrics['best_model']}
- Accuracy: {metrics['models_performance'][metrics['best_model']]['accuracy']:.1%}
- Date: {metrics['training_date'][:10]}
- Events: {metrics['dataset_size']}
""")

# ==================================================
# DASHBOARD
# ==================================================
if page == "🏠 Dashboard":
    st.header("📊 Dashboard Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    total_events = len(data)
    disruptions = data['disruption'].sum()
    non_disruptions = total_events - disruptions
    avg_impact = data['financial_impact'].mean()
    
    col1.metric("Total Events", total_events)
    col2.metric("Disruptions", disruptions, f"{disruptions/total_events*100:.0f}%")
    col3.metric("Non-Disruptions", non_disruptions)
    col4.metric("Avg Financial Impact", f"${avg_impact:,.0f}")
    
    st.markdown("---")
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Events by Severity")
        severity_counts = data['severity_level'].value_counts()
        fig = px.pie(values=severity_counts.values, 
                     names=severity_counts.index,
                     title="Severity Distribution",
                     color_discrete_sequence=px.colors.sequential.Reds_r)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Events by Type")
        type_counts = data['event_type'].value_counts().head(8)
        fig = px.bar(x=type_counts.index, y=type_counts.values,
                     title="Top Event Types",
                     labels={'x': 'Event Type', 'y': 'Count'},
                     color=type_counts.values,
                     color_continuous_scale='Blues')
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Top disruptions
    st.subheader("🔴 Top 10 Costly Disruptions")
    top_disruptions = data[data['disruption'] == 1].nlargest(10, 'financial_impact')
    display_cols = ['event_id', 'event_date', 'event_type', 'severity_level', 
                    'country', 'city', 'financial_impact']
    st.dataframe(top_disruptions[display_cols], use_container_width=True, hide_index=True)
    
    # Geographic distribution
    st.markdown("---")
    st.subheader("🗺️ Geographic Distribution")
    
    country_stats = data.groupby('country').agg({
        'disruption': 'sum',
        'event_id': 'count',
        'financial_impact': 'sum'
    }).reset_index()
    country_stats.columns = ['Country', 'Disruptions', 'Total_Events', 'Total_Impact']
    country_stats['Disruption_Rate'] = (country_stats['Disruptions'] / country_stats['Total_Events'] * 100).round(1)
    
    fig = px.scatter_geo(country_stats,
                         locations='Country',
                         locationmode='country names',
                         size='Total_Events',
                         color='Disruption_Rate',
                         hover_name='Country',
                         hover_data={'Total_Events': True, 'Disruptions': True, 'Disruption_Rate': ':.1f'},
                         title='Disruption Rate by Country',
                         color_continuous_scale='RdYlGn_r')
    st.plotly_chart(fig, use_container_width=True)

# ==================================================
# MAKE PREDICTION
# ==================================================
elif page == "🔮 Make Prediction":
    st.header("🔮 Predict Disruption Risk")
    st.write("Enter event details to predict if it will cause a major disruption")
    
    tab1, tab2 = st.tabs(["➕ Single Prediction", "📊 Batch Analysis"])
    
    with tab1:
        st.subheader("Event Details")
        
        # Simple, clean input form
        col1, col2 = st.columns(2)
        
        with col1:
            event_type = st.selectbox(
                "Event Type",
                options=sorted(data['event_type'].unique()),
                help="Type of supply chain event"
            )
            
            severity = st.selectbox(
                "Severity Level",
                options=['Low', 'Medium', 'High', 'Critical'],
                index=1,
                help="How severe is this event?"
            )
            
            cause = st.selectbox(
                "Root Cause",
                options=sorted(data['cause'].unique()),
                help="What caused this event?"
            )
        
        with col2:
            country = st.selectbox(
                "Country",
                options=sorted(data['country'].unique()),
                help="Where did this occur?"
            )
            
            financial_impact = st.number_input(
                "Financial Impact (USD)",
                min_value=1000,
                max_value=500000,
                value=50000,
                step=5000,
                help="Estimated financial impact in dollars"
            )
        
        st.markdown("---")
        
        if st.button("🔮 Predict Risk", type="primary", use_container_width=True):
            # Prepare input
            input_data = {
                'event_type_encoded': encoders['event_type'].transform([event_type])[0],
                'severity_level_encoded': encoders['severity_level'].transform([severity])[0],
                'cause_encoded': encoders['cause'].transform([cause])[0],
                'country_encoded': encoders['country'].transform([country])[0],
                'financial_impact': financial_impact
            }
            
            input_df = pd.DataFrame([input_data])[features]
            input_scaled = scaler.transform(input_df)
            
            # Make prediction
            prediction = model.predict(input_scaled)[0]
            probability = model.predict_proba(input_scaled)[0][1]
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction == 1:
                    st.error("### 🚨 DISRUPTION")
                    st.markdown("**Major disruption likely**")
                else:
                    st.success("### ✅ MANAGEABLE")
                    st.markdown("**Event is manageable**")
            
            with col2:
                st.metric("Disruption Probability", f"{probability*100:.1f}%")
            
            with col3:
                risk_level = "Very High" if probability > 0.8 else "High" if probability > 0.6 else "Medium" if probability > 0.4 else "Low"
                color = "🔴" if probability > 0.7 else "🟡" if probability > 0.4 else "🟢"
                st.metric("Risk Level", f"{color} {risk_level}")
            
            # Recommendations
            st.markdown("---")
            st.subheader("💡 Recommendations")
            
            if prediction == 1:
                st.markdown("#### ⚠️ Action Required:")
                
                if severity in ['High', 'Critical']:
                    st.markdown("- 🔴 **Immediate escalation** - This is a severe event requiring senior management attention")
                
                if financial_impact > 100000:
                    st.markdown("- 💰 **Financial mitigation** - Consider insurance claims and budget reallocation")
                
                if event_type in ['Natural Disaster', 'Labor Strike', 'Port Congestion']:
                    st.markdown("- 🔄 **Alternative routing** - Identify backup suppliers and routes immediately")
                
                if cause in ['Geopolitical', 'Natural Disaster']:
                    st.markdown("- 📢 **Stakeholder communication** - Notify customers and partners proactively")
                
                st.markdown("- 📋 **Documentation** - Record all mitigation actions for future reference")
                
            else:
                st.success("✅ Monitor the situation but no immediate action required")
                st.markdown("- Continue regular monitoring schedule")
                st.markdown("- Document event for trend analysis")
    
    with tab2:
        st.subheader("📊 Batch Event Analysis")
        st.write("Analyze all historical events")
        
        if st.button("Run Batch Analysis"):
            with st.spinner("Analyzing events..."):
                # Prepare data
                df_batch = data.copy()
                
                X_batch = df_batch[[col for col in df_batch.columns if col + '_encoded' in features or col in features]]
                
                # Encode if needed
                for col in ['event_type', 'severity_level', 'cause', 'country']:
                    if col + '_encoded' not in df_batch.columns:
                        df_batch[col + '_encoded'] = encoders[col].transform(df_batch[col])
                
                X_features = df_batch[features]
                X_scaled = scaler.transform(X_features)
                
                predictions = model.predict(X_scaled)
                probabilities = model.predict_proba(X_scaled)[:, 1]
                
                df_batch['predicted'] = predictions
                df_batch['probability'] = (probabilities * 100).round(1)
                
                # Results
                st.success("✅ Analysis complete!")
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Total Events", len(df_batch))
                col2.metric("Predicted Disruptions", (predictions == 1).sum())
                col3.metric("Model Accuracy", f"{metrics['models_performance'][metrics['best_model']]['accuracy']:.1%}")
                
                st.markdown("### Results")
                
                risk_filter = st.multiselect(
                    "Filter by Prediction:",
                    options=['Disruption', 'Manageable'],
                    default=['Disruption', 'Manageable']
                )
                
                filter_map = {'Disruption': 1, 'Manageable': 0}
                filter_values = [filter_map[f] for f in risk_filter]
                
                filtered = df_batch[df_batch['predicted'].isin(filter_values)]
                
                display_cols = ['event_id', 'event_date', 'event_type', 'severity_level', 
                               'country', 'financial_impact', 'probability', 'predicted']
                
                display_df = filtered[display_cols].copy()
                display_df['predicted'] = display_df['predicted'].map({0: '✅ Manageable', 1: '🚨 Disruption'})
                
                st.dataframe(display_df, use_container_width=True, hide_index=True)
                
                # Download
                csv = display_df.to_csv(index=False)
                st.download_button(
                    "📥 Download Results",
                    csv,
                    f"predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                    "text/csv"
                )

# ==================================================
# MODEL PERFORMANCE
# ==================================================
elif page == "📈 Model Performance":
    st.header("📈 Model Performance Metrics")
    
    st.subheader("Model Information")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model Type", metrics['best_model'])
    col2.metric("Training Date", metrics['training_date'][:10])
    col3.metric("Dataset Size", metrics['dataset_size'])
    col4.metric("Features", metrics['num_features'])
    
    st.markdown("---")
    
    # Model comparison
    st.subheader("📊 Model Comparison")
    
    models_df = pd.DataFrame(metrics['models_performance']).T
    models_df = models_df[['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']]
    models_df.columns = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    models_df = (models_df * 100).round(2)
    
    st.dataframe(models_df, use_container_width=True)
    
    # Bar chart
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
        title="Model Performance Comparison",
        xaxis_title="Model",
        yaxis_title="Score (%)",
        barmode='group',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Confusion matrix
    st.subheader("🎯 Confusion Matrix")
    
    best = metrics['best_model']
    conf_mat = np.array(metrics['models_performance'][best]['confusion_matrix'])
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Key Metrics")
        perf = metrics['models_performance'][best]
        st.metric("Accuracy", f"{perf['accuracy']*100:.1f}%")
        st.metric("Precision", f"{perf['precision']*100:.1f}%")
        st.metric("Recall", f"{perf['recall']*100:.1f}%")
        st.metric("F1-Score", f"{perf['f1_score']*100:.1f}%")
    
    with col2:
        fig = go.Figure(data=go.Heatmap(
            z=conf_mat,
            x=['Predicted: Manageable', 'Predicted: Disruption'],
            y=['Actual: Manageable', 'Actual: Disruption'],
            text=conf_mat,
            texttemplate='%{text}',
            textfont={"size": 18},
            colorscale='Blues'
        ))
        
        fig.update_layout(
            title=f"Confusion Matrix - {best}",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

# ==================================================
# ANALYTICS
# ==================================================
elif page == "📊 Analytics":
    st.header("📊 Data Analytics")
    
    # Event trends
    st.subheader("📅 Event Trends Over Time")
    
    data['event_date'] = pd.to_datetime(data['event_date'])
    monthly = data.groupby(data['event_date'].dt.to_period('M')).agg({
        'disruption': ['sum', 'count']
    }).reset_index()
    monthly.columns = ['Month', 'Disruptions', 'Total']
    monthly['Month'] = monthly['Month'].astype(str)
    monthly['Rate'] = (monthly['Disruptions'] / monthly['Total'] * 100).round(1)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=monthly['Month'], y=monthly['Total'],
                            mode='lines+markers', name='Total Events',
                            line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=monthly['Month'], y=monthly['Disruptions'],
                            mode='lines+markers', name='Disruptions',
                            line=dict(color='red', width=2)))
    
    fig.update_layout(title='Monthly Event Trends', 
                     xaxis_title='Month', 
                     yaxis_title='Count',
                     height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Risk analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Risk by Country")
        country_risk = data.groupby('country')['disruption'].mean().sort_values(ascending=False).head(10)
        fig = px.bar(x=country_risk.index, y=country_risk.values*100,
                     labels={'x': 'Country', 'y': 'Disruption Rate (%)'},
                     title='Top 10 Countries by Disruption Rate',
                     color=country_risk.values,
                     color_continuous_scale='Reds')
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Financial Impact by Cause")
        cause_impact = data.groupby('cause')['financial_impact'].sum().sort_values(ascending=False).head(10)
        fig = px.bar(x=cause_impact.index, y=cause_impact.values,
                     labels={'x': 'Cause', 'y': 'Total Impact (USD)'},
                     title='Top 10 Causes by Financial Impact',
                     color=cause_impact.values,
                     color_continuous_scale='Blues')
        fig.update_xaxes(tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # Summary stats
    st.subheader("📊 Summary Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Disruption Events")
        disruption_stats = data[data['disruption'] == 1].describe()[['financial_impact']].round(0)
        st.dataframe(disruption_stats, use_container_width=True)
    
    with col2:
        st.markdown("### Manageable Events")
        normal_stats = data[data['disruption'] == 0].describe()[['financial_impact']].round(0)
        st.dataframe(normal_stats, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray; padding: 15px;'>
        <p>Supply Chain Disruption Prediction Platform</p>
        <p>AI-Powered Risk Intelligence System</p>
    </div>
    """, unsafe_allow_html=True)
