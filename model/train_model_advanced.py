import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("   AI-POWERED SUPPLY CHAIN DISRUPTION PREDICTION MODEL")
print("=" * 70)
print()

# Load data
print("📊 Loading dataset...")
df = pd.read_csv("data/supply_chain_advanced.csv")
print(f"   ✓ Loaded {len(df)} records with {df.shape[1]} features")
print()

# Display class distribution
print("📈 Target Variable Distribution:")
print(f"   - High Risk (1): {(df['Disruption_Risk'] == 1).sum()} records ({(df['Disruption_Risk'] == 1).sum()/len(df)*100:.1f}%)")
print(f"   - Low Risk (0): {(df['Disruption_Risk'] == 0).sum()} records ({(df['Disruption_Risk'] == 0).sum()/len(df)*100:.1f}%)")
print()

# Feature engineering
print("🔧 Feature Engineering...")

# Select features for modeling
feature_columns = [
    'Lead_Time_Days', 'Order_Quantity', 'Stock_Level', 'Reorder_Point',
    'Safety_Stock', 'Inventory_Turnover', 'On_Time_Delivery_Rate',
    'Quality_Rating', 'Defect_Rate_Percent', 'Supplier_Responsiveness_Score',
    'Supplier_Capacity_Utilization', 'Financial_Stability_Score',
    'Shipping_Time_Days', 'Customs_Clearance_Time_Days',
    'Geopolitical_Risk_Score', 'Weather_Risk_Score', 'Demand_Volatility',
    'Supply_Volatility', 'Port_Congestion_Level', 'Supplier_Dependency_Score',
    'Alternative_Suppliers_Available', 'Historical_Disruptions_Count',
    'Average_Delay_Days', 'Contract_Compliance_Rate',
    'Communication_Response_Time_Hours', 'Payment_Terms_Days',
    'Price_Volatility_Percent', 'Customer_Impact_Score'
]

categorical_columns = [
    'Supplier_Country', 'Supplier_Region', 'Product_Category',
    'Product_Type', 'Transportation_Mode', 'Single_Source_Risk',
    'Production_Dependency_Level'
]

# Create a copy for processing
df_model = df.copy()

# Encode categorical variables
encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    df_model[col + '_Encoded'] = le.fit_transform(df_model[col])
    encoders[col] = le
    feature_columns.append(col + '_Encoded')

print(f"   ✓ Created {len(feature_columns)} features")
print()

# Prepare features and target
X = df_model[feature_columns]
y = df_model['Disruption_Risk']

# Split data
print("✂️  Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   ✓ Train set: {len(X_train)} samples")
print(f"   ✓ Test set: {len(X_test)} samples")
print()

# Scale features
print("⚖️  Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("   ✓ Features scaled")
print()

# Train multiple models
print("🤖 Training ML Models...")
print("-" * 70)

models = {
    'Random Forest': RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=150,
        learning_rate=0.1,
        max_depth=8,
        random_state=42
    ),
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        random_state=42
    )
}

results = {}
best_model = None
best_f1 = 0

for name, model in models.items():
    print(f"\n🔹 Training {name}...")
    
    # Use scaled data for LR, original for tree-based
    if name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'predictions': y_pred,
        'pred_proba': y_pred_proba
    }
    
    print(f"   Accuracy:  {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall:    {recall:.4f}")
    print(f"   F1-Score:  {f1:.4f}")
    print(f"   ROC-AUC:   {roc_auc:.4f}")
    
    if f1 > best_f1:
        best_f1 = f1
        best_model = name

print()
print("-" * 70)
print(f"\n🏆 Best Model: {best_model} (F1-Score: {best_f1:.4f})")
print()

# Save the best model
print("💾 Saving models and artifacts...")
best_model_obj = results[best_model]['model']
joblib.dump(best_model_obj, "model/supply_chain_model_advanced.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(encoders, "model/encoders.pkl")

# Save feature columns
with open("model/feature_columns.json", "w") as f:
    json.dump(feature_columns, f)

print("   ✓ Best model saved: model/supply_chain_model_advanced.pkl")
print("   ✓ Scaler saved: model/scaler.pkl")
print("   ✓ Encoders saved: model/encoders.pkl")
print()

# Feature importance (for tree-based models)
if best_model in ['Random Forest', 'Gradient Boosting']:
    print("📊 Top 15 Feature Importances:")
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': best_model_obj.feature_importances_
    }).sort_values('importance', ascending=False).head(15)
    
    for idx, row in feature_importance.iterrows():
        print(f"   {row['feature']:.<45} {row['importance']:.4f}")
    
    # Save feature importance
    feature_importance.to_csv("model/feature_importance.csv", index=False)
    print()

# Save detailed performance metrics
print("📈 Saving performance metrics...")
metrics_summary = {
    'training_date': datetime.now().isoformat(),
    'dataset_size': len(df),
    'train_size': len(X_train),
    'test_size': len(X_test),
    'num_features': len(feature_columns),
    'best_model': best_model,
    'models_performance': {
        name: {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1_score': float(metrics['f1_score']),
            'roc_auc': float(metrics['roc_auc']),
            'confusion_matrix': metrics['confusion_matrix']
        }
        for name, metrics in results.items()
    }
}

with open("model/training_metrics.json", "w") as f:
    json.dump(metrics_summary, f, indent=2)

print("   ✓ Metrics saved: model/training_metrics.json")
print()

# Detailed classification report for best model
print("📋 Classification Report (Best Model):")
print()
if best_model == 'Logistic Regression':
    print(classification_report(y_test, results[best_model]['predictions']))
else:
    print(classification_report(y_test, results[best_model]['predictions']))

print()
print("=" * 70)
print("✅ MODEL TRAINING COMPLETE!")
print("=" * 70)
