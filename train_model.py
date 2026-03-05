# Training script for supply chain disruption prediction
# Author: Supply Chain Analytics Team
# Last updated: March 2026

import pandas as pd
import numpy as np
import joblib
import json
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print(" "*15 + "SUPPLY CHAIN DISRUPTION MODEL")
print("="*70)
print()

# Load the dataset
print("Loading data...")
df = pd.read_csv("data/supply_chain_advanced.csv")
print(f"Loaded {len(df)} records with {df.shape[1]} columns")
print()

# Check class distribution
print("Target distribution:")
high_risk_count = (df['Disruption_Risk'] == 1).sum()
low_risk_count = (df['Disruption_Risk'] == 0).sum()
print(f"  High Risk: {high_risk_count} ({high_risk_count/len(df)*100:.1f}%)")
print(f"  Low Risk: {low_risk_count} ({low_risk_count/len(df)*100:.1f}%)")
print()

# Select features to use in the model
print("Preparing features...")

numerical_features = [
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

categorical_features = [
    'Supplier_Country', 'Supplier_Region', 'Product_Category',
    'Product_Type', 'Transportation_Mode', 'Single_Source_Risk',
    'Production_Dependency_Level'
]

# Make a copy to work with
df_processed = df.copy()

# Encode categorical variables
print("Encoding categorical variables...")
label_encoders = {}
for col in categorical_features:
    le = LabelEncoder()
    df_processed[col + '_Encoded'] = le.fit_transform(df_processed[col])
    label_encoders[col] = le
    numerical_features.append(col + '_Encoded')

print(f"Total features: {len(numerical_features)}")
print()

# Prepare X and y
X = df_processed[numerical_features]
y = df_processed['Disruption_Risk']

# Train/test split
print("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Training samples: {len(X_train)}")
print(f"  Testing samples: {len(X_test)}")
print()

# Feature scaling (important for logistic regression)
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print()

# Train multiple models and compare
print("Training models...")
print("-"*70)

model_configs = {
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

model_results = {}
best_model_name = None
best_score = 0

for name, clf in model_configs.items():
    print(f"\nTraining {name}...")
    
    # Use scaled data for logistic regression, original for tree models
    if name == 'Logistic Regression':
        clf.fit(X_train_scaled, y_train)
        y_pred = clf.predict(X_test_scaled)
        y_proba = clf.predict_proba(X_test_scaled)[:, 1]
    else:
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    model_results[name] = {
        'model': clf,
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
        'roc_auc': auc,
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'predictions': y_pred,
        'pred_proba': y_proba
    }
    
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {auc:.4f}")
    
    # Track best model based on F1 score
    if f1 > best_score:
        best_score = f1
        best_model_name = name

print()
print("-"*70)
print(f"\nBest model: {best_model_name} (F1: {best_score:.4f})")
print()

# Save the best model and related artifacts
print("Saving model and artifacts...")
best_model = model_results[best_model_name]['model']
joblib.dump(best_model, "model/supply_chain_model_advanced.pkl")
joblib.dump(scaler, "model/scaler.pkl")
joblib.dump(label_encoders, "model/encoders.pkl")

# Save feature list
with open("model/feature_columns.json", "w") as f:
    json.dump(numerical_features, f)

print("  Saved model: model/supply_chain_model_advanced.pkl")
print("  Saved scaler: model/scaler.pkl")
print("  Saved encoders: model/encoders.pkl")
print()

# Save training metrics
print("Saving performance metrics...")
metrics_dict = {
    'training_date': datetime.now().isoformat(),
    'dataset_size': len(df),
    'train_size': len(X_train),
    'test_size': len(X_test),
    'num_features': len(numerical_features),
    'best_model': best_model_name,
    'models_performance': {
        name: {
            'accuracy': float(res['accuracy']),
            'precision': float(res['precision']),
            'recall': float(res['recall']),
            'f1_score': float(res['f1_score']),
            'roc_auc': float(res['roc_auc']),
            'confusion_matrix': res['confusion_matrix']
        }
        for name, res in model_results.items()
    }
}

with open("model/training_metrics.json", "w") as f:
    json.dump(metrics_dict, f, indent=2)

print("  Saved metrics: model/training_metrics.json")
print()

# Print classification report for best model
print("Classification Report (Best Model):")
print()
print(classification_report(y_test, model_results[best_model_name]['predictions']))

print()
print("="*70)
print(" "*20 + "TRAINING COMPLETE!")
print("="*70)
