import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Number of records
n_records = 600

# Generate synthetic supply chain data
data = {
    # Identifiers
    'Record_ID': [f'SC{str(i).zfill(5)}' for i in range(1, n_records + 1)],
    'Supplier_ID': [f'SUP{str(random.randint(1, 50)).zfill(3)}' for _ in range(n_records)],
    'Supplier_Name': [random.choice(['Acme Corp', 'Global Supplies Inc', 'Tech Components Ltd', 
                                     'Prime Materials Co', 'Express Logistics', 'Mega Manufacturing',
                                     'Pacific Traders', 'Euro Distributors', 'Asia Sourcing',
                                     'Velocity Suppliers', 'Quantum Materials', 'Nexus Logistics']) 
                      for _ in range(n_records)],
    
    # Location & Geographic Data
    'Supplier_Country': [random.choice(['China', 'USA', 'Germany', 'India', 'Japan', 'Mexico', 
                                        'Vietnam', 'South Korea', 'Taiwan', 'Thailand']) 
                         for _ in range(n_records)],
    'Supplier_Region': [random.choice(['Asia-Pacific', 'North America', 'Europe', 'Latin America']) 
                        for _ in range(n_records)],
    'Warehouse_Location': [random.choice(['Los Angeles', 'New York', 'Chicago', 'Houston', 'Seattle',
                                          'Atlanta', 'Boston', 'Dallas', 'Phoenix', 'Denver']) 
                           for _ in range(n_records)],
    
    # Product Information
    'Product_Category': [random.choice(['Electronics', 'Raw Materials', 'Automotive Parts', 
                                        'Consumer Goods', 'Industrial Equipment', 'Chemicals',
                                        'Textiles', 'Food & Beverage', 'Pharmaceuticals']) 
                         for _ in range(n_records)],
    'Product_Type': [random.choice(['Component', 'Assembly', 'Raw Material', 'Finished Good', 
                                    'Semi-Finished']) for _ in range(n_records)],
    'SKU': [f'SKU{str(random.randint(1000, 9999))}' for _ in range(n_records)],
    'Product_Value_USD': np.random.uniform(5000, 500000, n_records).round(2),
    
    # Supply Chain Metrics
    'Lead_Time_Days': np.random.randint(5, 60, n_records),
    'Order_Quantity': np.random.randint(100, 10000, n_records),
    'Stock_Level': np.random.randint(0, 5000, n_records),
    'Reorder_Point': np.random.randint(200, 2000, n_records),
    'Safety_Stock': np.random.randint(100, 1500, n_records),
    'Inventory_Turnover': np.random.uniform(2, 12, n_records).round(2),
    
    # Supplier Performance Metrics
    'On_Time_Delivery_Rate': np.random.uniform(65, 100, n_records).round(2),
    'Quality_Rating': np.random.uniform(65, 100, n_records).round(2),
    'Defect_Rate_Percent': np.random.uniform(0, 8, n_records).round(2),
    'Supplier_Responsiveness_Score': np.random.uniform(60, 100, n_records).round(1),
    'Supplier_Capacity_Utilization': np.random.uniform(50, 100, n_records).round(1),
    'Financial_Stability_Score': np.random.uniform(40, 100, n_records).round(1),
    
    # Logistics & Transportation
    'Transportation_Mode': [random.choice(['Air Freight', 'Sea Freight', 'Road Transport', 
                                           'Rail', 'Multimodal']) for _ in range(n_records)],
    'Carrier_Name': [random.choice(['DHL', 'FedEx', 'UPS', 'Maersk', 'COSCO', 
                                    'DB Schenker', 'Kuehne+Nagel', 'XPO Logistics']) 
                     for _ in range(n_records)],
    'Shipping_Time_Days': np.random.randint(1, 45, n_records),
    'Customs_Clearance_Time_Days': np.random.randint(0, 10, n_records),
    'Transportation_Cost_USD': np.random.uniform(500, 50000, n_records).round(2),
    
    # Risk Factors
    'Geopolitical_Risk_Score': np.random.uniform(0, 10, n_records).round(1),
    'Weather_Risk_Score': np.random.uniform(0, 10, n_records).round(1),
    'Demand_Volatility': np.random.uniform(0, 10, n_records).round(1),
    'Supply_Volatility': np.random.uniform(0, 10, n_records).round(1),
    'Port_Congestion_Level': np.random.uniform(0, 10, n_records).round(1),
    'Supplier_Dependency_Score': np.random.uniform(0, 10, n_records).round(1),
    'Alternative_Suppliers_Available': np.random.randint(0, 5, n_records),
    'Single_Source_Risk': [random.choice(['Yes', 'No']) for _ in range(n_records)],
    
    # Historical Performance
    'Historical_Disruptions_Count': np.random.randint(0, 15, n_records),
    'Average_Delay_Days': np.random.uniform(0, 20, n_records).round(1),
    'Contract_Compliance_Rate': np.random.uniform(70, 100, n_records).round(2),
    'Communication_Response_Time_Hours': np.random.uniform(1, 72, n_records).round(1),
    
    # Financial Metrics
    'Total_Annual_Spend_USD': np.random.uniform(50000, 5000000, n_records).round(2),
    'Payment_Terms_Days': np.random.choice([30, 45, 60, 90], n_records),
    'Price_Volatility_Percent': np.random.uniform(0, 25, n_records).round(2),
    
    # Operational Metrics
    'Production_Dependency_Level': [random.choice(['Critical', 'High', 'Medium', 'Low']) 
                                    for _ in range(n_records)],
    'Customer_Impact_Score': np.random.uniform(0, 10, n_records).round(1),
    'Revenue_At_Risk_USD': np.random.uniform(0, 1000000, n_records).round(2),
}

# Create DataFrame
df = pd.DataFrame(data)

# Generate complex disruption label based on multiple factors
def calculate_disruption_risk(row):
    risk_score = 0
    
    # Lead time risk
    if row['Lead_Time_Days'] > 40:
        risk_score += 2
    elif row['Lead_Time_Days'] > 25:
        risk_score += 1
    
    # Stock level risk
    if row['Stock_Level'] < row['Reorder_Point']:
        risk_score += 3
    elif row['Stock_Level'] < row['Safety_Stock']:
        risk_score += 2
    
    # Supplier performance risk
    if row['On_Time_Delivery_Rate'] < 85:
        risk_score += 2
    if row['Defect_Rate_Percent'] > 4:
        risk_score += 2
    if row['Quality_Rating'] < 80:
        risk_score += 1
    
    # Geographic and external risks
    if row['Geopolitical_Risk_Score'] > 7:
        risk_score += 3
    elif row['Geopolitical_Risk_Score'] > 5:
        risk_score += 1
    
    if row['Weather_Risk_Score'] > 7:
        risk_score += 2
    
    # Supply chain risks
    if row['Demand_Volatility'] > 7:
        risk_score += 1
    if row['Supply_Volatility'] > 7:
        risk_score += 2
    if row['Port_Congestion_Level'] > 7:
        risk_score += 2
    
    # Dependency risks
    if row['Single_Source_Risk'] == 'Yes':
        risk_score += 2
    if row['Alternative_Suppliers_Available'] == 0:
        risk_score += 3
    if row['Production_Dependency_Level'] == 'Critical':
        risk_score += 2
    
    # Historical performance
    if row['Historical_Disruptions_Count'] > 8:
        risk_score += 2
    if row['Average_Delay_Days'] > 10:
        risk_score += 1
    
    # Financial stability
    if row['Financial_Stability_Score'] < 60:
        risk_score += 3
    elif row['Financial_Stability_Score'] < 75:
        risk_score += 1
    
    # Determine disruption based on cumulative risk score
    # Use threshold to create balanced classes
    return 1 if risk_score >= 10 else 0

df['Disruption_Risk'] = df.apply(calculate_disruption_risk, axis=1)

# Add timestamp
base_date = datetime(2024, 1, 1)
df['Date'] = [base_date + timedelta(days=random.randint(0, 730)) for _ in range(n_records)]

# Calculate disruption probability score (continuous)
df['Disruption_Probability'] = (
    (100 - df['On_Time_Delivery_Rate']) * 0.15 +
    df['Defect_Rate_Percent'] * 1.5 +
    df['Geopolitical_Risk_Score'] * 2 +
    df['Supply_Volatility'] * 1.8 +
    (df['Historical_Disruptions_Count'] / 15) * 10 +
    (df['Lead_Time_Days'] / 60) * 10
).clip(0, 100).round(2)

# Save to CSV
df.to_csv('data/supply_chain_advanced.csv', index=False)

print(f"✅ Generated {len(df)} records")
print(f"📊 Disruption Risk Distribution:")
print(f"   - High Risk (1): {(df['Disruption_Risk'] == 1).sum()} ({(df['Disruption_Risk'] == 1).sum()/len(df)*100:.1f}%)")
print(f"   - Low Risk (0): {(df['Disruption_Risk'] == 0).sum()} ({(df['Disruption_Risk'] == 0).sum()/len(df)*100:.1f}%)")
print(f"\n📁 Dataset saved to: data/supply_chain_advanced.csv")
print(f"\nDataset Shape: {df.shape}")
print(f"\nColumn Names ({len(df.columns)}):")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")
