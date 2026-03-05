# Script to generate synthetic supply chain data
# Creates a realistic dataset for testing the prediction model

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Set seed for reproducibility
np.random.seed(42)
random.seed(42)

print("Generating synthetic supply chain dataset...")
print()

# Number of records to generate
num_records = 600

print(f"Creating {num_records} records...")

# Generate the data
data = {}

# Basic identifiers
data['Record_ID'] = [f'SC{str(i).zfill(5)}' for i in range(1, num_records + 1)]
data['Supplier_ID'] = [f'SUP{str(random.randint(1, 50)).zfill(3)}' for _ in range(num_records)]

# Supplier names - mix of real-sounding company names
supplier_names = [
    'Acme Corp', 'Global Supplies Inc', 'Tech Components Ltd', 'Prime Materials Co',
    'Express Logistics', 'Mega Manufacturing', 'Pacific Traders', 'Euro Distributors',
    'Asia Sourcing', 'Velocity Suppliers', 'Quantum Materials', 'Nexus Logistics'
]
data['Supplier_Name'] = [random.choice(supplier_names) for _ in range(num_records)]

# Geographic data
countries = ['China', 'USA', 'Germany', 'India', 'Japan', 'Mexico', 'Vietnam', 'South Korea', 'Taiwan', 'Thailand']
data['Supplier_Country'] = [random.choice(countries) for _ in range(num_records)]

regions = ['Asia-Pacific', 'North America', 'Europe', 'Latin America']
data['Supplier_Region'] = [random.choice(regions) for _ in range(num_records)]

warehouses = ['Los Angeles', 'New York', 'Chicago', 'Houston', 'Seattle', 'Atlanta', 'Boston', 'Dallas', 'Phoenix', 'Denver']
data['Warehouse_Location'] = [random.choice(warehouses) for _ in range(num_records)]

# Product information
categories = ['Electronics', 'Raw Materials', 'Automotive Parts', 'Consumer Goods', 
              'Industrial Equipment', 'Chemicals', 'Textiles', 'Food & Beverage', 'Pharmaceuticals']
data['Product_Category'] = [random.choice(categories) for _ in range(num_records)]

product_types = ['Component', 'Assembly', 'Raw Material', 'Finished Good', 'Semi-Finished']
data['Product_Type'] = [random.choice(product_types) for _ in range(num_records)]

data['SKU'] = [f'SKU{str(random.randint(1000, 9999))}' for _ in range(num_records)]
data['Product_Value_USD'] = np.random.uniform(5000, 500000, num_records).round(2)

# Supply chain metrics
data['Lead_Time_Days'] = np.random.randint(5, 60, num_records)
data['Order_Quantity'] = np.random.randint(100, 10000, num_records)
data['Stock_Level'] = np.random.randint(0, 5000, num_records)
data['Reorder_Point'] = np.random.randint(200, 2000, num_records)
data['Safety_Stock'] = np.random.randint(100, 1500, num_records)
data['Inventory_Turnover'] = np.random.uniform(2, 12, num_records).round(2)

# Supplier performance
data['On_Time_Delivery_Rate'] = np.random.uniform(65, 100, num_records).round(2)
data['Quality_Rating'] = np.random.uniform(65, 100, num_records).round(2)
data['Defect_Rate_Percent'] = np.random.uniform(0, 8, num_records).round(2)
data['Supplier_Responsiveness_Score'] = np.random.uniform(60, 100, num_records).round(1)
data['Supplier_Capacity_Utilization'] = np.random.uniform(50, 100, num_records).round(1)
data['Financial_Stability_Score'] = np.random.uniform(40, 100, num_records).round(1)

# Logistics
transport_modes = ['Air Freight', 'Sea Freight', 'Road Transport', 'Rail', 'Multimodal']
data['Transportation_Mode'] = [random.choice(transport_modes) for _ in range(num_records)]

carriers = ['DHL', 'FedEx', 'UPS', 'Maersk', 'COSCO', 'DB Schenker', 'Kuehne+Nagel', 'XPO Logistics']
data['Carrier_Name'] = [random.choice(carriers) for _ in range(num_records)]

data['Shipping_Time_Days'] = np.random.randint(1, 45, num_records)
data['Customs_Clearance_Time_Days'] = np.random.randint(0, 10, num_records)
data['Transportation_Cost_USD'] = np.random.uniform(500, 50000, num_records).round(2)

# Risk factors
data['Geopolitical_Risk_Score'] = np.random.uniform(0, 10, num_records).round(1)
data['Weather_Risk_Score'] = np.random.uniform(0, 10, num_records).round(1)
data['Demand_Volatility'] = np.random.uniform(0, 10, num_records).round(1)
data['Supply_Volatility'] = np.random.uniform(0, 10, num_records).round(1)
data['Port_Congestion_Level'] = np.random.uniform(0, 10, num_records).round(1)
data['Supplier_Dependency_Score'] = np.random.uniform(0, 10, num_records).round(1)
data['Alternative_Suppliers_Available'] = np.random.randint(0, 5, num_records)
data['Single_Source_Risk'] = [random.choice(['Yes', 'No']) for _ in range(num_records)]

# Historical data
data['Historical_Disruptions_Count'] = np.random.randint(0, 15, num_records)
data['Average_Delay_Days'] = np.random.uniform(0, 20, num_records).round(1)
data['Contract_Compliance_Rate'] = np.random.uniform(70, 100, num_records).round(2)
data['Communication_Response_Time_Hours'] = np.random.uniform(1, 72, num_records).round(1)

# Financial metrics
data['Total_Annual_Spend_USD'] = np.random.uniform(50000, 5000000, num_records).round(2)
data['Payment_Terms_Days'] = np.random.choice([30, 45, 60, 90], num_records)
data['Price_Volatility_Percent'] = np.random.uniform(0, 25, num_records).round(2)

# Operational
dependency_levels = ['Critical', 'High', 'Medium', 'Low']
data['Production_Dependency_Level'] = [random.choice(dependency_levels) for _ in range(num_records)]
data['Customer_Impact_Score'] = np.random.uniform(0, 10, num_records).round(1)
data['Revenue_At_Risk_USD'] = np.random.uniform(0, 1000000, num_records).round(2)

# Create dataframe
df = pd.DataFrame(data)

# Calculate disruption risk based on multiple factors
def calc_risk(row):
    score = 0
    
    # Check various risk factors
    if row['Lead_Time_Days'] > 40:
        score += 2
    elif row['Lead_Time_Days'] > 25:
        score += 1
    
    if row['Stock_Level'] < row['Reorder_Point']:
        score += 3
    elif row['Stock_Level'] < row['Safety_Stock']:
        score += 2
    
    if row['On_Time_Delivery_Rate'] < 85:
        score += 2
    if row['Defect_Rate_Percent'] > 4:
        score += 2
    if row['Quality_Rating'] < 80:
        score += 1
    
    if row['Geopolitical_Risk_Score'] > 7:
        score += 3
    elif row['Geopolitical_Risk_Score'] > 5:
        score += 1
    
    if row['Weather_Risk_Score'] > 7:
        score += 2
    
    if row['Demand_Volatility'] > 7:
        score += 1
    if row['Supply_Volatility'] > 7:
        score += 2
    if row['Port_Congestion_Level'] > 7:
        score += 2
    
    if row['Single_Source_Risk'] == 'Yes':
        score += 2
    if row['Alternative_Suppliers_Available'] == 0:
        score += 3
    if row['Production_Dependency_Level'] == 'Critical':
        score += 2
    
    if row['Historical_Disruptions_Count'] > 8:
        score += 2
    if row['Average_Delay_Days'] > 10:
        score += 1
    
    if row['Financial_Stability_Score'] < 60:
        score += 3
    elif row['Financial_Stability_Score'] < 75:
        score += 1
    
    # Return 1 for high risk, 0 for low risk
    return 1 if score >= 10 else 0

print("Calculating disruption risk labels...")
df['Disruption_Risk'] = df.apply(calc_risk, axis=1)

# Add dates
base = datetime(2024, 1, 1)
df['Date'] = [base + timedelta(days=random.randint(0, 730)) for _ in range(num_records)]

# Calculate probability score
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

print()
print(f"✓ Generated {len(df)} records")
print(f"✓ High Risk: {(df['Disruption_Risk'] == 1).sum()} ({(df['Disruption_Risk'] == 1).sum()/len(df)*100:.1f}%)")
print(f"✓ Low Risk: {(df['Disruption_Risk'] == 0).sum()} ({(df['Disruption_Risk'] == 0).sum()/len(df)*100:.1f}%)")
print()
print(f"Dataset saved to: data/supply_chain_advanced.csv")
print(f"Shape: {df.shape}")
print()
print("Done!")
