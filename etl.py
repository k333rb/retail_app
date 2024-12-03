import pandas as pd

# 1. Extract: Load the dataset
file_path = './Online Retail.xlsx'
data = pd.read_excel(file_path)

# 2. Data Exploration
print(data.isnull().sum())  # Check missing values
print(data.describe())      # Check basic stats
print(data.duplicated().sum())  # Check duplicates

# 3. Transform: Data Cleaning
# Drop missing values in 'CustomerID' and 'Description'
data_clean = data.dropna(subset=['CustomerID', 'Description'])

# Remove rows with negative 'Quantity' and 'UnitPrice'
data_clean = data_clean[data_clean['Quantity'] > 0]
data_clean = data_clean[data_clean['UnitPrice'] > 0]

# Convert 'InvoiceDate' to datetime format
data_clean['InvoiceDate'] = pd.to_datetime(data_clean['InvoiceDate'])

# Handle outliers (remove extreme values)
upper_quantity_limit = data_clean['Quantity'].quantile(0.99)
upper_price_limit = data_clean['UnitPrice'].quantile(0.99)
data_clean = data_clean[(data_clean['Quantity'] <= upper_quantity_limit) &
                        (data_clean['UnitPrice'] <= upper_price_limit)]

# 4. Load: Save the cleaned data to CSV
data_clean.to_csv('cleaned_online_retail.csv', index=False)
