# Import necessary libraries
from sqlalchemy import create_engine
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Define your PostgreSQL connection string
# Format: postgresql://username:password@host:port/database
DATABASE_URI = 'postgresql://postgres:eratha123@127.0.0.1:5432/data_warehouse'

# Create a connection to the database
engine = create_engine(DATABASE_URI)

# Load data from PostgreSQL using SQLAlchemy (using pandas read_sql method)
query = '''
SELECT 
    fs.quantity,
    fs.unit_price,
    c.customer_number,
    p.stock_code,
    p.description AS product_description,
    t.invoice_date,
    t.day,
    t.month,
    t.year,
    t.weekday,
    c.country
FROM FactSales fs
JOIN DimProduct p ON fs.product_id = p.product_id
JOIN DimCustomer c ON fs.customer_id = c.customer_id
JOIN DimTime t ON fs.time_id = t.time_id;
'''
# Fetch data into a pandas DataFrame
data = pd.read_sql(query, engine)

# Check the first few rows of the data
print(data.head())

# Update customer features for clustering
customer_features = data[['quantity', 'unit_price']]

# Perform clustering for customer segmentation
kmeans = KMeans(n_clusters=3)
data['Cluster'] = kmeans.fit_predict(customer_features)

# Visualize the clustering
plt.figure()
plt.scatter(data['unit_price'], data['quantity'],  # Flip the axes
            c=data['Cluster'], cmap='viridis')
plt.xlabel('UnitPrice')  # Now x-axis represents UnitPrice
plt.ylabel('Quantity')  # Now y-axis represents Quantity
plt.title('Customer Segmentation')

#-----------------------------------------------------------------------

#Total Spend by Customer (aggregate feature)
data['total_spend'] = data['unit_price'] * data['quantity']
total_spend_customer = data.groupby('customer_number')['total_spend'].sum().reset_index()
data = data.merge(total_spend_customer, on='customer_number', suffixes=('', '_customer_total_spend'))

X = data[['unit_price', 'total_spend']]
y = data['quantity']  # Target variable

# Convert categorical features into numerical (if needed)
X = pd.get_dummies(X, drop_first=True)  # One-Hot Encoding for categorical variables

# Scale the features before splitting the data
scaler = StandardScaler()  # Initialize the StandardScaler
X_scaled = scaler.fit_transform(X)  # Fit and transform the features

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a K-Nearest Neighbors Regressor model
knn_regressor = KNeighborsRegressor(n_neighbors=7)  # You can adjust 'n_neighbors' for better results

# Train the model
knn_regressor.fit(X_train, y_train)

# Predict using the trained model
y_pred = knn_regressor.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output the results
print(f"Mean Squared Error (MSE): {mse}")
print(f"R² Score: {r2}")

# Visualization: Predicted vs Actual Values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5, label="Predicted vs Actual")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2, label="Perfect Prediction")
plt.xlabel("Actual Quantity")
plt.ylabel("Predicted Quantity")
plt.title("KNN Regressor: Predicted vs Actual Quantity (with Total Spend Feature)")
plt.legend()

# Display all plots
plt.show()