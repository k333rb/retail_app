# Import necessary libraries
from sqlalchemy import create_engine
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Define your PostgreSQL connection string
# Format: postgresql://username:password@host:port/database
DATABASE_URI = 'postgresql://postgres:password@127.0.0.1:5432/data_warehouse'

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
plt.scatter(data['unit_price'], data['quantity'],  # Flip the axes here
            c=data['Cluster'], cmap='viridis')
plt.xlabel('UnitPrice')  # Now x-axis represents UnitPrice
plt.ylabel('Quantity')  # Now y-axis represents Quantity
plt.title('Customer Segmentation')


# Predictive analysis: Forecast Quantity based on UnitPrice
X = data[['unit_price']]
y = data['quantity']

# Split data into training and testing sets (if desired)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Fit the linear regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predict and evaluate
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Visualize the regression line
plt.figure()
plt.scatter(X_test, y_test, color='blue')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.xlabel('UnitPrice')
plt.ylabel('Quantity')
plt.title('Sales Prediction')

# Display both plots
plt.show()
