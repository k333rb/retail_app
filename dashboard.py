import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

# Database connection
DATABASE_URI = 'postgresql://postgres:eratha123@127.0.0.1:5432/data_warehouse'
engine = create_engine(DATABASE_URI)

# ETL Process
@st.cache_data
def extract_data():
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
    return pd.read_sql(query, engine)

@st.cache_data
def transform_data(data):
    data_clean = data.dropna(subset=['customer_number', 'product_description'])
    data_clean = data_clean[data_clean['quantity'] > 0]
    data_clean = data_clean[data_clean['unit_price'] > 0]
    data_clean['invoice_date'] = pd.to_datetime(data_clean['invoice_date'])
    upper_quantity_limit = data_clean['quantity'].quantile(0.99)
    upper_price_limit = data_clean['unit_price'].quantile(0.99)
    data_clean = data_clean[
        (data_clean['quantity'] <= upper_quantity_limit) & 
        (data_clean['unit_price'] <= upper_price_limit)
    ]
    data_clean['total_price'] = data_clean['quantity'] * data_clean['unit_price']
    return data_clean

@st.cache_data
def load_data():
    data = extract_data()
    return transform_data(data)

# Clustering Function
def perform_clustering(data, n_clusters=3):
    customer_features = data[['quantity', 'unit_price', 'total_price']].dropna()
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    data['Cluster'] = kmeans.fit_predict(customer_features)
    return data

# Regression Analysis
def perform_sales_prediction(data):
    X = data[['total_price']]
    y = data['quantity']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=X_test.squeeze(), y=y_test, mode='markers', name='Actual', marker=dict(color='blue')))
    fig.add_trace(go.Scatter(x=X_test.squeeze(), y=y_pred, mode='lines', name='Predicted', line=dict(color='red')))
    fig.update_layout(
        title="Sales Prediction",
        xaxis_title="Total Price",
        yaxis_title="Quantity",
        legend_title="Legend"
    )
    
    return fig, y_test, y_pred

def generate_prediction_insights(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return f"""
    ### **Sales Prediction Insights**
    - The **Mean Absolute Error (MAE)** is **{mae:.2f}**, meaning the predictions are on average off by this much.
    - The **R-squared (R²)** value is **{r2:.2f}**, which means the model explains about **{r2*100:.2f}%** of the variation in sales quantity.
    - The blue dots represent actual sales data, while the red line shows predicted sales trends.
    - If most blue dots are close to the red line, the model's predictions are accurate.
    """

st.set_page_config(page_title="Business Intelligence Dashboard", layout="wide")

st.title('Dashboard')
st.markdown("""
Welcome to the Retail Dashboard! Use this tool to analyze sales data, segment customers, 
predict future sales, and visualize key metrics. Navigate through the tabs for specific insights.
""")

# Load and clean data
data = load_data()

if data.empty:
    st.error("No data available. Please check your database or ETL process.")
else:
    # Sidebar for filters
    with st.sidebar:
        st.header("Filters")
        year_filter = st.multiselect('Select Year', options=sorted(data['year'].unique()), help="Filter data by year.")
        country_filter = st.multiselect('Select Country', options=sorted(data['country'].unique()), help="Filter data by country.")
        date_range = st.date_input(
            "Select Date Range",
            [data['invoice_date'].min(), data['invoice_date'].max()],
            help="Choose a date range to view data within a specific period."
        )
        st.header("Export Options")
        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button("Download Filtered Data as CSV", data=csv, file_name="filtered_data.csv", mime="text/csv")

    # Apply filters to data
    if year_filter:
        data = data[data['year'].isin(year_filter)]
    if country_filter:
        data = data[data['country'].isin(country_filter)]
    if date_range:
        start_date, end_date = date_range
        data = data[(data['invoice_date'] >= pd.Timestamp(start_date)) & (data['invoice_date'] <= pd.Timestamp(end_date))]

    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Customer Segmentation", "Sales Prediction", "Visualizations"])

    # Tab 1: Overview
    with tab1:
        st.header("Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            total_sales = data['total_price'].sum()
            st.metric('Total Sales', f"${total_sales:,.2f}")
        with col2:
            total_orders = data['invoice_date'].nunique()
            st.metric('Total Orders', total_orders)
        with col3:
            total_customers = data['customer_number'].nunique()
            st.metric('Total Customers', total_customers)
        with col4:
            avg_order_value = total_sales / total_orders if total_orders else 0
            st.metric('Avg Order Value', f"${avg_order_value:,.2f}")

    # Tab 2: Customer Segmentation
    with tab2:
        st.header("Customer Segmentation")
        n_clusters = st.slider("Select Number of Clusters", min_value=2, max_value=10, value=3)
        data = perform_clustering(data, n_clusters=n_clusters)
        fig = px.scatter(
            data, x='unit_price', y='quantity', color='Cluster', title=f"Customer Segmentation ({n_clusters} Clusters)",
            labels={'unit_price': 'Unit Price', 'quantity': 'Quantity'}
        )
        st.plotly_chart(fig)

        # Add Non-Technical Insights
        st.markdown(f"""
        ### **Understanding the Customer Segmentation Chart**
        - This chart groups customers into **{n_clusters} segments (or clusters)** based on their purchasing behavior.
        - **Unit Price** (X-axis) shows how much customers typically pay for individual items.
        - **Quantity** (Y-axis) indicates the number of items customers buy in a single purchase.

        ### **Key Observations**
        1. Each color represents a different customer segment, with similar purchasing behaviors grouped together.
        2. **High Spend, Low Quantity**: Some customers tend to buy fewer items at higher prices.
        3. **Low Spend, High Quantity**: Others buy many items but at lower unit prices.
        4. **Use Case**: Use these segments to personalize marketing strategies, such as offering discounts to low-spend customers or premium services to high-spend ones.
        """)

    # Tab 3: Sales Prediction
    with tab3:
        st.header("Sales Prediction")
        prediction_fig, y_test, y_pred = perform_sales_prediction(data)
        st.plotly_chart(prediction_fig)
        insights = generate_prediction_insights(y_test, y_pred)
        st.markdown(insights)

   # Tab 4: Visualizations with Dynamic Insights
with tab4:
    st.header("Additional Visualizations")
    
    # Top Selling Products
    st.subheader("Top Selling Products")
    top_products = data.groupby('product_description')['quantity'].sum().nlargest(10)
    fig_products = px.bar(
        top_products,
        x=top_products.index,
        y=top_products.values,
        labels={'x': 'Product', 'y': 'Quantity Sold'},
        title="Top Selling Products"
    )
    st.plotly_chart(fig_products)

    # Insights for Top Selling Products
    st.markdown(f"""
    ### **Insights for Top Selling Products**
    - The product **"{top_products.idxmax()}"** is the most sold item, with a total quantity of **{top_products.max()}** units.
    - Among the top 10 products, the total combined sales quantity is **{top_products.sum()}** units.
    - This data helps identify which products are driving the most revenue and can guide inventory management and marketing campaigns.
    """)

    # Sales by Country
    st.subheader("Sales by Country")
    sales_by_country = data.groupby('country')['quantity'].sum().nlargest(10)
    fig_country = px.bar(
        sales_by_country,
        x=sales_by_country.index,
        y=sales_by_country.values,
        labels={'x': 'Country', 'y': 'Quantity Sold'},
        title="Sales by Country"
    )
    st.plotly_chart(fig_country)

    # Insights for Sales by Country
    st.markdown(f"""
    ### **Insights for Sales by Country**
    - The country with the highest sales is **"{sales_by_country.idxmax()}"**, with a total of **{sales_by_country.max()}** units sold.
    - The top 10 countries combined account for **{sales_by_country.sum()}** units sold.
    - The data shows a clear dominance by **"{sales_by_country.idxmax()}"**, indicating potential for scaling operations in other high-performing regions.
    - Use these insights to optimize distribution channels and tailor marketing strategies to specific countries.
    """)

