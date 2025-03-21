import streamlit as st
import pandas as pd
import plotly.express as px
import mysql.connector
from prophet import Prophet  # Facebook Prophet for forecasting

# Set page configuration
st.set_page_config(page_title="Fashion Demand Forecast", layout="wide")

# Apply custom CSS for styling
st.markdown(
    """
    <style>
        .stApp { background-color: #87CEFA !important; }
        section[data-testid="stSidebar"] { background-color: #D3D3D3 !important; }
        section[data-testid="stSidebar"] * { color: black !important; font-weight: bold !important; }
        div[data-testid="stFileUploaderDropzone"] div { color: black !important; font-weight: bold !important; }
        div[data-testid="stFileUploaderDropzone"] { background-color: #555555 !important; border-radius: 10px !important; padding: 20px !important; border: 2px solid #888888 !important; }
        div[data-testid="stFileUploaderDropzone"] button { color: #555555 !important; background-color: #555555 !important; font-weight: bold !important; border-radius: 5px !important; padding: 5px 10px !important; }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar for File Upload and Database Connection
st.sidebar.title("⚙️ Data Options")

# **Part 1: File Upload**
st.sidebar.subheader("📂 Upload CSV File")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv"])

# **Part 2: Database Connection**
st.sidebar.subheader("🔗 Connect to Database")
use_db = st.sidebar.checkbox("Use Database Instead")

sales_data = None  # Initialize as None

if use_db:
    db_host = st.sidebar.text_input("Database Host", "localhost")
    db_name = st.sidebar.text_input("Database Name")
    db_user = st.sidebar.text_input("Username")
    db_password = st.sidebar.text_input("Password", type="password")
    table_name = st.sidebar.text_input("Table Name", "fashion_sales")

    if st.sidebar.button("Connect"):
        try:
            conn = mysql.connector.connect(
                host=db_host,
                user=db_user,
                password=db_password,
                database=db_name
            )
            cursor = conn.cursor(dictionary=True)  # Fetch rows as dictionaries

            query = f"SELECT * FROM {table_name}"
            cursor.execute(query)
            rows = cursor.fetchall()
            cursor.close()
            conn.close()

            if rows:
                sales_data = pd.DataFrame(rows)
                st.sidebar.success("✅ Connected Successfully!")
            else:
                sales_data = pd.DataFrame(columns=["Date", "Category", "Sales"])  # Empty DataFrame
                st.sidebar.warning("⚠️ No data found in the table!")

        except mysql.connector.Error as e:
            st.sidebar.error(f"❌ Connection Failed: {e}")
            st.stop()

# **Part 3: Load CSV File (If Not Using Database)**
elif uploaded_file:
    sales_data = pd.read_csv(uploaded_file)
    st.sidebar.success("✅ File Uploaded Successfully!")

else:
    st.sidebar.warning("Please upload a CSV file or connect to a database.")
    st.stop()

# **Ensure 'Date' Column is Datetime**
# Standardize column names
sales_data.columns = sales_data.columns.str.strip().str.lower()

# Identify 'Date' column dynamically
date_col = [col for col in sales_data.columns if "date" in col]
if date_col:
    sales_data.rename(columns={date_col[0]: "Date"}, inplace=True)
    sales_data["Date"] = pd.to_datetime(sales_data["Date"])
else:
    st.error("❌ No column related to 'Date' found.")
    st.write("Available Columns:", sales_data.columns.tolist())
    st.stop()


# **Ensure 'Category' and 'Sales' Columns Exist**
# Standardize column names
sales_data.columns = sales_data.columns.str.strip().str.lower()

# Identify and rename key columns
column_map = {
    "date": [col for col in sales_data.columns if "date" in col],
    "sales": [col for col in sales_data.columns if "sales" in col],
    "category": [col for col in sales_data.columns if "category" in col]
}

for key, cols in column_map.items():
    if cols:
        sales_data.rename(columns={cols[0]: key.capitalize()}, inplace=True)
    else:
        st.error(f"❌ No column related to '{key.capitalize()}' found.")
        st.write("Available Columns:", sales_data.columns.tolist())
        st.stop()

# Convert Date column to datetime format
sales_data["Date"] = pd.to_datetime(sales_data["Date"])

# **Debugging - Display Data**
st.write("Fetched Data Preview:", sales_data.head())

# **Part 4: Category Selection**
st.sidebar.subheader("📌 Select Categories to Display:")
selected_categories = []

if not sales_data.empty:
    categories = sales_data["Category"].unique()
    for category in categories:
        if st.sidebar.checkbox(category, value=True):  # Default: All checked
            selected_categories.append(category)

# **Filter Data Based on Selected Categories**
filtered_data = sales_data[sales_data["Category"].isin(selected_categories)] if selected_categories else sales_data.iloc[0:0]

# **Main Page Content**
st.title("📊 Fashion Demand Forecasting Dashboard")

# **KPI Metrics**
col1, col2, col3 = st.columns(3)
col1.metric("Total Sales", f"{filtered_data['Sales'].sum()} units")
col2.metric("Top Selling Category", filtered_data.groupby("Category")["Sales"].sum().idxmax() if not filtered_data.empty else "N/A")
col3.metric("Highest Sales Day", filtered_data.loc[filtered_data["Sales"].idxmax(), "Date"].strftime('%Y-%m-%d') if not filtered_data.empty else "N/A")

# **Line Chart for Sales Trends**
fig = px.line(filtered_data, x="Date", y="Sales", color="Category", title="Sales Trends Over Time")
st.plotly_chart(fig, use_container_width=True)

# **Sales Forecasting with Prophet**
st.subheader("🔮 Future Sales Prediction")

forecast_period = st.slider("Select Forecast Period (days)", min_value=7, max_value=365, value=30)

# **Check if dataset is valid for forecasting**
if filtered_data.empty:
    st.error("⚠️ No data available for forecasting.")
else:
    df_prophet = filtered_data[["Date", "Sales"]].rename(columns={"Date": "ds", "Sales": "y"})

    # Train Prophet Model
    model = Prophet()
    model.fit(df_prophet)

    # Create future dataframe
    future = model.make_future_dataframe(periods=forecast_period)
    forecast = model.predict(future)

    # **Plot Forecast**
    fig_forecast = px.line(forecast, x="ds", y="yhat", title="📈 Forecasted Sales Trends")
    st.plotly_chart(fig_forecast, use_container_width=True)

# **Display Data Table**
st.subheader("📋 Sales Data")
st.dataframe(filtered_data)
