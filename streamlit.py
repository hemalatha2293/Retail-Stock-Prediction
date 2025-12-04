

import streamlit as st
import pandas as pd
import joblib
from prophet import Prophet
import holidays
import os

# --- 1. Load Model and Data ---

# Check if the model file exists
model_path = 'prophet_model_with_calendar.pkl'
if not os.path.exists(model_path):
    st.error(f"Error: Model file '{model_path}' not found. Please ensure the model is trained and saved.")
    st.stop()

# Load the trained Prophet model
loaded_model = joblib.load(model_path)

# Load the original dataset (needed for feature derivation placeholders and product list)
data_path = 'Retail_Dataset_with_Seasonality_and_Demand.csv'
if not os.path.exists(data_path):
    st.error(f"Error: Data file '{data_path}' not found. Please ensure the CSV is available.")
    st.stop()

df_original = pd.read_csv(data_path)
df_original['ds'] = pd.to_datetime(df_original['Date_Received'])
df_original = df_original.sort_values('ds').reset_index(drop=True)

# Generate holiday calendar
def generate_indian_festival_calendar(start_year=2020, end_year=2035):
    # India-specific holidays
    india_holidays = holidays.India(years=range(start_year, end_year + 1))

    # Custom general holidays (e.g., New Year's Day)
    custom_holidays_list = []
    for year in range(start_year, end_year + 1):
        custom_holidays_list.append({
            'holiday': 'New Year\'s Day',
            'ds': pd.to_datetime(f'{year}-01-01'),
            'lower_window': 0,
            'upper_window': 1
        })
    custom_holidays_df = pd.DataFrame(custom_holidays_list)

    # Combine all holidays
    all_holidays_rows = []
    for date, name in india_holidays.items():
        all_holidays_rows.append({
            "holiday": name,
            "ds": pd.to_datetime(date),
            "lower_window": 0,
            "upper_window": 1
        })
    # Convert to DataFrame and concatenate with custom holidays
    all_holidays_df = pd.concat([pd.DataFrame(all_holidays_rows), custom_holidays_df], ignore_index=True)
    all_holidays_df = all_holidays_df.drop_duplicates(subset=['ds', 'holiday'])

    return all_holidays_df

holiday_df = generate_indian_festival_calendar(2020, 2035)

# --- 2. Helper Functions ---

def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    elif month in [9, 10, 11]:
        return "Autumn/Monsoon"
    else:
        return "Unknown"

def get_retail_season(date, festival_df):
    for idx, row in festival_df.iterrows():
        start_date = row['ds'] + pd.Timedelta(days=row['lower_window'])
        end_date = row['ds'] + pd.Timedelta(days=row['upper_window'])
        if start_date <= date <= end_date:
            return row['holiday']
    return "Regular"

# --- 3. Streamlit UI ---
st.set_page_config(page_title="Stock Quantity Prediction", layout="wide")
st.title("📈 Stock Quantity Prediction Dashboard")

st.header("Prediction Settings")

# Input fields now on the main page
prediction_date = st.date_input("Select Prediction Date", pd.to_datetime(df_original['ds'].max()) + pd.Timedelta(days=1))

# Product Selection (for display, model predicts total stock)
product_names = df_original['Product_Name'].unique().tolist()
selected_product = st.selectbox("Select Product (Note: Model predicts total stock)", product_names)

st.markdown("---<br>", unsafe_allow_html=True)

# --- 4. Prediction Logic ---

if st.button("Get Prediction"):
    st.subheader(f"Prediction for {prediction_date.strftime('%Y-%m-%d')}")

    # Prepare future dataframe for the single prediction date
    # The lag and rolling mean features are derived from the overall historical data,
    # as the model was trained on aggregated daily stock quantities.
    future_user = pd.DataFrame({
        'ds': [pd.to_datetime(prediction_date)],
        'lag_1': [df_original['Stock_Quantity'].iloc[-1]],
        'lag_7': [df_original['Stock_Quantity'].iloc[-7]],
        'rolling_mean_3': [df_original['Stock_Quantity'].tail(3).mean()],
        'rolling_mean_7': [df_original['Stock_Quantity'].tail(7).mean()],
        'is_weekend': [int(pd.to_datetime(prediction_date).dayofweek in [5,6])],
        'month': [pd.to_datetime(prediction_date).month],
        'is_month_end': [int(pd.to_datetime(prediction_date).is_month_end)]
    })

    # Predict
    forecast_user = loaded_model.predict(future_user)
    predicted_stock = forecast_user['yhat'].values[0]

    # Display Core Prediction
    st.metric(
        label="Predicted Total Stock Quantity", 
        value=f"{predicted_stock:.2f}",
        help="This is the predicted overall stock quantity for the selected date, not specific to the product selected in the sidebar."
    )

    st.info(f"☀️ Season: {get_season(pd.to_datetime(prediction_date).month)}")
    st.info(f"🎉 Retail Season/Festival: {get_retail_season(pd.to_datetime(prediction_date), holiday_df)}")

st.markdown("---<br>", unsafe_allow_html=True)
st.markdown("Developed with ❤️ using Prophet & Streamlit")
