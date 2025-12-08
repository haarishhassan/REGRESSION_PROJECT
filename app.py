import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from haversine import haversine, Unit
import warnings
warnings.filterwarnings('ignore')

# ---------------- Streamlit Config ----------------
st.set_page_config(page_title="Cab Fare Predictor", page_icon="🚕", layout="wide")



MODEL_FILE = "xgb_best_model.joblib"
try:
    model = joblib.load(MODEL_FILE)
    st.success("Model Loaded Successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

MODEL_FEATURE_NAMES = model.feature_names_in_.tolist()

# ---------------- Locations ----------------
LOCATIONS_COORDS = {
    'Back Bay': (42.355, -71.065),
    'Beacon Hill': (42.358, -71.070),
    'Boston University': (42.350, -71.105),
    'Fenway': (42.346, -71.097),
    'Financial District': (42.360, -71.056),
    'Haymarket Square': (42.363, -71.058),
    'North End': (42.364, -71.054),
    'North Station': (42.365, -71.061),
    'Northeastern University': (42.339, -71.089),
    'South Station': (42.352, -71.055),
    'Theatre District': (42.351, -71.064),
    'West End': (42.363, -71.063)
}

# ---------------- Company → Ride Types Mapping ----------------
COMPANY_RIDE_TYPES = {
    "Lyft": ["Shared", "Lyft", "Lyft XL", "Lux", "Lux Black", "Lux Black XL"],
    "Uber": ["UberX", "UberXL", "UberPool", "Black", "Black SUV", "WAV"]
}

# ---------------- Surge Calculation ----------------
def calculate_surge(hour, rainfall, weekend=0):
    surge = 1.0
    if 7 <= hour <= 9 or 17 <= hour <= 20:
        surge += 0.3
    surge += rainfall * 0.5
    if weekend == 1:
        surge += 0.2
    return min(surge, 3.0)

# ---------------- Feature Preparation ----------------
def create_model_input_df(data):
    X = pd.DataFrame(0, index=[0], columns=MODEL_FEATURE_NAMES)
    numerical_features = [
        'distance_km', 'temperature', 'cloud_coverage', 'pressure', 'rainfall',
        'humidity', 'wind_speed', 'hour', 'day', 'mounth', 'weekday', 'weekend', 'surge'
    ]
    for feature in numerical_features:
        if feature in X.columns:
            X.loc[0, feature] = data.get(feature, 0)

    # One-hot encoding company
    for comp in ["Lyft", "Uber"]:
        col = f'company_{comp}'
        if col in X.columns:
            X.loc[0, col] = 1 if data["company"] == comp else 0

    # One-hot encoding ride type
    all_rides = COMPANY_RIDE_TYPES["Lyft"] + COMPANY_RIDE_TYPES["Uber"]
    for rt in all_rides:
        col = f'ride_type_{rt}'
        if col in X.columns:
            X.loc[0, col] = 1 if data["ride_type"] == rt else 0

    return X[MODEL_FEATURE_NAMES]

# ---------------- UI ----------------
st.title("🚕 Cab Fare Predictor with Surge Pricing")
st.write("Select ride details, pickup & drop, enter weather and time manually.")

# -------- Ride Details --------
st.header("1. Ride Details🚘")
col1, col2 = st.columns(2)
with col1:
    company = st.selectbox("Company", ["Lyft", "Uber"])
with col2:
    ride_type = st.selectbox("Ride Type", COMPANY_RIDE_TYPES[company])

# -------- Pickup & Drop --------
st.header("2. Pickup & Drop Locations📍")
col3, col4 = st.columns(2)
with col3:
    pickup = st.selectbox("Pickup Location", list(LOCATIONS_COORDS.keys()))
with col4:
    drop = st.selectbox("Drop Location", list(LOCATIONS_COORDS.keys()))

pickup_lat, pickup_lon = LOCATIONS_COORDS[pickup]
drop_lat, drop_lon = LOCATIONS_COORDS[drop]

distance_km = haversine((pickup_lat, pickup_lon), (drop_lat, drop_lon), unit=Unit.KILOMETERS)
st.info(f"📏 Distance: {distance_km:.2f} km")

# -------- Weather Input --------
st.header("3. Weather Conditions🌦️")
col5, col6, col7 = st.columns(3)
with col5:
    temperature = st.slider("Temperature (°C)", 0.0, 50.0, 25.0)
with col6:
    humidity = st.slider("Humidity (%)", 0.0, 1.0, 0.5)
with col7:
    rainfall = st.number_input("Rainfall (mm)", 0.0, 10.0, 0.0)

# -------- Time Input --------
st.header("4. Ride Time🕒")
col8, col9 = st.columns(2)
with col8:
    date_input = st.date_input("Select Date", datetime.today())
with col9:
    time_input = st.time_input("Select Time", datetime.now().time())

dt = datetime.combine(date_input, time_input)
hour = dt.hour
day = dt.day
month = dt.month
weekday = dt.weekday()
weekend = 1 if weekday >= 5 else 0

# -------- Predict Fare --------
if st.button("Predict Fare"):
    input_data = {
        "distance_km": distance_km,
        "temperature": temperature,
        "humidity": humidity,
        "rainfall": rainfall,
        "cloud_coverage": 0.5,
        "pressure": 1010,
        "wind_speed": 7,
        "hour": hour,
        "day": day,
        "mounth": month,
        "weekday": weekday,
        "weekend": weekend,
        "company": company,
        "ride_type": ride_type,
        "surge": 1.0
    }

    try:
        df_input = create_model_input_df(input_data)
        base_fare = model.predict(df_input)[0]
        base_fare = round(max(base_fare, 2.5), 2)

        surge_multiplier = calculate_surge(hour, rainfall, weekend)
        surge_amount = round(base_fare * (surge_multiplier - 1), 2)
        total_fare = round(base_fare * surge_multiplier, 2)

        st.subheader("💰 Fare Breakdown")
        st.info(f"📌 Base Fare: ${base_fare:.2f}")
        st.warning(f"⚡ Surge Multiplier: x{surge_multiplier:.2f}")
        st.success(f"🔥 Surge Price Added: ${surge_amount:.2f}")
        st.success(f"🚕 Total Fare: ${total_fare:.2f}")

    except Exception as e:
        st.error(f"Prediction Error: {e}")
