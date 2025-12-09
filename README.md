🚖 Cab Surge Price Prediction

Machine Learning project to predict surge prices for cab rides using real-world features like distance, weather, ride type, cab company, and more.

📌 Project Overview

Surge pricing changes dynamically based on demand, supply, weather, ride types, and other factors.
This project builds an ML model to predict cab surge price using:

Distance (km)

Ride Type

Cab Company

Pickup & Drop Locations

Weather Conditions (Temperature, Cloud Cover, Rainfall, Wind Speed)

Surge Multiplier

🧠 Machine Learning Workflow
1️⃣ Data Cleaning & Preprocessing

Handle missing values

Convert categorical columns

Create distance_km feature

Map your custom rename dictionary:

'distance': 'distance_km',
'cab_type': 'company',
'destination': 'drop_location',
'source': 'pickup_location',
'surge_multiplier': 'surge',
'name': 'ride_type',
'temp': 'temperature',
'clouds': 'cloud_coverage',
'rain': 'rainfall',
'wind': 'wind_speed'

| Model    | RMSE              | MAE               | R²                 |
| -------- | ----------------- | ----------------- | ------------------ |
| XGBoost  | 1.2284188270568848| 2.023281371231305 | 0.9526142477989197 |
