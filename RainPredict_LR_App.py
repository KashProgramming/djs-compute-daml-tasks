import streamlit as st
import pickle
import numpy as np
import pandas as pd
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA

# Loading the model, standard scaler and pca so standardising and pca
# can both be applied to the user's input
loaded_model = pickle.load(open("rain_LR_predict.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
pca = pickle.load(open("pca.pkl", "rb"))
dummy_columns = pickle.load(open("dummy_columns.pkl", "rb"))
st.title("Rain Prediction App")
st.header("Enter the Weather parameters:")

min_temp = st.number_input("Minimum Temperature")
max_temp = st.number_input("Maximum Temperature")
rainfall = st.number_input("Rainfall (mm)")
wind_gust_speed = st.number_input("Wind Gust Speed")
wind_speed_9am = st.number_input("Wind Speed at 9am")
wind_speed_3pm = st.number_input("Wind Speed at 3pm")
humidity_9am = st.number_input("Humidity at 9am")
humidity_3pm = st.number_input("Humidity at 3pm")
pressure_9am = st.number_input("Pressure at 9am")
pressure_3pm = st.number_input("Pressure at 3pm")
temp_9am = st.number_input("Temperature at 9am")
temp_3pm = st.number_input("Temperature at 3pm")
rain_today = st.number_input("Rain Today?", value=0)
risk_mm = st.number_input("RISK_MM")
wind_gust_dir = st.selectbox("Wind Gust Direction", ["E", "ENE", "ESE", "N", "NE", "NNE", "NNW", "NW", "S", "SE", "SSE", "SSW", "W", "WNW", "WSW"])

if st.button("Predict Rain"):
    # Creating a dataframe to store all the input data together
    input_data = pd.DataFrame({
        "MinTemp": [min_temp],
        "MaxTemp": [max_temp],
        "Rainfall": [rainfall],
        "WindGustDir": [wind_gust_dir],
        "WindGustSpeed": [wind_gust_speed],
        "WindSpeed9am": [wind_speed_9am],
        "WindSpeed3pm": [wind_speed_3pm],
        "Humidity9am": [humidity_9am],
        "Humidity3pm": [humidity_3pm],
        "Pressure9am": [pressure_9am],
        "Pressure3pm": [pressure_3pm],
        "Temp9am": [temp_9am],
        "Temp3pm": [temp_3pm],
        "RainToday": [rain_today],
        "RISK_MM": [risk_mm]
    })
    # Encoding the direction input using OneHotEncoding
    input_encoded = pd.get_dummies(input_data)
    input_encoded = input_encoded.reindex(columns=dummy_columns, fill_value=0)
    # scaling user inputs and applying PCA
    scaled_input = scaler.transform(input_encoded)
    pca_input = pca.transform(scaled_input)
    # predicting price using scaled and transformed user input and developed model
    prediction = loaded_model.predict(pca_input)
    if prediction[0] == 1:
        st.success("It will rain tomorrow.")
    else:
        st.success("It will not rain tomorrow.")
