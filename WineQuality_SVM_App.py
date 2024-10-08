import streamlit as st
import pickle
import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA

# Loading the model, standard scaler so standardising
# can both be applied to the user's input
loaded_model = pickle.load(open("wine_SVM_predict.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
st.title("Wine Quality Prediction App")
st.header("Enter the parameters for Red Wine:")

fixed_acidity = st.number_input("Fixed Acidity")
volatile_acidity = st.number_input("Volatile Acidity")
citric_acid = st.number_input("Citric Acid")
residual_sugar = st.number_input("Residual Sugar")
chlorides = st.number_input("Chlorides")
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide")
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide")
density = st.number_input("Density")
pH = st.number_input("pH")
sulphates = st.number_input("Sulphates")
alcohol = st.number_input("Alcohol")
user_input = np.array([[fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
                        free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates,
                        alcohol]])

if st.button("Predict Quality"):
    # scaling user inputs and applying PCA
    scaled_input = scaler.transform(user_input)
    prediction = loaded_model.predict(scaled_input)
    st.success(f"Wine Quality is: {prediction[0]}")
