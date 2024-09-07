import streamlit as st
import pickle
import numpy as np
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA

# Loading the model, standard scaler and pca so standardising and pca
# can both be applied to the user's input
loaded_model=pickle.load(open("automobile_price_predict.pkl","rb"))
scaler=pickle.load(open("scaler.pkl","rb"))
pca=pickle.load(open("pca.pkl","rb"))


st.title("Automobile Price Prediction App")
st.header("Enter the Automobile Features:")

symboling=st.number_input("Symboling")
normalised_losses=st.number_input("Normalised Losses")
wheel_base=st.number_input("Wheel Base")
length=st.number_input("length")
width=st.number_input("width")
curb_weight=st.number_input("Curb Weight")
num_of_cylin=st.number_input("Number of Cylinders")
engine_size=st.number_input("Engine Size")
bore=st.number_input("Bore")
horsepower=st.number_input("Horsepower")
peak_rpm=st.number_input("Peak RPM")
highway_mpg=st.number_input("Highway-MPG")
city_mpg=st.number_input("City-MPG")
user_input=np.array([[symboling,normalised_losses,wheel_base,length,width,curb_weight,num_of_cylin,engine_size,bore,horsepower,peak_rpm,highway_mpg,city_mpg]])

if st.button("Predict price"):
    # scaling user inputs and applying PCA
    scaled_input=scaler.transform(user_input)
    pca_input=pca.transform(scaled_input)
    # predicting price using scaled and transformed user input and developed model
    prediction=loaded_model.predict(pca_input)
    st.success(f"The predicted price of the automobile for given parameters is: ${prediction[0]}")