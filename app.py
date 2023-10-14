import streamlit as st
import numpy as np

# Function for Linear Regression
def linear_regression_predict(coefficients, feature_vector):
    """
    Perform linear regression prediction based on the coefficients and feature vector.

    Parameters:
        coefficients (list or numpy.array): Coefficients of the linear regression model.
        feature_vector (list or numpy.array): Input features for prediction.

    Returns:
        float: Predicted value.
    """
    return np.dot(coefficients, feature_vector)

# Streamlit interface
st.title("Heart Disease Prediction")
st.write("Enter the following information to predict heart disease:")

# Create input fields for user to enter data
age = st.number_input("Age", min_value=1, max_value=100)
sex = st.selectbox("Sex", ['Male', 'Female'])
cp = st.selectbox("Chest Pain Type", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=200)
chol = st.number_input("Cholesterol (mg/dl)", min_value=50, max_value=400)
fbs = st.selectbox("Fasting Blood Sugar", [0, 1])
restecg = st.selectbox("Resting Electrocardiographic Results", [0, 1, 2])
thalach = st.number_input("Maximum Heart Rate Achieved (bpm)", min_value=70, max_value=200)
exang = st.selectbox("Exercise-Induced Angina", [0, 1])
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=6.2)
slope = st.selectbox("Slope of Peak Exercise ST Segment", [0, 1, 2])
ca = st.number_input("Number of Major Vessels Colored by Fluoroscopy", min_value=0, max_value=4)
thal = st.selectbox("Thalassemia Type", [0, 1, 2, 3])

# Define example coefficients for linear regression - Replace with your actual model coefficients
coefficients = np.array([0.1, 0.2, 0.3, -0.1, 0.05, -0.2, 0.1, -0.15, -0.05, 0.2, -0.1, 0.05, -0.2])

# Create a button to make predictions
if st.button("Predict"):
    # Prepare user input as a feature vector
    feature_vector = [age, 1 if sex == 'Male' else 0, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    
    # Calculate the prediction using the linear_regression_predict function
    prediction = linear_regression_predict(coefficients, feature_vector)

    st.write(f"Predicted Heart Disease Probability: {prediction:.2f}")

# Note: You should replace the 'coefficients' array with the actual coefficients from your linear regression model.

