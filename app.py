import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_heart
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

# Title and description
st.title("Heart Disease Prediction")
st.write("This app predicts the presence of heart disease using a Linear Regression model.")

# Load the Heart Disease dataset
data = load_heart()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Create and train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test).round()

# Display the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
st.write(f"Model Accuracy: {accuracy * 100:.2f}%")

# Sidebar for user input
st.sidebar.header("User Input")

# Input features
age = st.sidebar.slider("Age", 29, 77, 55)
sex = st.sidebar.selectbox("Sex", ["Male", "Female"])
cp = st.sidebar.slider("Chest Pain Type", 0, 3, 1)
trestbps = st.sidebar.slider("Resting Blood Pressure", 94, 200, 120)
chol = st.sidebar.slider("Cholesterol", 126, 564, 250)
fbs = st.sidebar.selectbox("Fasting Blood Sugar", ["<= 120 mg/dl", "> 120 mg/dl"])
restecg = st.sidebar.slider("Resting Electrocardiographic Results", 0, 2, 1)
thalach = st.sidebar.slider("Maximum Heart Rate Achieved", 71, 202, 150)
exang = st.sidebar.selectbox("Exercise-Induced Angina", ["No", "Yes"])
oldpeak = st.sidebar.slider("ST Depression", 0.0, 6.2, 1.0)
slope = st.sidebar.slider("Slope of the Peak Exercise ST Segment", 0, 2, 1)
ca = st.sidebar.slider("Number of Major Vessels", 0, 4, 2)
thal = st.sidebar.slider("Thalassemia", 0, 3, 2)

# Transform user input into a prediction
input_data = np.array([[
    age,
    0 if sex == "Male" else 1,
    cp,
    trestbps,
    chol,
    0 if fbs == "<= 120 mg/dl" else 1,
    restecg,
    thalach,
    0 if exang == "No" else 1,
    oldpeak,
    slope,
    ca,
    thal
]])

prediction = model.predict(input_data)

st.write(f"Predicted Heart Disease: {'Yes' if prediction[0] == 1 else 'No'}")
