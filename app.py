import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

# Create a Streamlit web app
st.title("Heart Disease Prediction with Linear Regression")

# Load the dataset
@st.cache
def load_data():
    data = pd.read_csv("heart_disease_data.csv")
    return data

data = load_data()

# Data preprocessing
X = data.drop("target", axis=1)
y = data["target"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# User input for feature values
st.sidebar.header("User Input:")
age = st.sidebar.slider("Age:", min_value=29, max_value=77, value=45)
sex = st.sidebar.selectbox("Sex:", ["Male", "Female"])
cp = st.sidebar.slider("Chest Pain Type:", min_value=0, max_value=3, value=0)
trestbps = st.sidebar.slider("Resting Blood Pressure:", min_value=94, max_value=200, value=130)
chol = st.sidebar.slider("Cholesterol Level:", min_value=126, max_value=564, value=240)
fbs = st.sidebar.selectbox("Fasting Blood Sugar:", [0, 1])
restecg = st.sidebar.selectbox("Resting ECG:", [0, 1, 2])
thalach = st.sidebar.slider("Maximum Heart Rate:", min_value=71, max_value=202, value=160)
exang = st.sidebar.selectbox("Exercise-Induced Angina:", [0, 1])
oldpeak = st.sidebar.slider("ST Depression:", min_value=0.0, max_value=6.2, value=1.0)
slope = st.sidebar.slider("Slope of ST Segment:", min_value=0, max_value=2, value=1)
ca = st.sidebar.slider("Number of Major Vessels:", min_value=0, max_value=3, value=0)
thal = st.sidebar.slider("Thalassemia Type:", min_value=0, max_value=3, value=2)

# Transform user input into a DataFrame for prediction
user_input = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'cp': [cp],
    'trestbps': [trestbps],
    'chol': [chol],
    'fbs': [fbs],
    'restecg': [restecg],
    'thalach': [thalach],
    'exang': [exang],
    'oldpeak': [oldpeak],
    'slope': [slope],
    'ca': [ca],
    'thal': [thal]
})

# Make a prediction
prediction = model.predict(user_input)

# Convert the prediction to a binary result
prediction_result = "Heart Disease" if prediction[0] > 0.5 else "No Heart Disease"

# Calculate the accuracy score
y_pred = model.predict(X_test)
y_pred_binary = [1 if pred >= 0.5 else 0 for pred in y_pred]
accuracy = accuracy_score(y_test, y_pred_binary)

st.header("Prediction:")
st.write(f"The model predicts that the patient has: {prediction_result}")
st.write("Accuracy Score:", accuracy)
