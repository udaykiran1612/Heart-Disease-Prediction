import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

# Load the heart disease dataset
data = pd.read_csv('heart_disease.csv')

# Split the data into features and target
X = data.drop('target', axis=1)
y = data['target']

# Train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Create the Streamlit app
st.title('Heart Disease Prediction App')

# Create a sidebar to collect user input
st.sidebar.header('User Input')

# Add input fields for each feature
age = st.sidebar.number_input('Age')
sex = st.sidebar.selectbox('Sex', ['Male', 'Female'])
cp = st.sidebar.selectbox('Chest pain type', ['Typical Angina', 'Atypical Angina', 'Non-anginal Pain', 'Asymptomatic'])
trtbps = st.sidebar.number_input('Resting blood pressure (in mm Hg)')
chol = st.sidebar.number_input('Cholestoral in mg/dl fetched via BMI sensor')
fbs = st.sidebar.selectbox('Fasting blood sugar (> 120 mg/dl)', ['True', 'False'])
restecg = st.sidebar.selectbox('Resting electrocardiographic results', ['Normal', 'ST-T wave normality', 'Left ventricular hypertrophy'])
thalachh = st.sidebar.number_input('Maximum heart rate achieved')
oldpeak = st.sidebar.number_input('Previous peak')
slp = st.sidebar.selectbox('Slope', ['Upsloping', 'Flat', 'Downsloping'])
caa = st.sidebar.number_input('Number of major vessels')
thall = st.sidebar.selectbox('Thalium Stress Test result ~ (0,3)', ['3', '2', '6', '0'])
exng = st.sidebar.selectbox('Exercise induced angina ~ 1 = Yes, 0 = No', ['Yes', 'No'])

# Make a prediction if the user clicks the "Predict" button
if st.sidebar.button('Predict'):

    # Create a dataframe from the user input
    user_input = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'cp': [cp],
        'trtbps': [trtbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalachh': [thalachh],
        'oldpeak': [oldpeak],
        'slp': [slp],
        'caa': [caa],
        'thall': [thall],
        'exng': [exng]
    })

    # Make a prediction
    prediction = model.predict(user_input)[0]

    # Display the prediction to the user
    st.write('**Prediction:** {}'.format(prediction))
