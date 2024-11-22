import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Function to Load Data
@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv('HeartDiseaseTrain-Test (1).csv')
    return df

# Function to Train and Save the Model
def train_and_save_model(df):
    # Drop unnecessary columns
    X = df.drop('target', axis=1)
    y = df['target']

    # Preprocessing: Label Encoding for categorical features
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()
    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('scaler', StandardScaler(), numerical_features),
            ('labelencoder', LabelEncoder(), categorical_features)
        ],
        remainder='passthrough'
    )

    # Create a pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train the model
    pipeline.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(pipeline, 'heart_disease_model.pkl')

    # Evaluate the model
    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions)

    return accuracy, report

# Function to Load the Trained Model
@st.cache(allow_output_mutation=True)
def load_model():
    model = joblib.load('heart_disease_model.pkl')
    return model

# Streamlit Web Interface
st.title("Heart Disease Prediction App")

# Step 1: Load Data
df = load_data()
st.subheader("Dataset Preview")
st.write(df.head())

# Step 2: Visualize Data
st.write("### Target Distribution")
fig, ax = plt.subplots()
sns.countplot(x='target', data=df, ax=ax)
ax.set_title("Presence vs Absence of Heart Disease")
st.pyplot(fig)

# Step 3: Train the Model
st.subheader("Model Training")
if st.button("Train Model"):
    accuracy, report = train_and_save_model(df)
    st.success(f"Model Trained Successfully! Accuracy: {accuracy:.2f}")
    st.text("Classification Report:")
    st.text(report)

# Step 4: Load the Trained Model
model = None
try:
    model = load_model()
except:
    st.warning("Please train the model first by clicking the 'Train Model' button.")

# Step 5: Make Predictions
st.subheader("Make a Prediction")
if model:
    # Collect user input for prediction
    age = st.number_input('Age', min_value=0, max_value=120, value=50)
    sex = st.selectbox('Sex', ['Male', 'Female'])
    chest_pain_type = st.selectbox('Chest Pain Type', ['Typical angina', 'Atypical angina', 'Non-anginal pain', 'Asymptomatic'])
    resting_bp = st.number_input('Resting Blood Pressure', min_value=0, value=120)
    chol = st.number_input('Cholesterol', min_value=0, value=200)
    fasting_bs = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ['Yes', 'No'])
    rest_ecg = st.selectbox('Resting ECG', ['Normal', 'ST-T wave abnormality', 'Left ventricular hypertrophy'])
    max_hr = st.number_input('Max Heart Rate Achieved', min_value=0, value=150)
    exercise_angina = st.selectbox('Exercise-Induced Angina', ['Yes', 'No'])
    oldpeak = st.number_input('ST Depression', min_value=0.0, value=1.0)
    slope = st.selectbox('Slope of the Peak Exercise', ['Upsloping', 'Flat', 'Downsloping'])
    vessels = st.selectbox('Number of Major Vessels', [0, 1, 2, 3, 4])
    thal = st.selectbox('Thalassemia', ['Normal', 'Fixed defect', 'Reversible defect'])

    if st.button("Predict Heart Disease"):
        # Prepare input data for prediction
        input_data = pd.DataFrame({
            'age': [age],
            'sex': [sex],
            'chest_pain_type': [chest_pain_type],
            'resting_blood_pressure': [resting_bp],
            'cholestoral': [chol],
            'fasting_blood_sugar': [fasting_bs],
            'rest_ecg': [rest_ecg],
            'Max_heart_rate': [max_hr],
            'exercise_induced_angina': [exercise_angina],
            'oldpeak': [oldpeak],
            'slope': [slope],
            'vessels_colored_by_flourosopy': [vessels],
            'thalassemia': [thal]
        })

        try:
            # Use the trained pipeline to transform and predict
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0][1]

            # Display result
            if prediction == 1:
                st.error(f"*Heart Disease Detected!* (Probability: {prediction_proba:.2f})")
            else:
                st.success(f"*No Heart Disease Detected.* (Probability: {1 - prediction_proba:.2f})")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
