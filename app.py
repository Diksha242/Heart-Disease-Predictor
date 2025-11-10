import streamlit as st
import pandas as pd
import joblib

# === Load your trained model & assets ===
model = joblib.load('models/KNN_heart.pkl')
scaler = joblib.load('models/scaler.pkl')
expected_columns = joblib.load('models/columns.pkl')

# === App Title & Description ===
st.set_page_config(page_title="Heart Disease Predictor ❤️", page_icon="❤️", layout="centered")

st.markdown("<h1 style='text-align:center; color:#FF4B4B;'>❤️ Heart Disease Prediction App</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Know your heart's health — enter the details below 💓</p>", unsafe_allow_html=True)
st.markdown("---")

# === User Inputs (organized neatly) ===
col1, col2 = st.columns(2)

with col1:
    age = st.slider("Age", 18, 100, 40)
    sex = st.selectbox("Sex", ['M', 'F'])
    chest_pain = st.selectbox("Chest Pain Type", ['ATA', 'NAP', 'TA', 'ASY'])
    resting_bp = st.number_input("Resting Blood Pressure (mmHg)", 80, 200, 120)
    cholestrol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
    fasting_bs = st.selectbox("Fasting Blood Sugar >120 mg/dL", ['0', '1'])

with col2:
    resting_ecg = st.selectbox("Resting ECG Results", ['Normal', 'ST', 'LVH'])
    max_Hr = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise Induced Angina", ['Y', 'N'])
    oldpeak = st.slider("ST Depression (Oldpeak)", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("Slope of the Peak Exercise ST Segment", ['Up', 'Flat', 'Down'])

# === Prepare input data ===
raw_Input = {
    'Age': age,
    'RestingBP': resting_bp,
    'Cholesterol': cholestrol,
    'FastingBS': fasting_bs,
    'MaxHR': max_Hr,
    'Oldpeak': oldpeak,
    'Sex_' + sex: 1,
    'ChestPainType_' + chest_pain: 1,
    'RestingECG_' + resting_ecg: 1,
    'ExerciseAngina_' + exercise_angina: 1,
    'ST_Slope_' + st_slope: 1
}

input_df = pd.DataFrame([raw_Input])

# Add missing columns
for col in expected_columns:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[expected_columns]

# === Prediction Section ===
st.markdown("---")
if st.button("🔍 Predict Now"):
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("🚨 **Heart Disease Detected!** Please consult a doctor immediately. 💔")
    else:
        st.success("🎉 **No Heart Disease Detected!** Keep maintaining a healthy lifestyle. ❤️")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center; color:#FF69B4;'>💖 Thank you for visiting — take care of your heart 💖</h4>", unsafe_allow_html=True)

# === Footer ===
st.markdown("---")
st.caption("Built with ❤️ using Streamlit and Machine Learning by Diksha242.")
