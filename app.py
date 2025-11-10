import streamlit as st
import pandas as pd
import joblib

model = joblib.load('KNN_heart.pkl')
scaler = joblib.load('scaler.pkl')
expected_columns = joblib.load('columns.pkl')  

st.title("Heart Disease Prediction App (🤗Know your Heart ❤)")
st.markdown("Provide the following details")

age = st.slider("Age",18,100,40)
sex = st.selectbox("SEX",['M','F'])
chest_pain = st.selectbox("Chest Pain Type",['ATA','NAP','TA','ASY'])
resting_bp =st.number_input("Resting Blood Pressure (mmHg)",80,200,120)
cholestrol = st.number_input("Cholestrol (mg/dL)",100,600,200)
fasting_bs = st.selectbox("Fasting Blood Sugar >120 mg/dL",['0','1'])
resting_ecg = st.selectbox("Resting Electrocardiographic Results",['Normal','ST','LVH'])
max_Hr = st.number_input("Maximum Heart Rate Achieved",60,220,150)
exercise_angina = st.selectbox("Exercise Induced Angina",['Y','N'])
oldpeak = st.slider("ST Depression Induced by Exercise Relative to Rest",0.0,6.0,1.0)
st_slope = st.selectbox("Slope of the Peak Exercise ST Segment",['Up','Flat','Down'])

raw_Input = {
    'Age':age,
    'RestingBP':resting_bp,
    'Cholesterol':cholestrol,
    'FastingBS':fasting_bs,
    'MaxHR':max_Hr,
    'Oldpeak':oldpeak,
    'Sex_'+ sex :1,
    'ChestPainType_'+chest_pain:1,
    'RestingECG_'+resting_ecg:1,
    'ExerciseAngina_'+exercise_angina:1,
    'ST_Slope_'+st_slope:1
}

input_df = pd.DataFrame([raw_Input])

for col in expected_columns:
  if col not in input_df.columns:
    input_df[col] = 0

input_df = input_df[expected_columns]

scaled_input = scaler.transform(input_df)
prediction = model.predict(scaled_input)[0]

if prediction == 1:
  st.error("Heart Disease Detected")
else:
  st.success("No Heart Disease Detected")

  st.write("Thank you for visiting — take care of your heart ❤")
