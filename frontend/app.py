import streamlit as st
import requests
import os

st.set_page_config(
    page_title="MediPredict AI",
    page_icon="🏥",
    layout="wide"
)

API_URL = os.getenv("API_URL","http://127.0.0.1:8000")

if 'history' not in st.session_state:
    st.session_state.history = []

st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        background-color: #007bff;
        color: white;
    }
    .risk-high {
        color: #dc3545;
        font-weight: bold;
        padding: 10px;
        border-radius: 5px;
        background-color: #ffe6e6;
    }
    .risk-low {
        color: #28a745;
        font-weight: bold;
        padding: 10px;
        border-radius: 5px;
        background-color: #e6ffe6;
    }
    </style>
""", unsafe_allow_html=True)

st.title(" AI Disease Prediction System")
st.markdown(" Advanced Multi-Disease Risk Analysis powered by GenAI")

tab1, tab2, tab3 = st.tabs([" Diabetes", " Heart Disease", " Parkinson's"])

def handle_prediction(endpoint, payload, disease_name):
    try:
        
        with st.spinner("🤖 Analyzing vitals & generating AI report..."):
            response = requests.post(f"{API_URL}/predict/{endpoint}", json=payload)
            
        if response.status_code == 200:
            data = response.json()
            
          
            st.session_state.history.append({
                "disease": disease_name,
                "risk": data['risk_level'],
                "date": "Just now"
            })
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Results")
                if data['risk_level'] == "High":
                    st.markdown(f"<div class='risk-high'>RISK: HIGH ({data['probability']:.1%})</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='risk-low'>RISK: LOW ({data['probability']:.1%})</div>", unsafe_allow_html=True)
            
            with col2:
                st.subheader("AI Analysis")
                st.info(data['ai_analysis'])
                st.caption("⚠️ Disclaimer: This is an AI prediction. Consult a doctor.")
        else:
            st.error(f"Server Error: {response.text}")
            
    except requests.exceptions.ConnectionError:
        st.error(" Could not connect to backend")

# Diabetes
with tab1:
    st.header("Diabetes Risk Assessment")
    col1, col2 = st.columns(2)
    
    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        glucose = st.number_input("Glucose Level", 0, 300, 100)
        bp = st.number_input("Blood Pressure", 0, 200, 70)
        skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
    
    with col2:
        insulin = st.number_input("Insulin Level", 0, 900, 79)
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
        age = st.number_input("Age (Diabetes)", 1, 120, 30)

    if st.button("Analyze Diabetes Risk"):
        payload = {
            "Pregnancies": pregnancies, "Glucose": glucose, "BloodPressure": bp,
            "SkinThickness": skin_thickness, "Insulin": insulin, "BMI": bmi,
            "DiabetesPedigreeFunction": dpf, "Age": age
        }
        handle_prediction("diabetes",payload,"Diabetes")
        
# HeartDisease
with tab2:
    st.header("Heart Disease Risk Assessment")
    col1, col2 = st.columns(2)
    
    with col1:
        h_age = st.number_input("Age (Heart)", 1, 120, 45)
        sex = st.selectbox("Sex", ["M", "F"])
        chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "ASY", "TA"])
        resting_bp = st.number_input("Resting BP", 0, 250, 120)
        cholesterol = st.number_input("Cholesterol", 0, 600, 200)
    
    with col2:
        fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
        resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
        max_hr = st.number_input("Max Heart Rate", 0, 250, 150)
        exercise_angina = st.selectbox("Exercise Angina", ["N", "Y"])
        oldpeak = st.number_input("Oldpeak", -5.0, 10.0, 0.0)
        st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

    if st.button("Analyze Heart Risk"):
        payload = {
            "Age": h_age, "Sex": sex, "ChestPainType": chest_pain, "RestingBP": resting_bp,
            "Cholesterol": cholesterol, "FastingBS": fasting_bs, "RestingECG": resting_ecg,
            "MaxHR": max_hr, "ExerciseAngina": exercise_angina, "Oldpeak": oldpeak, "ST_Slope": st_slope
        }
        handle_prediction("heart",payload,"Heart Disease")
        

# Parkinsons
with tab3:
    st.header("Parkinson's Disease Detection")
    st.markdown("Enter vocal feature measurements:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        mdvp_fo = st.number_input("MDVP:Fo(Hz)", value=119.99)
        mdvp_fhi = st.number_input("MDVP:Fhi(Hz)", value=157.30)
        mdvp_flo = st.number_input("MDVP:Flo(Hz)", value=74.99)
        mdvp_jit_per = st.number_input("MDVP:Jitter(%)", value=0.007, format="%.5f")
        mdvp_jit_abs = st.number_input("MDVP:Jitter(Abs)", value=0.00007, format="%.6f")
        mdvp_rap = st.number_input("MDVP:RAP", value=0.003, format="%.5f")
        mdvp_ppq = st.number_input("MDVP:PPQ", value=0.005, format="%.5f")
        
    with col2:
        jit_ddp = st.number_input("Jitter:DDP", value=0.011, format="%.5f")
        mdvp_shim = st.number_input("MDVP:Shimmer", value=0.043, format="%.5f")
        mdvp_shim_db = st.number_input("MDVP:Shimmer(dB)", value=0.426)
        shim_apq3 = st.number_input("Shimmer:APQ3", value=0.021, format="%.5f")
        shim_apq5 = st.number_input("Shimmer:APQ5", value=0.031, format="%.5f")
        mdvp_apq = st.number_input("MDVP:APQ", value=0.029, format="%.5f")
        shim_dda = st.number_input("Shimmer:DDA", value=0.065, format="%.5f")

    with col3:
        nhr = st.number_input("NHR", value=0.022, format="%.5f")
        hnr = st.number_input("HNR", value=21.0)
        rpde = st.number_input("RPDE", value=0.414)
        dfa = st.number_input("DFA", value=0.815)
        spread1 = st.number_input("spread1", value=-4.81)
        spread2 = st.number_input("spread2", value=0.266)
        d2 = st.number_input("D2", value=2.30)
        ppe = st.number_input("PPE", value=0.284)

    if st.button("Analyze Parkinson's Risk"):
        payload = {
            "MDVP:Fo(Hz)": mdvp_fo, "MDVP:Fhi(Hz)": mdvp_fhi, "MDVP:Flo(Hz)": mdvp_flo,
            "MDVP:Jitter(%)": mdvp_jit_per, "MDVP:Jitter(Abs)": mdvp_jit_abs, "MDVP:RAP": mdvp_rap,
            "MDVP:PPQ": mdvp_ppq, "Jitter:DDP": jit_ddp, "MDVP:Shimmer": mdvp_shim,
            "MDVP:Shimmer(dB)": mdvp_shim_db, "Shimmer:APQ3": shim_apq3, "Shimmer:APQ5": shim_apq5,
            "MDVP:APQ": mdvp_apq, "Shimmer:DDA": shim_dda, "NHR": nhr, "HNR": hnr,
            "RPDE": rpde, "DFA": dfa, "spread1": spread1, "spread2": spread2, "D2": d2, "PPE": ppe
        }
        handle_prediction("parkinson's",payload,"Parkinson's")
        
       