import os

import requests
import streamlit as st

st.title("Multisimmer Prediction Model")

# Form
with st.form("prediction_form"):
    age = st.number_input("Age", min_value=18, max_value=100)
    gndr = st.selectbox("Gender", ["M", "F"])
    trf = st.text_input("Tariff", "M")
    region = st.text_input("Region", "BAKU")
    tenure = st.number_input("Tenure", min_value=0)
    age_dev = st.number_input("Age of Device")
    dev_man = st.text_input("Device Manufacturer", "Samsung")
    device_os_name = st.text_input("OS", "Android OS")
    dev_num = st.number_input("Device Number", min_value=0)
    is_dualsim = st.selectbox("Dual SIM", [0, 1])
    is_smartphone = st.selectbox("Smartphone", [0, 1])
    is_featurephone = st.selectbox("Featurephone", [0, 1])
    simcard_type = st.selectbox("SIM Type", ["2G", "3G", "4G", "5G"])

    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "records": [
            {
                "trf": trf,
                "age": age,
                "gndr": gndr,
                "tenure": tenure,
                "age_dev": age_dev,
                "dev_man": dev_man,
                "device_os_name": device_os_name,
                "dev_num": dev_num,
                "is_dualsim": is_dualsim,
                "is_smartphone": is_smartphone,
                "is_featurephone": is_featurephone,
                "simcard_type": simcard_type,
                "region": region,
            }
        ]
    }

    # response = requests.post("http://127.0.0.1:8000/predict", json=payload)
    backend_url = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
    response = requests.post(f"{backend_url}/predict", json=payload)
    result = response.json()

    st.write("**Prediction Result:**")
    st.json(result)
