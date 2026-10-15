
import sys
import importlib.util
import os

print("PYTHON VERSION:", sys.version)
print("JOBLIB FOUND:", importlib.util.find_spec("joblib"))
print("REQUIREMENTS FILE EXISTS:", os.path.exists("requirements.txt"))

if os.path.exists("requirements.txt"):
    with open("requirements.txt", "r") as file:
        print("REQUIREMENTS CONTENTS:", file.read())

import streamlit as st
import pandas as pd
import joblib

# Load the trained ML model
model = joblib.load("student_placement_model.pkl")

st.set_page_config(
    page_title="Student Placement Prediction",
    page_icon="🎓",
    layout="centered"
)

st.title("🎓 Student Placement Prediction")
st.write(
    "Enter your academic details to predict your placement status."
)

st.info(
    "This application provides an ML-based prediction, "
    "not a guarantee of actual placement."
)

with st.form("placement_form"):

    st.subheader("Personal Information")

    gender = st.selectbox(
        "Gender", ["Male", "Female"]
    )

    st.subheader("SSC Details")

    ssc_p = st.number_input(
        "SSC Percentage",
        min_value=0.0, max_value=100.0, value=75.0
    )

    ssc_b = st.selectbox(
        "SSC Board", ["Central", "Others"]
    )

    st.subheader("HSC Details")

    hsc_p = st.number_input(
        "HSC Percentage",
        min_value=0.0, max_value=100.0, value=70.0
    )

    hsc_b = st.selectbox(
        "HSC Board", ["Central", "Others"]
    )

    hsc_s = st.selectbox(
        "HSC Stream", ["Commerce", "Science", "Arts"]
    )

    st.subheader("Degree Details")

    degree_p = st.number_input(
        "Degree Percentage",
        min_value=0.0, max_value=100.0, value=75.0
    )

    degree_t = st.selectbox(
        "Degree Field",
        ["Sci&Tech", "Comm&Mgmt", "Others"]
    )

    workex = st.selectbox(
        "Work Experience", ["No", "Yes"]
    )

   st.subheader("Employability & MBA Details")

    etest_p = st.number_input(
        "Employability Test Percentage",
        min_value=0.0, max_value=100.0, value=70.0
    )

    specialisation = st.selectbox(
        "MBA Specialisation",
        ["Mkt&Fin", "Mkt&HR"]
    )

    mba_p = st.number_input(
        "MBA Percentage",
        min_value=0.0, max_value=100.0, value=70.0
    )

    submitted = st.form_submit_button(
        "Predict Placement"
    )

if submitted:

    input_data = pd.DataFrame([{
        "gender": gender,
        "ssc_p": ssc_p,
        "ssc_b": ssc_b,
        "hsc_p": hsc_p,
        "hsc_b": hsc_b,
        "hsc_s": hsc_s,
        "degree_p": degree_p,
        "degree_t": degree_t,
        "workex": workex,
        "etest_p": etest_p,
        "specialisation": specialisation,
        "mba_p": mba_p
    }])

    prediction = model.predict(input_data)[0]

    st.subheader("Prediction Result")

    if prediction == "Placed":
        st.success("Predicted Status: PLACED 🎉")
    else:
        st.warning("Predicted Status: NOT PLACED")

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]
        classes = list(model.classes_)

        if prediction in classes:
            confidence = probabilities[
                classes.index(prediction)
            ] * 100

            st.metric(
                "Model confidence for this prediction",
                f"{confidence:.2f}%"
            )
