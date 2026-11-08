"""Streamlit app: Wellness Tourism Package purchase prediction."""
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "best_model.joblib"


@st.cache_resource
def load_model():
    artifact = joblib.load(MODEL_PATH)
    return artifact["model"], artifact["threshold"]


st.set_page_config(page_title="Wellness Tourism Package Predictor", page_icon="🧳")
st.title("🧳 Wellness Tourism Package Predictor")
st.write("Enter customer details to estimate whether they will purchase the package.")

if not MODEL_PATH.exists():
    st.error("Model file not found. Run the GitHub Actions pipeline to train and commit it.")
    st.stop()

model, threshold = load_model()

c1, c2 = st.columns(2)
with c1:
    age = st.number_input("Age", 18, 100, 35)
    type_of_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Enquiry"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly Income", 1000, 200000, 20000, step=500)
    passport = st.selectbox("Holds a valid passport?", ["Yes", "No"])
    own_car = st.selectbox("Owns a car?", ["Yes", "No"])
with c2:
    persons = st.number_input("Number of Persons Visiting", 1, 10, 3)
    children = st.number_input("Number of Children Visiting (under 5)", 0, 5, 1)
    star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    trips = st.number_input("Average Trips per Year", 0, 25, 3)
    product = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    duration = st.number_input("Duration of Pitch (minutes)", 1, 130, 15)
    followups = st.number_input("Number of Follow-ups", 0, 10, 3)
    satisfaction = st.slider("Pitch Satisfaction Score", 1, 5, 3)

input_df = pd.DataFrame([{
    "Age": age, "TypeofContact": type_of_contact, "CityTier": city_tier,
    "DurationOfPitch": duration, "Occupation": occupation, "Gender": gender,
    "NumberOfPersonVisiting": persons, "NumberOfFollowups": followups,
    "ProductPitched": product, "PreferredPropertyStar": star, "MaritalStatus": marital_status,
    "NumberOfTrips": trips, "Passport": int(passport == "Yes"),
    "PitchSatisfactionScore": satisfaction, "OwnCar": int(own_car == "Yes"),
    "NumberOfChildrenVisiting": children, "Designation": designation,
    "MonthlyIncome": monthly_income,
}])

if st.button("Predict"):
    prob = float(model.predict_proba(input_df)[:, 1][0])
    if prob >= threshold:
        st.success(f"Likely to purchase the Wellness Tourism Package (probability {prob:.1%}).")
    else:
        st.warning(f"Unlikely to purchase (probability {prob:.1%}).")
    with st.expander("Input data"):
        st.dataframe(input_df)
