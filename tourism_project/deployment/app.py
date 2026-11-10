"""Streamlit app: customer purchase-propensity scoring BEFORE sales contact."""
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

HERE = Path(__file__).parent
MODEL_PATH, META_PATH = HERE / "best_model.joblib", HERE / "model_metadata.json"


@st.cache_resource
def load_artifacts():
    artifact = joblib.load(MODEL_PATH)
    meta = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
    sha = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()
    return artifact["model"], artifact["threshold"], artifact["features"], meta, sha


st.set_page_config(page_title="Tourism Package Propensity", page_icon="🧳")
st.title("🧳 Tourism Package Purchase Propensity")
st.info(
    "**Model purpose:** estimates how likely a customer is to buy a tourism package, using only "
    "information available **before** the first sales contact. It was trained on historical "
    "package purchases (`ProdTaken`), which are not labelled by package type, so it is a proxy "
    "for Wellness Tourism Package interest. Use it to prioritise leads, and validate it on "
    "Wellness campaign results before relying on it."
)

if not MODEL_PATH.exists():
    st.error("Model file not found. Run the GitHub Actions pipeline to train and commit it.")
    st.stop()

model, threshold, features, meta, sha = load_artifacts()

c1, c2 = st.columns(2)
with c1:
    age = st.number_input("Age", 18, 100, 35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly Income (gross)", 1000, 200000, 20000, step=500)
    city_tier = st.selectbox("City Tier (1 = most developed)", [1, 2, 3])
with c2:
    passport = st.selectbox("Holds a valid passport?", ["Yes", "No"])
    own_car = st.selectbox("Owns a car?", ["Yes", "No"])
    persons = st.number_input("Number of Persons Visiting", 1, 10, 3)
    children = st.number_input("Children Visiting (under 5)", 0, 5, 0)
    star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])
    trips = st.number_input("Average Trips per Year", 0, 50, 3)

input_df = pd.DataFrame([{
    "Age": age, "CityTier": city_tier, "Occupation": occupation, "Gender": gender,
    "NumberOfPersonVisiting": persons, "PreferredPropertyStar": star,
    "MaritalStatus": marital_status, "NumberOfTrips": trips, "Passport": int(passport == "Yes"),
    "OwnCar": int(own_car == "Yes"), "NumberOfChildrenVisiting": children,
    "Designation": designation, "MonthlyIncome": monthly_income,
}])[features]  # exact training feature order

problems = []
if children > persons:
    problems.append("Children visiting cannot exceed the number of persons visiting.")
if monthly_income < 1000:
    problems.append("Monthly income looks too low.")

if st.button("Predict"):
    if problems:
        for msg in problems:
            st.error(msg)
    else:
        prob = float(model.predict_proba(input_df)[:, 1][0])
        if prob >= threshold:
            st.success(f"High propensity: prioritise this lead (score {prob:.1%}, threshold {threshold:.0%}).")
        else:
            st.warning(f"Lower propensity: lower priority (score {prob:.1%}, threshold {threshold:.0%}).")
        with st.expander("Input data sent to the model"):
            st.dataframe(input_df)

with st.expander("Model details and deployment check"):
    st.write(f"Loaded model file SHA-256: `{sha[:12]}`")
    expected = meta.get("model_sha256")
    if expected:
        st.write("Matches the model recorded in the repository metadata." if expected == sha
                 else "WARNING: model file does not match the recorded metadata.")
    if meta:
        t = meta["test_metrics"]
        st.write(f"Trained (UTC): {meta['trained_at_utc']}")
        st.write(f"Held-out test: ROC-AUC {t['roc_auc']:.2f}, PR-AUC {t['pr_auc']:.2f}, "
                 f"recall {t['recall']:.2f}, precision {t['precision']:.2f} at threshold {threshold:.2f}")
