import streamlit as st
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import joblib

st.set_page_config(page_title="Tourism Package Prediction", page_icon="✈️", layout="wide")

@st.cache_resource
def load_production_model():
    try:
        model_path = hf_hub_download(
            repo_id="singhpayal/tourism_package_prediction_model",
            filename="top_tourism_model_v1.joblib"
        )
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error initializing production engine: {e}")
        return None

model = load_production_model()

st.title("✈️ Intelligent Wellness Tourism Purchase Predictor")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Customer Base Features")
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    type_of_contact = st.selectbox("Type of Contact", options=["Self Enquiry", "Company Invited"])
    city_tier = st.selectbox("City Tier", options=[1, 2, 3])
    occupation = st.selectbox("Occupation", options=["Salaried", "Small Business", "Free Lancer", "Large Business"])
    gender = st.selectbox("Gender", options=["Male", "Female"])
    marital_status = st.selectbox("Marital Status", options=["Single", "Married", "Divorced", "Unmarried"])
    designation = st.selectbox("Designation", options=["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly Income (₹)", min_value=0.0, value=25000.0)

with col2:
    st.subheader("Engagement Metrics")
    duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=0.0, value=15.0)
    number_of_persons_visiting = st.number_input("Number of Persons Visiting", min_value=1, value=2)
    number_of_followups = st.number_input("Number of Follow-ups", min_value=0.0, value=3.0)
    product_pitched = st.selectbox("Product Pitched", options=["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    preferred_property_star = st.selectbox("Preferred Property Star Rating", options=[3.0, 4.0, 5.0])
    number_of_trips = st.number_input("Number of Trips (per year)", min_value=0.0, value=3.0)
    passport = st.selectbox("Has Passport?", options=["Yes", "No"])
    pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", min_value=1, max_value=5, value=3)
    own_car = st.selectbox("Owns Car?", options=["Yes", "No"])
    number_of_children_visiting = st.number_input("Number of Children Visiting", min_value=0.0, value=0.0)

# Build DataFrame using RAW values—letting the Scikit-Learn Pipeline do the work!
input_data = pd.DataFrame([{
    'Age': age,
    'TypeofContact': type_of_contact,
    'CityTier': city_tier,
    'DurationOfPitch': duration_of_pitch,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': number_of_persons_visiting,
    'NumberOfFollowups': number_of_followups,
    'ProductPitched': product_pitched,
    'PreferredPropertyStar': preferred_property_star,
    'MaritalStatus': marital_status,
    'NumberOfTrips': number_of_trips,
    'Passport': 1 if passport == "Yes" else 0,
    'PitchSatisfactionScore': pitch_satisfaction_score,
    'OwnCar': 1 if own_car == "Yes" else 0,
    'NumberOfChildrenVisiting': number_of_children_visiting,
    'Designation': designation,
    'MonthlyIncome': monthly_income
}])

if st.button("Run Purchase Prediction", use_container_width=True):
    if model is not None:
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        st.markdown("### 📊 Prediction Result")
        if prediction == 1:
          st.markdown(
              f'<div class="prediction-box success-box">'
              f'<h2 style="color: #155724;">🎯 High Likelihood of Purchase!</h2>'
              f'<p style="font-size: 18px; color: #155724;">This customer is <b>likely to purchase</b> the Wellness Tourism Package.</p>'
              f'<p style="font-size: 16px; color: #155724;">Confidence: <b>{prediction_proba[1]*100:.2f}%</b></p>'
              f'</div>',
              unsafe_allow_html=True
              )
          st.success("💡 **Recommendation:** Prioritize follow-up with this customer!")
        else:
          st.markdown(
              f'<div class="prediction-box warning-box">'
              f'<h2 style="color: #856404;">⚠️ Low Likelihood of Purchase</h2>'
              f'<p style="font-size: 18px; color: #856404;">This customer is <b>unlikely to purchase</b> the Wellness Tourism Package.</p>'
              f'<p style="font-size: 16px; color: #856404;">Confidence: <b>{prediction_proba[0]*100:.2f}%</b></p>'
              f'</div>',
              unsafe_allow_html=True
              )
          st.info("💡 **Recommendation:** Consider alternative packages or additional engagement strategies.")
        col_prob1, col_prob2 = st.columns(2)
        with col_prob1:
          st.metric("Probability of No Purchase", f"{prediction_proba[0]*100:.2f}%")
        with col_prob2:
          st.metric("Probability of Purchase", f"{prediction_proba[1]*100:.2f}%")
