"""Streamlit interface for the trained L&T Realty price prediction model."""
import json
import os
import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "artifacts/lt_realty_price_model.pkl"
META_PATH = "artifacts/model_metadata.json"

st.set_page_config(page_title="L&T Realty Price Estimator", page_icon="🏠", layout="wide")
st.title("🏠 L&T Realty — House Price Prediction")
st.caption("Machine learning prototype for educational use")

if not os.path.exists(MODEL_PATH) or not os.path.exists(META_PATH):
    st.error("Trained model files were not found.")
    st.markdown("Run `python train_model.py --data train.csv` first, then restart the app.")
    st.stop()

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

with open(META_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

model = load_model()
features = metadata["features"]
numeric_features = metadata.get("numeric_features", [])
categorical_features = metadata.get("categorical_features", [])

st.info(
    "This is a student project prototype. It is not an official L&T Realty valuation tool. "
    "The dataset used for training may be synthetic and may not represent actual L&T Realty sales."
)

with st.form("property_form"):
    st.subheader("Enter property details")
    cols = st.columns(2)
    values = {}

    for i, feature in enumerate(features):
        with cols[i % 2]:
            if feature in numeric_features:
                label = feature.replace("_", " ").title()
                # Generic, safe form control; exact ranges depend on the dataset.
                values[feature] = st.number_input(
                    label, value=0.0, step=1.0, key=f"num_{feature}"
                )
            else:
                label = feature.replace("_", " ").title()
                values[feature] = st.text_input(
                    label, value="", key=f"cat_{feature}",
                    help="Enter a category exactly as represented in the training data when possible."
                )

    submitted = st.form_submit_button("Predict property price", type="primary")

if submitted:
    row = pd.DataFrame([{f: values[f] for f in features}], columns=features)
    try:
        prediction = float(model.predict(row)[0])
        st.subheader("Estimated price")
        st.metric("Estimated house price (INR)", f"₹{prediction:,.0f}")
        st.caption(
            f"Original dataset target: {metadata.get('target', 'unknown')}. "
            f"Displayed unit: INR rupees (₹). "
            f"{metadata.get('metrics', {}).get('target_unit_note', '')}"
        )
        st.warning(
            "This is a model estimate, not a guarantee or formal valuation. "
            "Accuracy depends on the dataset, location coverage, and market date."
        )
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        st.write("Check that the inputs match the dataset's expected columns and types.")
