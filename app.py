import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from PIL import Image

st.set_page_config(page_title="Exoplanet Predictor", page_icon=None, layout="wide")

st.markdown("""
<style>
    .stApp {
        background: url("/static/img/stardust.png"), linear-gradient(135deg,#0b0f1a 0%, #0b0f1a 100%);
        background-size: cover;
        color:#e0e0e0;
    }
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.6);  /* <-- adjust opacity here */
        pointer-events: none;
        z-index: 0;
    }
    .main {
        position: relative;
        z-index: 1;
    }
    .title {
        font-weight: 700;
        font-size: 3.2rem;
        letter-spacing: 0.1em;
        text-align: center;
        color: #ffd86e;
        text-shadow: 0 0 10px #ffbb33, 0 0 20px #ffaa00;
        margin-top: 1rem;
    }
    .subtitle {
        font-size: 1.1rem;
        text-align: center;
        color: #ffffffcc;
        margin-bottom: 2rem;
    }
    .section {
        background: rgba(0, 0, 0, 0.65);
        border-radius: 18px;
        padding: 2rem 3rem;
        margin-bottom: 3rem;
        box-shadow: 0 8px 32px 0 rgba(255, 255, 255, 0.1);
    }
    div.stButton > button:first-child {
        background: linear-gradient(45deg, #ffbb33, #ff9900);
        border-radius: 50px;
        padding: 0.75rem 3rem;
        font-size: 1.2rem;
        font-weight: 700;
        color: #111;
        transition: all 0.4s ease;
        box-shadow: 0 0 10px #ffbb33;
    }
    div.stButton > button:first-child:hover {
        background: linear-gradient(135deg, #ffcc66, #ffaa00);
        box-shadow: 0 0 12px #ffcc66, 0 0 20px #ffcc66;
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_html=True)

try:
    logo = Image.open("logo1.png")
    st.image(logo, width=300)
except:
    st.warning("")


st.markdown('<div class="title">NASA Exoplanet Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Input features and predict the status of an exoplanet candidate</div>', unsafe_allow_html=True)
st.sidebar.markdown("🏠 [← Back to Lunatics home](/)")


with st.expander("About This App"):
    st.markdown("""
        This app uses a trained machine learning model on Kepler exoplanet data to classify a celestial object as:
        - Confirmed Exoplanet
        - Candidate
        - False Positive

        You can adjust physical and observational parameters to get a real-time classification.

        **Tech Stack:** Streamlit, Scikit-learn, Pandas, Matplotlib.
    """)

# Main Section
st.markdown('<div class="section">', unsafe_allow_html=True)

FEATURES = [
    "koi_score", "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_prad",
    "koi_teq", "koi_duration", "koi_depth", "koi_insol", "koi_model_snr",
    "koi_steff", "koi_slogg", "koi_srad", "koi_kepmag"
]
label_names = {1: "CONFIRMED EXOPLANET", 0: "CANDIDATE", -1: "FALSE POSITIVE"}

try:
    scaler = joblib.load("scaler.joblib")
    model = joblib.load("model.joblib")
except:
    st.error("Model files not found. Please run the training script first.")
    st.stop()

cols = st.columns(4)
input_vals = {}
for i, feature in enumerate(FEATURES):
    col = cols[i % 4]
    if "fpflag" in feature:
        input_vals[feature] = col.selectbox(feature.replace('_', ' ').title(), [0, 1])
    else:
        input_vals[feature] = col.number_input(
            feature.replace('_', ' ').title(), 
            value=0.5 if feature == 'koi_score' else 1.0, 
            min_value=0.0
        )

if st.button("Predict Exoplanet Status"):
    input_df = pd.DataFrame([input_vals])
    input_scaled = scaler.transform(input_df)
    proba = model.predict_proba(input_scaled)[0]
    pred = model.predict(input_scaled)[0]

    if pred == 1:
        st.balloons()

    st.markdown(f"<h2 style='color:#fdda6e;text-align:center;'>{label_names.get(pred, 'Unknown')}</h2>", unsafe_allow_html=True)

    label_map = {-1: "False Positive", 0: "Candidate", 1: "Confirmed"}
    labels = [label_map[c] for c in model.classes_]
    colors = ['#ff6666', '#33ccff', '#ffcc00']

    fig, ax = plt.subplots(figsize=(6, 2), facecolor='none')  # Transparent figure background

    ax.set_facecolor('none')

    bars = ax.bar(labels, proba * 100, color=colors)
    ax.set_ylim(0, 100)
    ax.set_ylabel('Confidence %', fontsize=10, color='white')
    ax.set_title('Prediction Confidence', fontsize=12, color='white')
    ax.tick_params(axis='both', labelsize=9, colors='white')  # Make ticks white

   
    for bar in bars:
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 1,
            f'{yval:.2f}%',
            ha='center',
            va='bottom',
            fontsize=9,
            color='white'
        )

    
    for spine in ax.spines.values():
        spine.set_visible(False)

    
    st.pyplot(fig, use_container_width=False)

    st.write("Confidence shows the model's certainty for each classification.")

st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<hr style='border: 1px solid #666;'>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#ccc;'>Made by THE LUNATICS. Powered by NASA Kepler mission insights.</p>", unsafe_allow_html=True)
