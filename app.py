import streamlit as st
import numpy as np
from PIL import Image
import gdown, os, cv2
import tensorflow as tf
import random
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Healthcare", layout="wide")

MODEL_PATH = "model.tflite"
FILE_ID = "1CBdRBXsze5YgdbRnC8H3GYtqLlydeF-j"

KERAS_PATH = "model.keras"
KERAS_ID = "1GRO5EwB9PDX61G1lZfIHChvCK7JkYe6v"

CLASS_NAMES = ['COVID19','NORMAL','PNEUMONIA','TURBERCULOSIS']

# ---------------- MODERN UI STYLE ----------------
st.markdown("""
<style>
.main {
    background: linear-gradient(to right, #eef2f3, #ffffff);
}

.card {
    background: white;
    padding: 18px;
    border-radius: 16px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.08);
    margin-bottom: 15px;
}

.metric-card {
    background: linear-gradient(135deg, #4facfe, #00f2fe);
    padding: 15px;
    border-radius: 12px;
    color: white;
    font-weight: bold;
    text-align:center;
}

.title {
    font-size: 32px;
    font-weight: 800;
}

.subtitle {
    font-size: 15px;
    color: gray;
    margin-bottom: 20px;
}

.disease-text {
    font-size: 14px;
    font-weight: 600;
    line-height: 1.7;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="title">🩺 AI Healthcare System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered Disease Detection & Smart Hospital Recommendation</div>', unsafe_allow_html=True)

# ---------------- DISEASE INFO ----------------
DISEASE_INFO = {
    "TURBERCULOSIS": {
        "desc": "Tuberculosis is a serious infectious disease that mainly affects the lungs and spreads through air.",
        "sym": "persistent cough, weight loss, night sweats, fatigue",
        "cause": "Mycobacterium tuberculosis bacteria",
        "treat": "long-term antibiotic therapy (6–9 months)",
        "prec": "early diagnosis, proper medication, avoiding close contact"
    },
    "PNEUMONIA": {
        "desc": "Pneumonia causes inflammation in lung air sacs, often filled with fluid or pus.",
        "sym": "fever, cough, breathing difficulty, chest pain",
        "cause": "bacteria, viruses, or fungi",
        "treat": "antibiotics, rest, oxygen support if needed",
        "prec": "vaccination, hygiene, avoiding smoking"
    },
    "COVID19": {
        "desc": "COVID-19 is a viral respiratory illness affecting lungs and immune system.",
        "sym": "fever, dry cough, breathing issues, fatigue",
        "cause": "SARS-CoV-2 virus",
        "treat": "supportive care, isolation, oxygen if severe",
        "prec": "mask, vaccination, distancing"
    },
    "NORMAL": {
        "desc": "No abnormalities detected in lungs.",
        "sym": "no symptoms",
        "cause": "healthy condition",
        "treat": "not required",
        "prec": "maintain healthy lifestyle"
    }
}

# ---------------- HOSPITALS ----------------
HOSPITALS = {
    "Chennai": [
        {"name":"Apollo Hospital","doc":"Dr. Ramesh"},
        {"name":"MIOT International","doc":"Dr. Priya"}
    ],
    "Coimbatore":[
        {"name":"KG Hospital","doc":"Dr. Vignesh"},
        {"name":"Ganga Hospital","doc":"Dr. Suresh"}
    ]
}

def maps_link(name,city):
    return f"https://www.google.com/maps/search/{name}+{city}"

def rating():
    return round(random.uniform(3.8,5.0),1)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH)
    model = tf.lite.Interpreter(model_path=MODEL_PATH)
    model.allocate_tensors()
    return model

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---------------- SIDEBAR ----------------
st.sidebar.header("👤 Patient Details")
name = st.sidebar.text_input("Name")
age = st.sidebar.number_input("Age",0,120)
uploaded = st.sidebar.file_uploader("Upload X-ray")

# ---------------- MAIN ----------------
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    img_r = img.resize((224,224))
    arr = np.expand_dims(np.array(img_r)/255,0).astype('float32')

    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])

    disease = CLASS_NAMES[np.argmax(preds[0])]
    conf = np.max(preds[0])*100

    # ---------------- METRICS ----------------
    colA, colB = st.columns(2)

    with colA:
        st.markdown(f'<div class="metric-card">🧠 Prediction<br><br>{disease}</div>', unsafe_allow_html=True)

    with colB:
        st.markdown(f'<div class="metric-card">📊 Confidence<br><br>{conf:.2f}%</div>', unsafe_allow_html=True)

    # ---------------- MAIN LAYOUT ----------------
    col1, col2 = st.columns([1,1])

    # -------- LEFT: IMAGE + CHART --------
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.image(img, use_container_width=True)

        st.markdown("### 📊 Prediction Distribution")

        fig, ax = plt.subplots(figsize=(3,3))
        ax.bar(CLASS_NAMES, preds[0])
        ax.set_xticklabels(CLASS_NAMES, rotation=45)

        st.pyplot(fig)

        st.markdown('</div>', unsafe_allow_html=True)

    # -------- RIGHT: DISEASE INFO --------
    info = DISEASE_INFO[disease]

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📖 Disease Insights")

        st.markdown(f"""
        <div class="disease-text">
        <b>Overview:</b> {info['desc']}<br><br>
        <b>Symptoms:</b> {info['sym']}<br><br>
        <b>Causes:</b> {info['cause']}<br><br>
        <b>Treatment:</b> {info['treat']}<br><br>
        <b>Precautions:</b> {info['prec']}<br><br>
        <b>Severity:</b> {"High ⚠️" if disease != "NORMAL" else "None ✅"}
        </div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # ---------------- HOSPITAL ----------------
    st.markdown("## 🏥 Recommended Hospitals")

    city = st.text_input("Enter your city")

    if city:
        city_clean = city.strip().title()

        if city_clean in HOSPITALS:
            cols = st.columns(2)

            for i, h in enumerate(HOSPITALS[city_clean]):
                with cols[i % 2]:
                    st.markdown(f"""
                    <div class="card">
                    <b>🏥 {h['name']}</b><br><br>
                    👨‍⚕️ {h['doc']}<br><br>
                    ⭐ {rating()}/5<br><br>
                    <a href="{maps_link(h['name'], city_clean)}" target="_blank">📍 Open Map</a>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("No hospitals found")
