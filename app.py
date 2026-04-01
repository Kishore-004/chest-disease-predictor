import streamlit as st
import numpy as np
from PIL import Image
import gdown, os
import matplotlib.pyplot as plt
import tensorflow as tf
import random

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI Healthcare", layout="wide")

# -----------------------------
# DARK MODE TOGGLE
# -----------------------------
dark_mode = st.sidebar.toggle("🌙 Dark Mode")

if dark_mode:
    bg = "#1e1e1e"
    text = "white"
    card = "#2c2c2c"
else:
    bg = "#f5f7fa"
    text = "#2c3e50"
    card = "white"

# -----------------------------
# CUSTOM CSS
# -----------------------------
st.markdown(f"""
<style>
body {{
    background-color: {bg};
    color: {text};
}}
.card {{
    background-color: {card};
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}}
.result {{
    font-size: 32px;
    font-weight: bold;
    color: #27ae60;
}}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown("<h1 style='text-align:center;'>🩺 AI Healthcare Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Smart Disease Detection & Hospital Finder</p>", unsafe_allow_html=True)
st.markdown("---")

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "model.tflite"
FILE_ID = "1CBdRBXsze5YgdbRnC8H3GYtqLlydeF-j"

CLASS_NAMES = ['COVID19','NORMAL','PNEUMONIA','TURBERCULOSIS']

# -----------------------------
# HOSPITAL DATA
# -----------------------------
HOSPITALS = {
    "Chennai": ["Apollo Hospital", "MIOT International", "Fortis Malar"],
    "Madurai": ["Meenakshi Mission Hospital", "Apollo Specialty"],
    "Coimbatore": ["KG Hospital", "Ganga Hospital", "PSG Hospitals"],
    "Salem": ["Gokulam Hospital", "Vinayaka Mission"],
    "Trichy": ["Kauvery Hospital", "Apollo Trichy"],
}

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH)

    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -----------------------------
# FUNCTIONS
# -----------------------------
def get_maps_link(hospital, city):
    return f"https://www.google.com/maps/search/{hospital}+{city}"

def get_rating():
    return round(random.uniform(3.5, 5.0), 1)

# -----------------------------
# SIDEBAR INPUT
# -----------------------------
st.sidebar.header("👤 Patient Info")
name = st.sidebar.text_input("Name")
age = st.sidebar.number_input("Age", 0, 120)

uploaded_file = st.file_uploader("📤 Upload X-ray")

# -----------------------------
# MAIN
# -----------------------------
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((224,224))

    arr = np.expand_dims(np.array(img_resized)/255, axis=0).astype('float32')

    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])

    disease = CLASS_NAMES[np.argmax(preds)]
    conf = np.max(preds)*100

    # -----------------------------
    # METRICS
    # -----------------------------
    col1, col2 = st.columns(2)

    col1.metric("🧠 Disease", disease)
    col2.metric("📊 Confidence", f"{conf:.2f}%")

    # -----------------------------
    # IMAGE + GRAPH
    # -----------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.image(img)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        fig, ax = plt.subplots()
        ax.bar(CLASS_NAMES, preds[0])
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------
    # HOSPITALS GRID
    # -----------------------------
    st.markdown("## 🏥 Nearby Hospitals")

    city = st.text_input("📍 Enter your city")

    if city:
        hospitals = HOSPITALS.get(city.title())

        if hospitals:
            cols = st.columns(2)

            for i, h in enumerate(hospitals):
                with cols[i % 2]:
                    rating = get_rating()
                    link = get_maps_link(h, city)

                    st.markdown(f"""
                    <div class="card">
                        <h4>{h}</h4>
                        ⭐ {rating}/5 <br>
                        <a href="{link}" target="_blank">📍 View Map</a>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("No hospitals found")

    # -----------------------------
    # PDF
    # -----------------------------
    if name:
        file = f"/tmp/report.pdf"
        doc = SimpleDocTemplate(file, pagesize=A4)
        styles = getSampleStyleSheet()

        elements = [
            Paragraph("AI MEDICAL REPORT", styles["Title"]),
            Spacer(1, 20),
            Paragraph(f"Name: {name}", styles["Normal"]),
            Paragraph(f"Age: {age}", styles["Normal"]),
            Paragraph(f"Disease: {disease}", styles["Normal"]),
            Paragraph(f"Confidence: {conf:.2f}%", styles["Normal"]),
        ]

        doc.build(elements)

        with open(file, "rb") as f:
            st.download_button("📄 Download Report", f)
