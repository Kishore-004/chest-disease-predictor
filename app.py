import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os
import time

# PDF Imports
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Medical System",
    page_icon="🩺",
    layout="wide"
)

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "final_chest_disease_model.keras"
FILE_ID = "1GRO5EwB9PDX61G1lZfIHChvCK7JkYe6v"
CLASS_NAMES = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TURBERCULOSIS']

# -----------------------------
# DISEASE DATA
# -----------------------------
DISEASE_SPECIALIST = {
    "COVID19": "Pulmonologist",
    "NORMAL": "General Physician",
    "PNEUMONIA": "Pulmonologist",
    "TURBERCULOSIS": "Chest Specialist"
}

DISEASE_INFO = {
    "COVID19": "COVID-19 is a viral respiratory infection affecting the lungs.",
    "PNEUMONIA": "Pneumonia is a lung infection causing inflammation in air sacs.",
    "TURBERCULOSIS": "Tuberculosis is a bacterial infection affecting lungs.",
    "NORMAL": "No major abnormalities detected. Consult doctor if symptoms persist."
}

FALLBACK_HOSPITALS = {
    "Chennai": ["Apollo Hospitals", "MIOT International", "Fortis Malar Hospital"],
    "Madurai": ["Meenakshi Mission Hospital", "Government Rajaji Hospital"],
    "Trichy": ["Kauvery Hospital", "Government Hospital Trichy"]
}

# -----------------------------
# HEADER
# -----------------------------
st.markdown("""
<h1 style='text-align: center; color: #2E86C1;'>
🩺 AI Chest Disease Detection System
</h1>
<p style='text-align: center;'>Early Diagnosis | Smart Healthcare</p>
<hr>
""", unsafe_allow_html=True)

# -----------------------------
# SIDEBAR (PATIENT DETAILS)
# -----------------------------
st.sidebar.header("👤 Patient Details")
patient_name = st.sidebar.text_input("Patient Name")
patient_age = st.sidebar.number_input("Patient Age", 0, 120)

# -----------------------------
# DOWNLOAD MODEL
# -----------------------------
if not os.path.exists(MODEL_PATH):
    st.info("📥 Downloading model...")
    gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
st.success("✅ Model loaded successfully!")

# -----------------------------
# PDF GENERATION
# -----------------------------
def generate_report(name, age, disease, confidence, specialist, description):
    file_path = "/tmp/AI_Medical_Report.pdf"
    doc = SimpleDocTemplate(file_path, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("<b>AI Medical Report</b>", styles["Title"]))
    elements.append(Spacer(1, 0.3 * inch))

    text = f"""
    Patient Name: {name}<br/>
    Age: {age}<br/><br/>
    Predicted Disease: {disease}<br/>
    Confidence: {confidence:.2f}%<br/>
    Specialist: {specialist}<br/><br/>
    {description}
    """

    elements.append(Paragraph(text, styles["Normal"]))
    doc.build(elements)

    return file_path

# -----------------------------
# UPLOAD SECTION
# -----------------------------
st.subheader("📤 Upload Chest X-ray")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

with col2:
    if uploaded_file:
        st.image(uploaded_file, caption="X-ray Preview", use_container_width=True)

# -----------------------------
# PREDICTION
# -----------------------------
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    with st.spinner("🔍 Analyzing X-ray..."):
        time.sleep(2)
        prediction = model.predict(img_array)

    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    specialist = DISEASE_SPECIALIST.get(predicted_class)
    description = DISEASE_INFO.get(predicted_class)

    # -----------------------------
    # RESULT CARD
    # -----------------------------
    st.markdown(f"""
    <div style="padding:20px;
                border-radius:10px;
                background-color:#EAF2F8;
                border-left:6px solid #2E86C1;">
    <h3>🧠 Prediction: {predicted_class}</h3>
    <p>👨‍⚕ Specialist: {specialist}</p>
    <p>🎯 Confidence: {confidence:.2f}%</p>
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Confidence Level")
    st.progress(int(confidence))

    st.info(f"📖 {description}")

    # -----------------------------
    # HOSPITAL SECTION
    # -----------------------------
    user_city = st.text_input("🏙 Enter your city")

    if user_city:
        city = user_city.title()
        hospitals = FALLBACK_HOSPITALS.get(city, [])

        st.subheader(f"🏥 Hospitals in {city}")

        if hospitals:
            for hospital in hospitals:
                map_link = f"https://www.google.com/maps/search/{hospital}+{city}"
                st.success(f"🏥 {hospital}")
                st.markdown(f"[📍 View on Map]({map_link})")
        else:
            st.warning("No hospital data available.")

    # -----------------------------
    # DOWNLOAD REPORT
    # -----------------------------
    if patient_name and patient_age:
        report_file = generate_report(
            patient_name,
            patient_age,
            predicted_class,
            confidence,
            specialist,
            description
        )

        with open(report_file, "rb") as f:
            st.download_button(
                "📄 Download Full Medical Report",
                f,
                file_name="AI_Medical_Report.pdf",
                mime="application/pdf"
            )

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("""
<hr>
<p style='text-align:center'>
Developed by Krish | AI Healthcare System 🚀
</p>
""", unsafe_allow_html=True)


