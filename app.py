import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os
import requests

# PDF Imports
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "final_chest_disease_model.keras"
FILE_ID = "1GRO5EwB9PDX61G1lZfIHChvCK7JkYe6v"
CLASS_NAMES = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TURBERCULOSIS']

# -----------------------------
# DISEASE → SPECIALIST
# -----------------------------
DISEASE_SPECIALIST = {
    "COVID19": "Pulmonologist",
    "NORMAL": "General Physician",
    "PNEUMONIA": "Pulmonologist",
    "TURBERCULOSIS": "Chest Specialist"
}

# -----------------------------
# DISEASE EXPLANATION
# -----------------------------
DISEASE_INFO = {
    "COVID19": "COVID-19 is a viral respiratory infection affecting the lungs. Symptoms include fever, cough, and breathing difficulty.",
    "PNEUMONIA": "Pneumonia is a lung infection causing inflammation in air sacs. Symptoms include chest pain and cough.",
    "TURBERCULOSIS": "Tuberculosis is a bacterial infection affecting lungs and requires long-term treatment.",
    "NORMAL": "No major abnormalities detected. Consult doctor if symptoms persist."
}

# -----------------------------
# FALLBACK HOSPITALS
# -----------------------------
FALLBACK_HOSPITALS = {
    "Chennai": ["Apollo Hospitals", "MIOT International", "Fortis Malar Hospital"],
    "Madurai": ["Meenakshi Mission Hospital", "Government Rajaji Hospital"],
    "Tirunelveli": ["Shifa Hospital", "Government Medical College Hospital Tirunelveli"],
    "Trichy": ["Kauvery Hospital Trichy", "Government Medical College Hospital Trichy"],
    "Villupuram": ["Government Medical College Hospital Villupuram"],
    "Virudhunagar": ["Government Hospital Virudhunagar"],
    "Tiruvannamalai": ["Government Medical College Hospital Tiruvannamalai"],
    "Tiruvallur": ["Government Hospital Tiruvallur"],
    "Tenkasi": ["Government Hospital Tenkasi"]
}

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
    Recommended Specialist: {specialist}<br/><br/>
    Explanation:<br/>{description}
    """

    elements.append(Paragraph(text, styles["Normal"]))
    doc.build(elements)

    return file_path

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🩺 AI Chest Disease Prediction System")

st.subheader("👤 Patient Details")
patient_name = st.text_input("Patient Name")
patient_age = st.number_input("Patient Age", min_value=0, max_value=120, step=1)

uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded X-ray", use_container_width=True)

    img = img.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    prediction = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    specialist = DISEASE_SPECIALIST.get(predicted_class)
    description = DISEASE_INFO.get(predicted_class)

    st.markdown(f"### 🧠 Predicted: **{predicted_class}**")
    st.markdown(f"### 👨‍⚕ Recommended Specialist: **{specialist}**")
    st.markdown(f"### 🎯 Confidence: **{confidence:.2f}%**")

    st.markdown("### 📖 Disease Explanation")
    st.write(description)

    # -----------------------------
    # HOSPITAL SECTION
    # -----------------------------
    user_city = st.text_input("🏙 Enter your city")

    if user_city:
        formatted_city = user_city.title()
        hospitals = FALLBACK_HOSPITALS.get(formatted_city, [])

        st.subheader(f"🏥 Hospitals in {formatted_city}")

        if hospitals:
            for hospital in hospitals:
                map_query = hospital.replace(" ", "+")
                map_link = f"https://www.google.com/maps/search/?api=1&query={map_query}+{formatted_city}"
                st.markdown(f"• {hospital}  |  [📍 View on Map]({map_link})")
        else:
            st.write("No hospital data available for this city.")

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
                "📄 Download Medical Report",
                f,
                file_name="AI_Medical_Report.pdf",
                mime="application/pdf"
            )

# -----------------------------
# DISCLAIMER
# -----------------------------
st.markdown("---")
st.markdown("⚠ This system is for educational purposes only. Please consult a certified medical professional.")


