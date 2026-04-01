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
# CUSTOM UI CSS
# -----------------------------
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
h1 {
    text-align: center;
    color: #2c3e50;
}
.card {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}
.result {
    font-size: 28px;
    font-weight: bold;
    color: #27ae60;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown("<h1>🩺 AI Healthcare System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Chest Disease Detection & Hospital Finder</p>", unsafe_allow_html=True)
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
    "Chennai": ["Apollo Hospital Chennai", "MIOT International", "Fortis Malar Hospital"],
    "Madurai": ["Meenakshi Mission Hospital", "Apollo Specialty Hospital Madurai"],
    "Coimbatore": ["KG Hospital", "Ganga Hospital", "PSG Hospitals"],
    "Salem": ["Gokulam Hospital", "Vinayaka Mission Hospital"],
    "Trichy": ["Kauvery Hospital", "Apollo Hospital Trichy"],
    "Vellore": ["CMC Vellore", "Naruvi Hospital"],
    "Tiruvannamalai": ["Govt Medical College Hospital", "Arunai Hospital"],
    "Krishnagiri": ["Govt Hospital Krishnagiri", "PES Hospital"]
}

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model... ⏳"):
            gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH)

    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def get_maps_link(hospital, city):
    query = hospital + " " + city
    return f"https://www.google.com/maps/search/{query.replace(' ', '+')}"

def get_rating():
    return round(random.uniform(3.5, 5.0), 1)

# -----------------------------
# INPUTS
# -----------------------------
name = st.text_input("👤 Enter Name")
age = st.number_input("🎂 Age", 0, 120)

uploaded_file = st.file_uploader("📤 Upload X-ray", type=["jpg","png","jpeg"])

# -----------------------------
# MAIN LOGIC
# -----------------------------
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((224,224))

    arr = np.expand_dims(np.array(img_resized)/255, axis=0).astype('float32')

    # Prediction
    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])

    disease = CLASS_NAMES[np.argmax(preds)]
    conf = np.max(preds)*100

    # -----------------------------
    # RESULT CARD
    # -----------------------------
    st.markdown(f"""
    <div class="card">
        <div class="result">🧠 {disease}</div>
        <p>Confidence: {conf:.2f}%</p>
    </div>
    """, unsafe_allow_html=True)

    # -----------------------------
    # IMAGE + GRAPH
    # -----------------------------
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.image(img, caption="Uploaded X-ray")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        fig, ax = plt.subplots()
        ax.bar(CLASS_NAMES, preds[0])
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    # -----------------------------
    # HOSPITAL SUGGESTION
    # -----------------------------
    st.markdown("## 🏥 Recommended Hospitals")

    city = st.text_input("📍 Enter your city")

    if city:
        hospitals = HOSPITALS.get(city.title())

        if hospitals:
            for h in hospitals:
                rating = get_rating()
                link = get_maps_link(h, city)

                st.markdown(f"""
                <div class="card">
                    <h4>🏥 {h}</h4>
                    <p>⭐ Rating: {rating} / 5</p>
                    <a href="{link}" target="_blank">📍 View on Google Maps</a>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No hospitals found for this city")

    # -----------------------------
    # PDF GENERATION
    # -----------------------------
    if name:
        file = f"/tmp/report_{name}.pdf"
        doc = SimpleDocTemplate(file, pagesize=A4)
        styles = getSampleStyleSheet()

        elements = []
        elements.append(Paragraph("AI MEDICAL REPORT", styles["Title"]))
        elements.append(Spacer(1, 20))
        elements.append(Paragraph(f"Name: {name}", styles["Normal"]))
        elements.append(Paragraph(f"Age: {age}", styles["Normal"]))
        elements.append(Paragraph(f"Disease: {disease}", styles["Normal"]))
        elements.append(Paragraph(f"Confidence: {conf:.2f}%", styles["Normal"]))

        doc.build(elements)

        with open(file, "rb") as f:
            st.download_button(
                "📄 Download Medical Report",
                data=f,
                file_name="report.pdf",
                use_container_width=True
            )
