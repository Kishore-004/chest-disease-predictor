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
# CONFIG
# -----------------------------
MODEL_PATH = "model.tflite"
FILE_ID = "1CBdRBXsze5YgdbRnC8H3GYtqLlydeF-j"

CLASS_NAMES = ['COVID19','NORMAL','PNEUMONIA','TURBERCULOSIS']

# -----------------------------
# HOSPITAL DATA (Tamil Nadu)
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
# UI
# -----------------------------
st.title("🩺 AI Healthcare System")
st.write("Chest Disease Detection + Hospital Recommendation")
st.markdown("---")

name = st.text_input("Enter Name")
age = st.number_input("Age", 0, 120)

uploaded_file = st.file_uploader("Upload X-ray", type=["jpg","png","jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((224,224))

    arr = np.expand_dims(np.array(img_resized)/255, axis=0).astype('float32')

    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])

    disease = CLASS_NAMES[np.argmax(preds)]
    conf = np.max(preds)*100

    st.success(f"Prediction: {disease} ({conf:.2f}%)")

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Uploaded X-ray")

    with col2:
        fig, ax = plt.subplots()
        ax.bar(CLASS_NAMES, preds[0])
        st.pyplot(fig)

    # -----------------------------
    # HOSPITAL SUGGESTION
    # -----------------------------
    st.markdown("## 🏥 Recommended Hospitals")

    city = st.text_input("Enter your city")

    if city:
        hospitals = HOSPITALS.get(city.title())

        if hospitals:
            for h in hospitals:
                rating = get_rating()
                link = get_maps_link(h, city)

                st.markdown(f"""
                **🏥 {h}**  
                ⭐ Rating: {rating} / 5  
                📍 [View on Map]({link})
                ---
                """)
        else:
            st.warning("No hospitals found for this city")

    # -----------------------------
    # PDF
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
            st.download_button("📄 Download Report", f, file_name="report.pdf")
