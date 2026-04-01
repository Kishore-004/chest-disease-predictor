import streamlit as st
import numpy as np
from PIL import Image
import gdown, os, cv2, uuid
import matplotlib.pyplot as plt
import tensorflow as tf

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI Healthcare", layout="wide")

# -----------------------------
# HEADER
# -----------------------------
st.title("🩺 AI Healthcare System")
st.write("Chest Disease Detection Platform")
st.markdown("---")

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "model.tflite"
FILE_ID = "1CBdRBXsze5YgdbRnC8H3GYtqLlydeF-j"

CLASS_NAMES = ['COVID19','NORMAL','PNEUMONIA','TURBERCULOSIS']

# -----------------------------
# LOAD MODEL (TFLITE)
# -----------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model... ⏳"):
            gdown.download(
                f"https://drive.google.com/uc?id={FILE_ID}",
                MODEL_PATH,
                quiet=False
            )

    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()

    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("Patient Details")
name = st.sidebar.text_input("Name")
age = st.sidebar.number_input("Age", 0, 120)

symptoms = st.sidebar.multiselect(
    "Symptoms",
    ["Fever","Cough","Chest Pain","Breathing Difficulty","Fatigue"]
)

# -----------------------------
# PDF FUNCTION
# -----------------------------
def generate_pdf(name, age, disease, conf):
    file = f"/tmp/report_{uuid.uuid4().hex}.pdf"

    doc = SimpleDocTemplate(file, pagesize=A4)
    styles = getSampleStyleSheet()

    elements = []
    elements.append(Paragraph("AI MEDICAL REPORT", styles["Title"]))
    elements.append(Spacer(1, 20))
    elements.append(Paragraph(f"Name: {name}", styles["Normal"]))
    elements.append(Paragraph(f"Age: {age}", styles["Normal"]))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph("Diagnosis:", styles["Heading2"]))
    elements.append(Paragraph(f"Disease: {disease}", styles["Normal"]))
    elements.append(Paragraph(f"Confidence: {conf:.2f}%", styles["Normal"]))

    doc.build(elements)
    return file

# -----------------------------
# UPLOAD IMAGE
# -----------------------------
uploaded_file = st.file_uploader("Upload X-ray", type=["jpg","png","jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((224,224))

    arr = np.expand_dims(np.array(img_resized)/255, axis=0).astype('float32')

    # TFLite Prediction
    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])

    disease = CLASS_NAMES[np.argmax(preds)]
    conf = np.max(preds)*100

    st.success(f"{disease} ({conf:.2f}%)")

    # Display
    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Uploaded X-ray")

    with col2:
        fig, ax = plt.subplots()
        ax.bar(CLASS_NAMES, preds[0])
        st.pyplot(fig)

    # Grad-CAM removed (not supported in TFLite)

    # PDF Download
    if name:
        pdf = generate_pdf(name, age, disease, conf)

        with open(pdf, "rb") as f:
            st.download_button(
                "📄 Download Report",
                data=f,
                file_name="report.pdf"
            )
