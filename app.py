import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os
import time
import cv2
import matplotlib.pyplot as plt

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
    "NORMAL": "No major abnormalities detected."
}

FALLBACK_HOSPITALS = {
    "Chennai": ["Apollo Hospitals", "MIOT International"],
    "Madurai": ["Meenakshi Mission Hospital"],
    "Trichy": ["Kauvery Hospital"]
}

# -----------------------------
# HEADER
# -----------------------------
st.markdown("""
<h1 style='text-align: center; color: #2E86C1;'>
🩺 AI Chest Disease Detection System
</h1>
<p style='text-align: center;'>Early Diagnosis | Explainable AI</p>
<hr>
""", unsafe_allow_html=True)

# -----------------------------
# SIDEBAR
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

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
st.success("✅ Model loaded successfully!")

# -----------------------------
# GRAD-CAM FUNCTION
# -----------------------------
def get_gradcam_heatmap(model, img_array, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = np.maximum(heatmap, 0) / tf.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(img, heatmap):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)

    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * 0.4 + img

    return np.uint8(superimposed_img)

# -----------------------------
# PDF
# -----------------------------
def generate_report(name, age, disease, confidence):
    file_path = "/tmp/report.pdf"
    doc = SimpleDocTemplate(file_path, pagesize=A4)
    styles = getSampleStyleSheet()
    elements = []

    text = f"""
    Patient: {name}<br/>
    Age: {age}<br/><br/>
    Disease: {disease}<br/>
    Confidence: {confidence:.2f}%<br/>
    """

    elements.append(Paragraph(text, styles["Normal"]))
    doc.build(elements)

    return file_path

# -----------------------------
# UI
# -----------------------------
st.subheader("📤 Upload Chest X-ray")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

with col2:
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.image(img, use_container_width=True)

# -----------------------------
# PREDICTION
# -----------------------------
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((224,224))
    img_array = np.expand_dims(np.array(img_resized)/255.0, axis=0)

    with st.spinner("🔍 Analyzing..."):
        time.sleep(2)
        preds = model.predict(img_array)

    predicted_class = CLASS_NAMES[np.argmax(preds)]
    confidence = np.max(preds)*100

    st.markdown(f"""
    <div style="padding:20px;background:#EAF2F8;border-left:6px solid #2E86C1">
    <h3>Prediction: {predicted_class}</h3>
    <p>Confidence: {confidence:.2f}%</p>
    </div>
    """, unsafe_allow_html=True)

    st.progress(int(confidence))

    # -----------------------------
    # GRAD-CAM VISUALIZATION
    # -----------------------------
    st.subheader("🧠 Model Focus (Grad-CAM)")

    try:
        last_conv_layer_name = model.layers[-1].name  # may need change
        heatmap = get_gradcam_heatmap(model, img_array, last_conv_layer_name)

        img_np = np.array(img_resized)
        gradcam_img = overlay_heatmap(img_np, heatmap)

        col1, col2 = st.columns(2)
        with col1:
            st.image(img_resized, caption="Original")
        with col2:
            st.image(gradcam_img, caption="Grad-CAM")

    except Exception as e:
        st.warning("Grad-CAM not working. Check last conv layer name.")

    # -----------------------------
    # REPORT
    # -----------------------------
    if patient_name and patient_age:
        file = generate_report(patient_name, patient_age, predicted_class, confidence)
        with open(file,"rb") as f:
            st.download_button("📄 Download Report", f)
