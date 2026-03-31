import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os
import time
import cv2

# PDF Imports
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI Healthcare", layout="wide")

# -----------------------------
# HIDE DEFAULT LABELS
# -----------------------------
st.markdown("""
<style>
label {display: none !important;}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown("""
<h1 style='text-align:center; font-size:48px; color:#2E86C1;'>
🩺 AI Chest Disease Detection
</h1>
<p style='text-align:center; font-size:22px;'>
Smart Healthcare Platform | Explainable AI
</p>
<hr>
""", unsafe_allow_html=True)

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "final_chest_disease_model.keras"
FILE_ID = "1GRO5EwB9PDX61G1lZfIHChvCK7JkYe6v"
CLASS_NAMES = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TURBERCULOSIS']

DISEASE_SPECIALIST = {
    "COVID19": "Pulmonologist",
    "NORMAL": "General Physician",
    "PNEUMONIA": "Pulmonologist",
    "TURBERCULOSIS": "Chest Specialist"
}

DISEASE_INFO = {
    "COVID19": "COVID-19 affects lungs.",
    "PNEUMONIA": "Pneumonia is lung infection.",
    "TURBERCULOSIS": "TB is bacterial lung disease.",
    "NORMAL": "No major abnormalities."
}

FALLBACK_HOSPITALS = {
    "Chennai": ["Apollo Hospitals", "MIOT International"],
    "Madurai": ["Meenakshi Mission Hospital"],
    "Trichy": ["Kauvery Hospital"]
}

# -----------------------------
# SIDEBAR (BIG LABELS)
# -----------------------------
st.sidebar.markdown("<h2>👤 Patient Details</h2>", unsafe_allow_html=True)

st.sidebar.markdown("<p style='font-size:20px;'>Name</p>", unsafe_allow_html=True)
patient_name = st.sidebar.text_input("", key="name")

st.sidebar.markdown("<p style='font-size:20px;'>Age</p>", unsafe_allow_html=True)
patient_age = st.sidebar.number_input("", 0, 120, key="age")

# -----------------------------
# LOAD MODEL
# -----------------------------
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model...")
    gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH)

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
st.success("✅ Model Loaded")

# -----------------------------
# GRAD-CAM
# -----------------------------
def get_gradcam_heatmap(model, img_array, layer_name):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        if isinstance(predictions, list):
            predictions = predictions[0]

        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]

    heatmap = conv_outputs * pooled_grads
    heatmap = tf.reduce_sum(heatmap, axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)

    return heatmap

def overlay_heatmap(img, heatmap):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return np.uint8(heatmap * 0.4 + img)

# -----------------------------
# PDF
# -----------------------------
def generate_report(name, age, disease, confidence):
    file = "/tmp/report.pdf"
    doc = SimpleDocTemplate(file, pagesize=A4)
    styles = getSampleStyleSheet()

    text = f"""
    Patient: {name}<br/>
    Age: {age}<br/>
    Disease: {disease}<br/>
    Confidence: {confidence:.2f}%
    """

    doc.build([Paragraph(text, styles["Normal"])])
    return file

# -----------------------------
# UPLOAD SECTION (BIG LABEL)
# -----------------------------
st.markdown("<h2 style='font-size:32px;'>📤 Upload Chest X-ray</h2>", unsafe_allow_html=True)
st.markdown("<p style='font-size:20px;'>Upload Image</p>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("", type=["jpg","png","jpeg"])

with col2:
    if uploaded_file:
        st.image(uploaded_file, caption="X-ray Preview", use_container_width=True)

# -----------------------------
# PREDICTION
# -----------------------------
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((224,224))
    img_array = np.expand_dims(np.array(img_resized)/255.0, axis=0)

    with st.spinner("🔍 Analyzing X-ray..."):
        time.sleep(2)
        preds = model.predict(img_array)

    pred = CLASS_NAMES[np.argmax(preds)]
    conf = np.max(preds)*100

    st.markdown(f"""
    <div style="padding:20px;border-radius:12px;background:#f4f6f7;border-left:6px solid #2E86C1;font-size:22px;">
    <b>🧠 Prediction:</b> {pred} <br>
    <b>👨‍⚕ Specialist:</b> {DISEASE_SPECIALIST.get(pred)} <br>
    <b>🎯 Confidence:</b> {conf:.2f}%
    </div>
    """, unsafe_allow_html=True)

    st.progress(int(conf))
    st.info(DISEASE_INFO.get(pred))

    # Grad-CAM
    st.markdown("<h2 style='font-size:32px;'>🧠 AI Explanation (Grad-CAM)</h2>", unsafe_allow_html=True)

    try:
        layer_name = "conv5_block16_concat"
        heatmap = get_gradcam_heatmap(model, img_array, layer_name)
        gradcam = overlay_heatmap(np.array(img_resized), heatmap)

        c1, c2 = st.columns(2)
        c1.image(img_resized, caption="Original")
        c2.image(gradcam, caption="AI Focus")

    except Exception as e:
        st.error(f"Grad-CAM Error: {e}")

    # Hospitals
    st.markdown("<h2 style='font-size:32px;'>🏥 Recommended Hospitals</h2>", unsafe_allow_html=True)

    city = st.text_input("Enter City")

    if city:
        for h in FALLBACK_HOSPITALS.get(city.title(), []):
            st.markdown(f"<div style='padding:15px;background:#eef3f7;margin:10px 0;border-radius:10px;font-size:20px;'>🏥 {h}</div>", unsafe_allow_html=True)

    # Download
    if patient_name and patient_age:
        file = generate_report(patient_name, patient_age, pred, conf)
        with open(file,"rb") as f:
            st.download_button("📄 Download Report", f)

# Footer
st.markdown("<hr><p style='text-align:center;font-size:18px;'>AI Healthcare System 🚀</p>", unsafe_allow_html=True)
