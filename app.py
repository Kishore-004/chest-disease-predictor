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
# FINAL UI FIX (WORKING)
# -----------------------------
st.markdown("""
<style>

/* 🔥 FIX LABEL SIZE */
div[data-testid="stWidgetLabel"] label {
    font-size: 22px !important;
    font-weight: 600 !important;
}

/* Sidebar labels */
section[data-testid="stSidebar"] div[data-testid="stWidgetLabel"] label {
    font-size: 22px !important;
}

/* Input text */
input {
    font-size: 18px !important;
}

/* Number input */
div[data-baseweb="input"] input {
    font-size: 18px !important;
}

/* File uploader text */
div[data-testid="stFileUploader"] label {
    font-size: 22px !important;
    font-weight: 600;
}

/* Buttons */
button {
    font-size: 18px !important;
}

/* Headings */
h1 { font-size: 48px !important; }
h2 { font-size: 32px !important; }

</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown("""
<h1 style='text-align:center; color:#2E86C1;'>🩺 AI Chest Disease Detection</h1>
<p style='text-align:center; font-size:22px;'>Smart Healthcare Platform | Explainable AI</p>
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
# SIDEBAR
# -----------------------------
st.sidebar.header("👤 Patient Details")
patient_name = st.sidebar.text_input("Name")
patient_age = st.sidebar.number_input("Age", 0, 120)

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
# UPLOAD
# -----------------------------
st.subheader("📤 Upload Chest X-ray")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

with col2:
    if uploaded_file:
        st.image(uploaded_file, use_container_width=True)

# -----------------------------
# PREDICTION
# -----------------------------
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((224,224))
    img_array = np.expand_dims(np.array(img_resized)/255.0, axis=0)

    with st.spinner("Analyzing..."):
        time.sleep(2)
        preds = model.predict(img_array)

    pred = CLASS_NAMES[np.argmax(preds)]
    conf = np.max(preds)*100

    st.success(f"Prediction: {pred}")
    st.write(f"Confidence: {conf:.2f}%")

    st.progress(int(conf))
    st.info(DISEASE_INFO.get(pred))

    # Grad-CAM
    st.subheader("🧠 AI Explanation")

    try:
        heatmap = get_gradcam_heatmap(model, img_array, "conv5_block16_concat")
        gradcam = overlay_heatmap(np.array(img_resized), heatmap)

        c1, c2 = st.columns(2)
        c1.image(img_resized, caption="Original")
        c2.image(gradcam, caption="AI Focus")

    except Exception as e:
        st.error(e)

    # Hospitals
    st.subheader("🏥 Recommended Hospitals")

    city = st.text_input("Enter City")

    if city:
        for h in FALLBACK_HOSPITALS.get(city.title(), []):
            st.write("🏥", h)

# Footer
st.markdown("---")
st.write("AI Healthcare System 🚀")
