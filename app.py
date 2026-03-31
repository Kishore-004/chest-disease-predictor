import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os
import time
import cv2

# PDF Imports
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
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
# DATA
# -----------------------------
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
# HEADER
# -----------------------------
st.markdown("""
<h1 style='text-align: center; color: #2E86C1;'>
🩺 AI Chest Disease Detection System
</h1>
<p style='text-align: center;'>Explainable AI</p>
<hr>
""", unsafe_allow_html=True)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("👤 Patient Details")
patient_name = st.sidebar.text_input("Name")
patient_age = st.sidebar.number_input("Age", 0, 120)

# -----------------------------
# MODEL LOAD
# -----------------------------
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model...")
    gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH)

model = tf.keras.models.load_model(MODEL_PATH, compile=False)
st.success("Model Loaded")

# -----------------------------
# FIXED GRAD-CAM
# -----------------------------
def get_gradcam_heatmap(model, img_array, layer_name):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
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
    heatmap = cv2.resize(heatmap.numpy(), (img.shape[1], img.shape[0]))
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
    elements = []

    text = f"""
    Patient: {name}<br/>
    Age: {age}<br/>
    Disease: {disease}<br/>
    Confidence: {confidence:.2f}%
    """

    elements.append(Paragraph(text, styles["Normal"]))
    doc.build(elements)
    return file

# -----------------------------
# UI
# -----------------------------
st.subheader("Upload X-ray")

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

    st.markdown(f"""
    <div style="padding:20px;background:#EAF2F8;border-left:6px solid #2E86C1">
    <h3>{pred}</h3>
    <p>Confidence: {conf:.2f}%</p>
    </div>
    """, unsafe_allow_html=True)

    st.progress(int(conf))

    # -----------------------------
    # GRAD-CAM (FINAL FIX)
    # -----------------------------
    st.subheader("Grad-CAM")

    try:
        layer = "conv5_block16_concat"
        heatmap = get_gradcam_heatmap(model, img_array, layer)

        img_np = np.array(img_resized)
        gradcam = overlay_heatmap(img_np, heatmap)

        c1, c2 = st.columns(2)
        c1.image(img_resized, caption="Original")
        c2.image(gradcam, caption="Grad-CAM")

    except Exception as e:
        st.error(e)

    # -----------------------------
    # HOSPITAL
    # -----------------------------
    city = st.text_input("Enter City")

    if city:
        hospitals = FALLBACK_HOSPITALS.get(city.title(), [])
        for h in hospitals:
            st.success(h)

    # -----------------------------
    # DOWNLOAD
    # -----------------------------
    if patient_name and patient_age:
        file = generate_report(patient_name, patient_age, pred, conf)
        with open(file,"rb") as f:
            st.download_button("Download Report", f)

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.markdown("AI Healthcare System 🚀")
