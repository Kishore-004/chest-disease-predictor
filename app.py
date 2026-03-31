import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os, cv2, uuid
import matplotlib.pyplot as plt

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "model.h5"   # keep your model locally

CLASS_NAMES = ['COVID19','NORMAL','PNEUMONIA','TUBERCULOSIS']

# -----------------------------
# LOAD MODEL (SAFE)
# -----------------------------
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

model = load_model()

# -----------------------------
# HEADER
# -----------------------------
st.title("🩺 AI Healthcare System")

# -----------------------------
# INPUT
# -----------------------------
name = st.text_input("Name")
age = st.number_input("Age", 0, 120)

# -----------------------------
# UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("Upload X-ray")

# -----------------------------
# SAFE GRADCAM
# -----------------------------
def gradcam(img_array):
    try:
        layer_name = model.layers[-1].name  # fallback layer

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.layers[-2].output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs * pooled_grads
        heatmap = tf.reduce_sum(heatmap, axis=-1)

        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / (np.max(heatmap) + 1e-8)

        return heatmap

    except Exception as e:
        st.warning(f"Grad-CAM failed: {e}")
        return None

# -----------------------------
# MAIN LOGIC
# -----------------------------
if uploaded_file and model:

    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((224,224))

    arr = np.expand_dims(np.array(img_resized)/255, axis=0)

    try:
        preds = model.predict(arr)
    except Exception as e:
        st.error(f"Prediction error: {e}")
        st.stop()

    disease = CLASS_NAMES[np.argmax(preds)]
    conf = np.max(preds) * 100

    st.success(f"{disease} ({conf:.2f}%)")

    col1, col2 = st.columns(2)

    with col1:
        st.image(img)

    with col2:
        fig, ax = plt.subplots()
        ax.bar(CLASS_NAMES, preds[0])
        st.pyplot(fig)

    # SAVE GRAPH (cross-platform)
    graph_path = f"graph_{uuid.uuid4().hex}.png"
    fig.savefig(graph_path)

    # -----------------------------
    # GRAD CAM
    # -----------------------------
    heat = gradcam(arr)

    if heat is not None:
        heat = cv2.resize(heat, (224,224))
        heat = np.uint8(255*heat)
        heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

        grad_img = heat*0.4 + np.array(img_resized)
        grad_img = np.uint8(grad_img)

        st.image(grad_img)

        grad_path = f"grad_{uuid.uuid4().hex}.jpg"
        cv2.imwrite(grad_path, grad_img)

    # -----------------------------
    # PDF GENERATION
    # -----------------------------
    if name:

        file = f"report_{uuid.uuid4().hex}.pdf"

        doc = SimpleDocTemplate(file, pagesize=A4)
        styles = getSampleStyleSheet()

        elements = []
        elements.append(Paragraph("AI MEDICAL REPORT", styles["Title"]))
        elements.append(Spacer(1, 20))

        elements.append(Paragraph(f"Name: {name}", styles["Normal"]))
        elements.append(Paragraph(f"Age: {age}", styles["Normal"]))
        elements.append(Spacer(1, 10))

        elements.append(Paragraph(f"Disease: {disease}", styles["Normal"]))
        elements.append(Paragraph(f"Confidence: {conf:.2f}%", styles["Normal"]))

        if os.path.exists(graph_path):
            elements.append(RLImage(graph_path, width=4*inch, height=3*inch))

        doc.build(elements)

        with open(file, "rb") as f:
            st.download_button(
                "📄 Download Report",
                f.read(),
                file_name="report.pdf"
            )
