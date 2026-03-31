import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os, cv2, uuid
import matplotlib.pyplot as plt

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI Healthcare", layout="wide")
st.title("🩺 AI Healthcare System")

# -----------------------------
# MODEL LOAD (SAFE)
# -----------------------------
MODEL_PATH = "model.keras"
CLASS_NAMES = ['COVID19','NORMAL','PNEUMONIA','TUBERCULOSIS']

@st.cache_resource
def load_model():
    try:
        if not os.path.exists(MODEL_PATH):
            st.error("❌ model.keras not found in folder")
            return None

        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model

    except Exception as e:
        st.error(f"❌ Model load error: {e}")
        return None

model = load_model()

if model is None:
    st.stop()
else:
    st.success("✅ Model loaded")

# -----------------------------
# USER INPUT
# -----------------------------
name = st.sidebar.text_input("Patient Name")
age = st.sidebar.number_input("Age", 0, 120)

# -----------------------------
# GRAD-CAM (AUTO SAFE)
# -----------------------------
def gradcam(img_array):
    try:
        last_conv_layer = None
        for layer in reversed(model.layers):
            if "conv" in layer.name:
                last_conv_layer = layer.name
                break

        if last_conv_layer is None:
            return None

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer).output, model.output]
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
        heatmap /= (np.max(heatmap) + 1e-8)

        return heatmap

    except Exception as e:
        st.warning(f"Grad-CAM skipped: {e}")
        return None

# -----------------------------
# PDF GENERATION
# -----------------------------
def generate_pdf(name, age, disease, conf, grad_path):
    file = f"/tmp/report_{uuid.uuid4().hex}.pdf"

    doc = SimpleDocTemplate(file, pagesize=A4)
    styles = getSampleStyleSheet()

    elements = []

    elements.append(Paragraph("AI Medical Report", styles["Title"]))
    elements.append(Spacer(1,20))

    elements.append(Paragraph(f"Name: {name}", styles["Normal"]))
    elements.append(Paragraph(f"Age: {age}", styles["Normal"]))
    elements.append(Paragraph(f"Disease: {disease}", styles["Normal"]))
    elements.append(Paragraph(f"Confidence: {conf:.2f}%", styles["Normal"]))

    elements.append(Spacer(1,20))

    if grad_path and os.path.exists(grad_path):
        elements.append(RLImage(grad_path, width=4*inch, height=4*inch))

    doc.build(elements)
    return file

# -----------------------------
# FILE UPLOAD
# -----------------------------
file = st.file_uploader("Upload Chest X-ray")

if file:
    try:
        img = Image.open(file).convert("RGB")

        img_resized = img.resize((224,224))
        arr = np.expand_dims(np.array(img_resized)/255, axis=0)

        preds = model.predict(arr)

        disease = CLASS_NAMES[np.argmax(preds)]
        conf = np.max(preds)*100

        st.success(f"{disease} ({conf:.2f}%)")

        col1, col2 = st.columns(2)

        with col1:
            st.image(img)

        with col2:
            fig, ax = plt.subplots()
            ax.bar(CLASS_NAMES, preds[0])
            st.pyplot(fig)

        # -----------------------------
        # GRAD-CAM
        # -----------------------------
        heat = gradcam(arr)
        grad_path = None

        if heat is not None:
            heat = cv2.resize(heat, (224,224))
            heat = np.uint8(255*heat)
            heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

            grad_img = heat*0.4 + np.array(img_resized)
            grad_img = np.uint8(grad_img)

            st.image(grad_img)

            grad_path = f"/tmp/grad_{uuid.uuid4().hex}.jpg"
            cv2.imwrite(grad_path, grad_img)

        # -----------------------------
        # PDF DOWNLOAD
        # -----------------------------
        if name:
            pdf = generate_pdf(name, age, disease, conf, grad_path)

            with open(pdf, "rb") as f:
                pdf_bytes = f.read()

            st.download_button(
                "📄 Download Report",
                data=pdf_bytes,
                file_name="AI_Report.pdf",
                mime="application/pdf"
            )

    except Exception as e:
        st.error(f"❌ Error: {e}")
