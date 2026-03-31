import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown, os, cv2, uuid
import matplotlib.pyplot as plt

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

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "model.keras"
FILE_ID = "1GRO5EwB9PDX61G1lZfIHChvCK7JkYe6v"

CLASS_NAMES = ['COVID19','NORMAL','PNEUMONIA','TUBERCULOSIS']

# -----------------------------
# SAFE MODEL LOAD
# -----------------------------
@st.cache_resource
def load_model():
    try:
        if not os.path.exists(MODEL_PATH):
            with st.spinner("Downloading model..."):
                gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)

        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model

    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None

model = load_model()

if model is None:
    st.stop()

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.header("Patient Details")
name = st.sidebar.text_input("Name")
age = st.sidebar.number_input("Age", 0, 120)

# -----------------------------
# SAFE GRADCAM
# -----------------------------
def gradcam(img_array):
    try:
        # auto-detect last conv layer
        last_conv = None
        for layer in reversed(model.layers):
            if "conv" in layer.name:
                last_conv = layer.name
                break

        if last_conv is None:
            return None

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if isinstance(predictions, list):
                predictions = predictions[0]

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

    except:
        return None

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

    elements.append(Paragraph(f"Disease: {disease}", styles["Heading2"]))
    elements.append(Paragraph(f"Confidence: {conf:.2f}%", styles["Normal"]))

    doc.build(elements)
    return file

# -----------------------------
# UPLOAD
# -----------------------------
uploaded_file = st.file_uploader("Upload X-ray", type=["jpg","png","jpeg"])

if uploaded_file:
    try:
        img = Image.open(uploaded_file).convert("RGB")
        img_resized = img.resize((224,224))

        arr = np.expand_dims(np.array(img_resized)/255, axis=0)

        preds = model.predict(arr)

        if isinstance(preds, list):
            preds = preds[0]

        preds = np.array(preds)

        disease = CLASS_NAMES[np.argmax(preds)]
        conf = np.max(preds)*100

        st.success(f"{disease} ({conf:.2f}%)")

        # -----------------------------
        # DISPLAY IMAGE
        # -----------------------------
        col1, col2 = st.columns(2)

        with col1:
            st.image(img, caption="Uploaded X-ray")

        with col2:
            fig, ax = plt.subplots()
            ax.bar(CLASS_NAMES, preds[0])
            st.pyplot(fig)
            plt.close(fig)

        # -----------------------------
        # GRADCAM
        # -----------------------------
        heat = gradcam(arr)

        if heat is not None:
            heat = cv2.resize(heat, (224,224))
            heat = np.uint8(255 * heat)
            heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

            grad_img = heat * 0.4 + np.array(img_resized)
            grad_img = np.uint8(grad_img)

            st.image(grad_img, caption="Grad-CAM")

        # -----------------------------
        # PDF DOWNLOAD
        # -----------------------------
        if name:
            pdf = generate_pdf(name, age, disease, conf)

            with open(pdf, "rb") as f:
                st.download_button(
                    "📄 Download Report",
                    data=f.read(),
                    file_name="report.pdf"
                )

    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
