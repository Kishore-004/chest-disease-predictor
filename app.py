import streamlit as st
import numpy as np
from PIL import Image
import os
import uuid
import matplotlib.pyplot as plt

st.set_page_config(page_title="AI Healthcare", layout="wide")

st.title("🩺 AI Chest Disease Prediction System")

CLASS_NAMES = ['COVID19','NORMAL','PNEUMONIA','TURBERCULOSIS']

# -----------------------------
# SAFE MODEL LOAD (NO CRASH)
# -----------------------------
model = None

try:
    import tensorflow as tf

    MODEL_PATH = "final_chest_disease_model.keras"

    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        st.success("Model loaded")
    else:
        st.warning("Model file not found")

except Exception as e:
    st.error("TensorFlow not supported in this environment")

# -----------------------------
# INPUT
# -----------------------------
name = st.sidebar.text_input("Patient Name")
age = st.sidebar.number_input("Age", 0, 120)

# -----------------------------
# UPLOAD
# -----------------------------
file = st.file_uploader("Upload Chest X-ray")

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="X-ray")

    img_resized = img.resize((224,224))
    arr = np.expand_dims(np.array(img_resized)/255, axis=0)

    if model:
        preds = model.predict(arr)
        disease = CLASS_NAMES[np.argmax(preds)]
        conf = np.max(preds)*100
    else:
        disease = "MODEL NOT LOADED"
        conf = 0

    st.success(f"{disease} ({conf:.2f}%)")

    # GRAPH
    fig, ax = plt.subplots(figsize=(4,3))
    ax.bar(CLASS_NAMES, [0.25,0.25,0.25,0.25] if model is None else preds[0])
    st.pyplot(fig)

    # PDF
    if name:
        from reportlab.platypus import SimpleDocTemplate, Paragraph

        file_path = f"/tmp/report_{uuid.uuid4().hex}.pdf"
        doc = SimpleDocTemplate(file_path)

        doc.build([
            Paragraph(f"Name: {name}", None),
            Paragraph(f"Age: {age}", None),
            Paragraph(f"Disease: {disease}", None)
        ])

        with open(file_path, "rb") as f:
            st.download_button("Download PDF", f.read())
