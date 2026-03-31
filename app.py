import streamlit as st
from PIL import Image
import numpy as np
import os

st.title("🩺 AI Chest Disease Prediction System")

name = st.text_input("Enter Patient Name")

file = st.file_uploader("Upload Chest X-ray", type=["jpg","png","jpeg"])

CLASS_NAMES = ['COVID19','NORMAL','PNEUMONIA','TUBERCULOSIS']

# -----------------------------
# SAFE MODEL LOAD
# -----------------------------
model = None

try:
    import tensorflow as tf

    MODEL_PATH = "final_chest_disease_model.keras"

    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        st.success("✅ Model Loaded Successfully")
    else:
        st.warning("⚠ Model file not found (using dummy prediction)")

except:
    st.warning("⚠ TensorFlow not supported (using dummy prediction)")

# -----------------------------
# UPLOAD + PREDICTION
# -----------------------------
if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded X-ray")

    if name:
        st.success(f"Hello {name}, image uploaded successfully ✅")

    img_resized = img.resize((224,224))
    arr = np.expand_dims(np.array(img_resized)/255, axis=0)

    # -----------------------------
    # REAL OR DUMMY PREDICTION
    # -----------------------------
    if model:
        preds = model.predict(arr)
        disease = CLASS_NAMES[np.argmax(preds)]
        confidence = np.max(preds)*100
    else:
        import random
        disease = random.choice(CLASS_NAMES)
        confidence = round(random.uniform(80, 99), 2)

    st.subheader("🧠 Prediction Result")
    st.success(f"{disease} ({confidence:.2f}%)")
