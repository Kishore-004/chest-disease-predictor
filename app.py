import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Layer Debug", layout="wide")

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "final_chest_disease_model.keras"
FILE_ID = "1GRO5EwB9PDX61G1lZfIHChvCK7JkYe6v"
CLASS_NAMES = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TURBERCULOSIS']

# -----------------------------
# HEADER
# -----------------------------
st.title("🔍 Grad-CAM Layer Finder")

# -----------------------------
# DOWNLOAD MODEL
# -----------------------------
if not os.path.exists(MODEL_PATH):
    st.info("Downloading model...")
    gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
st.success("Model loaded!")

# -----------------------------
# SHOW ALL LAYERS
# -----------------------------
st.subheader("📋 Model Layers")

for layer in model.layers:
    st.write(layer.name)

# -----------------------------
# OPTIONAL TEST PREDICTION
# -----------------------------
uploaded_file = st.file_uploader("Upload X-ray", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img)

    img = img.resize((224, 224))
    img_array = np.expand_dims(np.array(img)/255.0, axis=0)

    preds = model.predict(img_array)
    st.write("Prediction:", CLASS_NAMES[np.argmax(preds)])
