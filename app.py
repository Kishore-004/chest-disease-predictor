import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random

# -----------------------------
# TITLE
# -----------------------------
st.title("🩺 AI Chest Disease Prediction System")

# -----------------------------
# INPUT
# -----------------------------
name = st.text_input("Enter Patient Name")

file = st.file_uploader("Upload Chest X-ray", type=["jpg","png","jpeg"])

CLASS_NAMES = ['COVID19','NORMAL','PNEUMONIA','TUBERCULOSIS']

# -----------------------------
# MAIN LOGIC
# -----------------------------
if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded X-ray")

    if name:
        st.success(f"Hello {name}, image uploaded successfully ✅")

    # -----------------------------
    # DUMMY PREDICTION (STABLE)
    # -----------------------------
    prediction = random.choice(CLASS_NAMES)
    confidence = round(random.uniform(80, 99), 2)

    st.subheader("🧠 Prediction Result")
    st.success(f"{prediction} ({confidence}%)")

    # -----------------------------
    # GRAPH (SAFE)
    # -----------------------------
    st.subheader("📊 Prediction Confidence Graph")

    fig, ax = plt.subplots(figsize=(4,3))
    fake_values = [random.random() for _ in CLASS_NAMES]
    ax.bar(CLASS_NAMES, fake_values)

    st.pyplot(fig)
