import streamlit as st
from PIL import Image
import random

st.title("🩺 AI Chest Disease Prediction System")

name = st.text_input("Enter Patient Name")

file = st.file_uploader("Upload Chest X-ray", type=["jpg","png","jpeg"])

if file:
    img = Image.open(file)
    st.image(img, caption="Uploaded X-ray")

    if name:
        st.success(f"Hello {name}, image uploaded successfully ✅")

    # -----------------------------
    # DUMMY PREDICTION (SAFE)
    # -----------------------------
    diseases = ["COVID19", "NORMAL", "PNEUMONIA", "TUBERCULOSIS"]
    prediction = random.choice(diseases)
    confidence = round(random.uniform(80, 99), 2)

    st.subheader("🧠 Prediction Result")
    st.success(f"{prediction} ({confidence}%)")
