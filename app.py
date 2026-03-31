import streamlit as st
from PIL import Image

st.title("🩺 AI Chest Disease Prediction System")

name = st.text_input("Enter Patient Name")

file = st.file_uploader("Upload Chest X-ray", type=["jpg","png","jpeg"])

if file:
    img = Image.open(file)
    st.image(img, caption="Uploaded X-ray")

    if name:
        st.success(f"Hello {name}, image uploaded successfully ✅")
