import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown
import os
import requests


# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "final_chest_disease_model.keras"
FILE_ID = "1GRO5EwB9PDX61G1lZfIHChvCK7JkYe6v"  # your Drive file ID
CLASS_NAMES = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TURBERCULOSIS']

# -----------------------------
# DOWNLOAD MODEL IF NOT PRESENT
# -----------------------------
if not os.path.exists(MODEL_PATH):
    st.info("📥 Downloading model from Google Drive (first time only)...")
    gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
st.success("✅ Model loaded successfully!")

# -----------------------------
# -----------------------------
# HOSPITAL FETCH FUNCTION (FREE - OpenStreetMap)
# -----------------------------
def get_hospitals(city):
    
    overpass_url = "http://overpass-api.de/api/interpreter"
    
    query = f"""
    [out:json];
    area["name"="{city}"]->.searchArea;
    (
      node["amenity"="hospital"](area.searchArea);
      way["amenity"="hospital"](area.searchArea);
      relation["amenity"="hospital"](area.searchArea);
    );
    out center 5;
    """
    
    response = requests.get(overpass_url, params={'data': query})
    data = response.json()
    
    hospitals = []
    
    for element in data.get('elements', [])[:5]:
        name = element['tags'].get('name', 'N/A')
        hospitals.append(name)
    
    return hospitals

# STREAMLIT UI
# -----------------------------
st.title("🩺 Chest Disease Prediction App")
st.write("Upload a chest X-ray image and get the predicted disease type!")

uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded X-ray", use_container_width=True)

    img = img.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    prediction = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.markdown(f"### 🧠 Predicted: **{predicted_class}**")
    st.markdown(f"### 🎯 Confidence: **{confidence:.2f}%**")
