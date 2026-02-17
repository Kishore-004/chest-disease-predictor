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
# -----------------------------
# DISEASE → SPECIALIST MAPPING
# -----------------------------
DISEASE_SPECIALIST = {
    "COVID19": "Pulmonologist",
    "NORMAL": "General Physician",
    "PNEUMONIA": "Pulmonologist",
    "TURBERCULOSIS": "Chest Specialist"
}

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

    try:
        # Step 1: Get coordinates of city using Nominatim
        geo_url = "https://nominatim.openstreetmap.org/search"
        geo_params = {
            "q": city,
            "format": "json"
        }

        headers = {
            "User-Agent": "AI-Healthcare-App"
        }

        geo_response = requests.get(geo_url, params=geo_params, headers=headers, timeout=10)
        geo_data = geo_response.json()

        if not geo_data:
            return []

        lat = geo_data[0]["lat"]
        lon = geo_data[0]["lon"]

        # Step 2: Search hospitals and clinics within 20km radius
        overpass_url = "https://overpass-api.de/api/interpreter"

        query = f"""
        [out:json][timeout:25];
        (
          node["amenity"="hospital"](around:20000,{lat},{lon});
          node["amenity"="clinic"](around:20000,{lat},{lon});
          node["healthcare"="hospital"](around:20000,{lat},{lon});
          node["healthcare"="clinic"](around:20000,{lat},{lon});
        );
        out body;
        """

        response = requests.get(overpass_url, params={'data': query}, headers=headers, timeout=20)
        data = response.json()

        hospitals = []

        for element in data.get('elements', [])[:5]:
            name = element.get('tags', {}).get('name')
            if name:
                hospitals.append(name)

        return hospitals

    except Exception:
        return []




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

    specialist = DISEASE_SPECIALIST.get(predicted_class, "General Physician")
    st.markdown(f"### 👨‍⚕ Recommended Specialist: **{specialist}**")

    st.markdown(f"### 🎯 Confidence: **{confidence:.2f}%**")


    # -----------------------------
    # HOSPITAL RECOMMENDATION
    # -----------------------------
    user_city = st.text_input("🏙 Enter your city to find nearby hospitals")

    if user_city:
        hospitals = get_hospitals(user_city)

        st.subheader(f"🏥 {specialist} Hospitals in {user_city}")


        if hospitals:
            for hospital in hospitals:
                st.write("•", hospital)
        else:
            st.write("No hospitals found. Try another city.")

st.markdown("---")
st.markdown("⚠ This system is for educational purposes only. Please consult a certified medical professional for proper diagnosis and treatment.")

