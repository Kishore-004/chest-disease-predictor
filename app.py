
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Chest Disease Prediction", page_icon="🩻")

st.title("🩺 Chest X-ray Disease Prediction App")
st.write("Upload a chest X-ray image and get the predicted disease type!")

# ✅ Load the trained model
model = tf.keras.models.load_model("final_chest_disease_model.keras")

# ✅ Define class names (same as your training dataset)
CLASS_NAMES = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TURBERCULOSIS']

uploaded_file = st.file_uploader("Upload an X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded X-ray", use_column_width=True)

    img = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)

    prediction = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.success(f"🩺 Prediction: {predicted_class}")
    st.write(f"📊 Confidence: {confidence:.2f}%")
