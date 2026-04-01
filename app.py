import streamlit as st
import numpy as np
from PIL import Image
import gdown, os, cv2
import tensorflow as tf
import random
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Healthcare", layout="wide")

MODEL_PATH = "model.tflite"
FILE_ID = "1CBdRBXsze5YgdbRnC8H3GYtqLlydeF-j"

KERAS_PATH = "model.keras"
KERAS_ID = "1GRO5EwB9PDX61G1lZfIHChvCK7JkYe6v"

CLASS_NAMES = ['COVID19','NORMAL','PNEUMONIA','TURBERCULOSIS']

# ---------------- STYLE ----------------
st.markdown("""
<style>
.main {background: linear-gradient(to right, #eef2f3, #ffffff);}
.card {background:white;padding:18px;border-radius:16px;
box-shadow:0px 4px 12px rgba(0,0,0,0.08);margin-bottom:15px;}
.metric-card {background: linear-gradient(135deg,#4facfe,#00f2fe);
padding:15px;border-radius:12px;color:white;text-align:center;font-weight:bold;}
.title {font-size:32px;font-weight:800;}
.subtitle {color:gray;margin-bottom:20px;}
.disease-text {font-size:14px;font-weight:600;line-height:1.7;}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="title">🩺 AI Healthcare System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered Disease Detection & Smart Analysis</div>', unsafe_allow_html=True)

# ---------------- DISEASE INFO ----------------
DISEASE_INFO = {
    "TURBERCULOSIS":{"sym":["cough","weight loss","fatigue"]},
    "PNEUMONIA":{"sym":["fever","cough","chest pain"]},
    "COVID19":{"sym":["fever","cough","breathing"]},
    "NORMAL":{"sym":[]}
}

# ---------------- HOSPITALS ----------------
HOSPITALS = {
    "Chennai":[{"name":"Apollo Hospital","doc":"Dr. Ramesh"}],
    "Coimbatore":[{"name":"KG Hospital","doc":"Dr. Vignesh"}]
}

def maps_link(name,city):
    return f"https://www.google.com/maps/search/{name}+{city}"

def rating():
    return round(random.uniform(3.8,5.0),1)

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_tflite():
    if not os.path.exists(MODEL_PATH):
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH)
    model = tf.lite.Interpreter(model_path=MODEL_PATH)
    model.allocate_tensors()
    return model

@st.cache_resource
def load_grad_model():
    if not os.path.exists(KERAS_PATH):
        gdown.download(f"https://drive.google.com/uc?id={KERAS_ID}", KERAS_PATH)
    return tf.keras.models.load_model(KERAS_PATH, compile=False)

interpreter = load_tflite()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---------------- SIDEBAR ----------------
st.sidebar.header("👤 Patient Details")
name = st.sidebar.text_input("Name")
age = st.sidebar.number_input("Age",0,120)
symptoms_input = st.sidebar.text_area("Enter Symptoms (comma separated)")
uploaded = st.sidebar.file_uploader("Upload X-ray")

# ---------------- SYMPTOM MATCH ----------------
def match_symptoms(user_symptoms):
    scores = {}
    for disease, data in DISEASE_INFO.items():
        match = len(set(user_symptoms) & set(data["sym"]))
        scores[disease] = match
    return max(scores, key=scores.get)

# ---------------- MAIN ----------------
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    img_resized = img.resize((224,224))
    arr = np.expand_dims(np.array(img_resized)/255,0).astype('float32')

    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])

    ai_disease = CLASS_NAMES[np.argmax(preds[0])]
    conf = np.max(preds[0])*100

    # Symptom prediction
    if symptoms_input:
        user_symptoms = [s.strip().lower() for s in symptoms_input.split(",")]
        symptom_disease = match_symptoms(user_symptoms)
    else:
        symptom_disease = "Not provided"

    # -------- METRICS --------
    colA,colB,colC = st.columns(3)

    colA.markdown(f'<div class="metric-card">🧠 AI Prediction<br><br>{ai_disease}</div>', unsafe_allow_html=True)
    colB.markdown(f'<div class="metric-card">📊 Confidence<br><br>{conf:.2f}%</div>', unsafe_allow_html=True)
    colC.markdown(f'<div class="metric-card">🤒 Symptom Match<br><br>{symptom_disease}</div>', unsafe_allow_html=True)

    # -------- LAYOUT --------
    col1,col2 = st.columns([1,1])

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(img, use_container_width=True)

        fig, ax = plt.subplots(figsize=(3,3))
        ax.bar(CLASS_NAMES, preds[0])
        st.pyplot(fig)

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📖 AI + Symptom Analysis")

        st.markdown(f"""
        <div class="disease-text">
        AI predicts <b>{ai_disease}</b> with {conf:.2f}% confidence.<br><br>
        Based on symptoms, possible condition: <b>{symptom_disease}</b>.<br><br>
        <b>Recommendation:</b> Please consult a doctor for confirmation.
        </div>
        """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # -------- GRADCAM --------
    if st.button("🔥 Show Affected Area"):
        try:
            model = load_grad_model()

            last_conv = None
            for layer in reversed(model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv = layer.name
                    break

            if last_conv:
                grad_model = tf.keras.models.Model(
                    inputs=model.input,
                    outputs=[model.get_layer(last_conv).output, model.output]
                )

                with tf.GradientTape() as tape:
                    conv_outputs, predictions = grad_model(arr)

                    if isinstance(predictions, list):
                        predictions = predictions[0]

                    loss = predictions[0][tf.argmax(predictions[0])]

                grads = tape.gradient(loss, conv_outputs)

                pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
                conv_outputs = conv_outputs[0]

                heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
                heatmap = tf.squeeze(heatmap)

                heatmap = np.maximum(heatmap,0)
                heatmap /= (np.max(heatmap)+1e-8)

                if hasattr(heatmap, "numpy"):
                    heatmap = heatmap.numpy()

                heatmap = cv2.resize(heatmap, (224,224))
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

                overlay = cv2.addWeighted(np.array(img_resized),0.6,heatmap,0.4,0)
                st.image(overlay, caption="🔥 Affected Area", use_container_width=True)

        except Exception as e:
            st.error(f"GradCAM Error: {e}")

    # -------- HOSPITAL --------
    st.markdown("## 🏥 Hospitals")
    city = st.text_input("Enter City")

    if city:
        city = city.strip().title()
        if city in HOSPITALS:
            for h in HOSPITALS[city]:
                st.markdown(f"""
                <div class="card">
                🏥 {h['name']}<br>
                👨‍⚕️ {h['doc']}<br>
                ⭐ {rating()}/5<br>
                <a href="{maps_link(h['name'], city)}">📍 Map</a>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No hospitals found")
