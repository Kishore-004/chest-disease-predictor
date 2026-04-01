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

# ---------------- UI STYLE ----------------
st.markdown("""
<style>
.main {background: linear-gradient(to right, #eef2f3, #ffffff);}
.card {
    background:white;
    padding:18px;
    border-radius:16px;
    box-shadow:0px 4px 12px rgba(0,0,0,0.08);
    margin-bottom:15px;
}
.metric-card {
    background: linear-gradient(135deg,#4facfe,#00f2fe);
    padding:15px;
    border-radius:12px;
    color:white;
    text-align:center;
    font-weight:bold;
}
.title {font-size:32px;font-weight:800;}
.subtitle {color:gray;margin-bottom:20px;}
.disease-text {font-size:14px;font-weight:600;line-height:1.7;}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="title">🩺 AI Healthcare System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered Disease Detection & Smart Hospital Recommendation</div>', unsafe_allow_html=True)

# ---------------- DISEASE INFO ----------------
DISEASE_INFO = {
    "TURBERCULOSIS":{
        "desc":"Tuberculosis is a serious infectious lung disease spread through air.",
        "sym":"persistent cough, weight loss, fatigue",
        "cause":"Mycobacterium tuberculosis",
        "treat":"6–9 months antibiotics",
        "prec":"early diagnosis and full treatment"
    },
    "PNEUMONIA":{
        "desc":"Pneumonia fills lung air sacs with fluid.",
        "sym":"fever, cough, chest pain",
        "cause":"bacteria/virus",
        "treat":"antibiotics, rest",
        "prec":"vaccination, hygiene"
    },
    "COVID19":{
        "desc":"COVID-19 affects lungs and breathing.",
        "sym":"fever, cough, breathlessness",
        "cause":"coronavirus",
        "treat":"supportive care",
        "prec":"mask, vaccination"
    },
    "NORMAL":{
        "desc":"Healthy lungs detected.",
        "sym":"none",
        "cause":"normal",
        "treat":"not required",
        "prec":"healthy lifestyle"
    }
}

# ---------------- HOSPITALS ----------------
HOSPITALS = {
    "Chennai":[
        {"name":"Apollo Hospital","doc":"Dr. Ramesh"},
        {"name":"MIOT","doc":"Dr. Priya"}
    ]
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
uploaded = st.sidebar.file_uploader("Upload X-ray")

# ---------------- MAIN ----------------
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    img_resized = img.resize((224,224))
    arr = np.expand_dims(np.array(img_resized)/255,0).astype('float32')

    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])

    disease = CLASS_NAMES[np.argmax(preds[0])]
    conf = np.max(preds[0])*100

    # -------- METRICS --------
    colA,colB = st.columns(2)
    with colA:
        st.markdown(f'<div class="metric-card">🧠 Prediction<br><br>{disease}</div>', unsafe_allow_html=True)
    with colB:
        st.markdown(f'<div class="metric-card">📊 Confidence<br><br>{conf:.2f}%</div>', unsafe_allow_html=True)

    # -------- MAIN LAYOUT --------
    col1,col2 = st.columns([1,1])

    # LEFT
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(img, use_container_width=True)

        fig, ax = plt.subplots(figsize=(3,3))
        ax.bar(CLASS_NAMES, preds[0])
        ax.set_xticklabels(CLASS_NAMES, rotation=45)
        st.pyplot(fig)

        st.markdown('</div>', unsafe_allow_html=True)

    # RIGHT
    info = DISEASE_INFO[disease]
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📖 Disease Insights")
        st.markdown(f"""
        <div class="disease-text">
        <b>Overview:</b> {info['desc']}<br><br>
        <b>Symptoms:</b> {info['sym']}<br><br>
        <b>Causes:</b> {info['cause']}<br><br>
        <b>Treatment:</b> {info['treat']}<br><br>
        <b>Precautions:</b> {info['prec']}<br><br>
        <b>Severity:</b> {"High ⚠️" if disease!="NORMAL" else "Normal ✅"}
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # -------- GRADCAM FIX --------
    if st.button("🔥 Show Affected Area"):
        try:
            model = load_grad_model()

            # ✅ SAFE Conv layer detection
            last_conv = None
            for layer in reversed(model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    last_conv = layer.name
                    break

            if last_conv is None:
                st.error("❌ No Conv2D layer found → GradCAM not possible")
            else:
                grad_model = tf.keras.models.Model(
                    inputs=model.input,
                    outputs=[model.get_layer(last_conv).output, model.output]
                )

                with tf.GradientTape() as tape:
                    conv_outputs, predictions = grad_model(arr)
                    pred_index = tf.argmax(predictions[0])
                    loss = predictions[:, pred_index]

                grads = tape.gradient(loss, conv_outputs)

                if grads is None:
                    st.error("❌ Gradients not computed")
                else:
                    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))
                    conv_outputs = conv_outputs[0]

                    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
                    heatmap = tf.squeeze(heatmap)

                    heatmap = np.maximum(heatmap,0)
                    heatmap /= (np.max(heatmap)+1e-8)

                    heatmap = cv2.resize(heatmap.numpy(), (224,224))
                    heatmap = np.uint8(255*heatmap)
                    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

                    overlay = cv2.addWeighted(
                        np.array(img_resized),0.6,
                        heatmap,0.4,0
                    )

                    st.image(overlay, caption="🔥 Affected Lung Region", use_container_width=True)

        except Exception as e:
            st.error(f"GradCAM Error: {e}")

    # -------- HOSPITALS --------
    st.markdown("## 🏥 Recommended Hospitals")
    city = st.text_input("Enter City")

    if city:
        city = city.strip().title()
        if city in HOSPITALS:
            cols = st.columns(2)
            for i,h in enumerate(HOSPITALS[city]):
                with cols[i%2]:
                    st.markdown(f"""
                    <div class="card">
                    <b>🏥 {h['name']}</b><br><br>
                    👨‍⚕️ {h['doc']}<br><br>
                    ⭐ {rating()}/5<br><br>
                    <a href="{maps_link(h['name'], city)}">📍 View Map</a>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("No hospitals found")
