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
st.markdown('<div class="subtitle">AI + Symptom Based Disease Prediction</div>', unsafe_allow_html=True)

# ---------------- SYMPTOMS ----------------
SYMPTOMS_DB = {
    "COVID19": ["Fever","Dry Cough","Breathing Difficulty","Fatigue"],
    "PNEUMONIA": ["Fever","Chest Pain","Cough","Shortness of Breath"],
    "TURBERCULOSIS": ["Chronic Cough","Weight Loss","Night Sweats","Fatigue"],
    "NORMAL": []
}

ALL_SYMPTOMS = sorted(list(set(sum(SYMPTOMS_DB.values(), []))))

# ---------------- HOSPITALS ----------------
HOSPITALS = {
    "Chennai":[
        {"name":"Apollo Hospital","doc":"Dr. Ramesh (Pulmonologist)"},
        {"name":"MIOT International","doc":"Dr. Priya (Chest Specialist)"}
    ],
    "Coimbatore":[
        {"name":"KG Hospital","doc":"Dr. Vignesh (Pulmonologist)"},
        {"name":"Ganga Hospital","doc":"Dr. Suresh (Respiratory Specialist)"}
    ],
    "Madurai":[
        {"name":"Meenakshi Mission","doc":"Dr. Karthik (Chest Specialist)"},
        {"name":"Apollo Specialty","doc":"Dr. Anitha (Pulmonologist)"}
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
selected_symptoms = st.sidebar.multiselect("Select Symptoms", ALL_SYMPTOMS)
uploaded = st.sidebar.file_uploader("Upload X-ray")

# ---------------- SYMPTOM MATCH ----------------
def predict_from_symptoms(selected):
    scores = {}
    for disease, sym_list in SYMPTOMS_DB.items():
        scores[disease] = len(set(selected) & set(sym_list))
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

    symptom_disease = predict_from_symptoms(selected_symptoms) if selected_symptoms else "Not Selected"

    # FINAL DECISION
    if selected_symptoms:
        if ai_disease == symptom_disease:
            final_decision = f"✅ Strong Match: {ai_disease}"
        else:
            final_decision = f"⚠️ Conflict: AI → {ai_disease}, Symptoms → {symptom_disease}"
    else:
        final_decision = ai_disease

    # -------- METRICS --------
    colA,colB,colC = st.columns(3)
    colA.markdown(f'<div class="metric-card">🧠 AI Prediction<br><br>{ai_disease}</div>', unsafe_allow_html=True)
    colB.markdown(f'<div class="metric-card">📊 Confidence<br><br>{conf:.2f}%</div>', unsafe_allow_html=True)
    colC.markdown(f'<div class="metric-card">🤒 Symptoms<br><br>{symptom_disease}</div>', unsafe_allow_html=True)

    # -------- LAYOUT --------
    col1,col2 = st.columns([1,1])

    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.image(img, use_container_width=True)

        # FIXED CHART
        fig, ax = plt.subplots(figsize=(5,4))
        ax.bar(CLASS_NAMES, preds[0])
        ax.set_xticks(range(len(CLASS_NAMES)))
        ax.set_xticklabels(CLASS_NAMES, rotation=30, ha='right')
        ax.set_ylim(0,1)
        plt.tight_layout()
        st.pyplot(fig)

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📖 Final Analysis")

        st.markdown(f"""
        <div class="disease-text">
        <b>AI Prediction:</b> {ai_disease} ({conf:.2f}%)<br><br>
        <b>Symptom Prediction:</b> {symptom_disease}<br><br>
        <b>Final Decision:</b> {final_decision}<br><br>
        <b>Selected Symptoms:</b> {", ".join(selected_symptoms) if selected_symptoms else "None"}<br><br>
        <b>Advice:</b> Please consult a doctor.
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

                # MATCH SIZE WITH GRAPH
                st.image(overlay, caption="🔥 Affected Region", width=400)

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
            st.warning("❌ No hospitals found")
