import streamlit as st
import numpy as np
from PIL import Image
import gdown, os, cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import random

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI Healthcare", layout="wide")

# -----------------------------
# DARK MODE
# -----------------------------
dark_mode = st.sidebar.toggle("🌙 Dark Mode")

bg = "#1e1e1e" if dark_mode else "#f5f7fa"
text = "white" if dark_mode else "#2c3e50"
card = "#2c2c2c" if dark_mode else "white"

st.markdown(f"""
<style>
body {{
    background-color: {bg};
    color: {text};
}}
.card {{
    background-color: {card};
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    margin-bottom: 20px;
}}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.markdown("<h1 style='text-align:center;'>🩺 AI Healthcare Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Disease Detection + Explainable AI</p>", unsafe_allow_html=True)
st.markdown("---")

# -----------------------------
# MODEL CONFIG
# -----------------------------
MODEL_PATH = "model.tflite"
FILE_ID = "1CBdRBXsze5YgdbRnC8H3GYtqLlydeF-j"

KERAS_MODEL_PATH = "model.keras"
KERAS_FILE_ID = "1GRO5EwB9PDX61G1lZfIHChvCK7JkYe6v"

CLASS_NAMES = ['COVID19','NORMAL','PNEUMONIA','TURBERCULOSIS']

# -----------------------------
# DISEASE EXPLANATION
# -----------------------------
DISEASE_INFO = {
    "COVID19": {
        "description": "COVID-19 is a contagious respiratory disease caused by SARS-CoV-2 virus.",
        "symptoms": "Fever, cough, fatigue, breathing issues.",
        "causes": "Spread through droplets.",
        "treatment": "Rest, fluids, antiviral support.",
        "precautions": "Mask, hygiene, vaccination."
    },
    "PNEUMONIA": {
        "description": "Lung infection causing fluid buildup.",
        "symptoms": "Chest pain, fever, cough.",
        "causes": "Bacteria/virus.",
        "treatment": "Antibiotics, oxygen.",
        "precautions": "Vaccination, hygiene."
    },
    "TURBERCULOSIS": {
        "description": "Serious bacterial lung disease.",
        "symptoms": "Cough, weight loss, night sweats.",
        "causes": "TB bacteria.",
        "treatment": "Long-term antibiotics.",
        "precautions": "Early diagnosis."
    },
    "NORMAL": {
        "description": "No abnormality detected.",
        "symptoms": "None.",
        "causes": "Healthy lungs.",
        "treatment": "Not needed.",
        "precautions": "Healthy lifestyle."
    }
}

# -----------------------------
# SYMPTOMS
# -----------------------------
SYMPTOM_MAP = {
    "COVID19": ["Fever","Cough","Fatigue"],
    "PNEUMONIA": ["Fever","Chest Pain","Breathing Difficulty"],
    "TURBERCULOSIS": ["Weight Loss","Night Sweats","Cough"],
    "NORMAL": []
}

# -----------------------------
# HOSPITALS + SPECIALISTS
# -----------------------------
HOSPITALS = {
    "Chennai": [
        {"name": "Apollo Hospital", "specialist": "Dr. Ramesh Kumar (Pulmonologist)"},
        {"name": "MIOT International", "specialist": "Dr. Priya Sharma"},
    ],
    "Madurai": [
        {"name": "Meenakshi Mission", "specialist": "Dr. Karthik"},
    ]
}

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH)
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

@st.cache_resource
def load_gradcam_model():
    if not os.path.exists(KERAS_MODEL_PATH):
        gdown.download(f"https://drive.google.com/uc?id={KERAS_FILE_ID}", KERAS_MODEL_PATH)
    return tf.keras.models.load_model(KERAS_MODEL_PATH)

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -----------------------------
# FUNCTIONS
# -----------------------------
def symptom_score(disease, symptoms):
    match = set(symptoms).intersection(SYMPTOM_MAP.get(disease, []))
    return len(match) / (len(SYMPTOM_MAP.get(disease, [])) + 1e-5)

def generate_gradcam(image, model):
    last_conv_layer = [l.name for l in model.layers if "conv" in l.name][-1]
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        class_idx = tf.argmax(predictions[0])
        loss = tf.gather(predictions[0], class_idx)

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs * pooled_grads
    heatmap = tf.reduce_sum(heatmap, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= (np.max(heatmap) + 1e-8)
    return heatmap

# -----------------------------
# INPUTS
# -----------------------------
st.sidebar.header("Patient Info")
name = st.sidebar.text_input("Name")
age = st.sidebar.number_input("Age", 0, 120)

symptoms = st.sidebar.multiselect(
    "Symptoms",
    ["Fever","Cough","Chest Pain","Breathing Difficulty","Fatigue","Weight Loss","Night Sweats"]
)

uploaded_file = st.file_uploader("Upload X-ray")

# -----------------------------
# MAIN
# -----------------------------
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((224,224))
    arr = np.expand_dims(np.array(img_resized)/255, axis=0).astype('float32')

    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])

    final_scores = []
    for i, d in enumerate(CLASS_NAMES):
        final = 0.7 * preds[0][i] + 0.3 * symptom_score(d, symptoms)
        final_scores.append(final)

    final_scores = np.array(final_scores)
    disease = CLASS_NAMES[np.argmax(final_scores)]
    conf = np.max(final_scores) * 100

    st.metric("Disease", disease)
    st.metric("Confidence", f"{conf:.2f}%")

    # 📖 Explanation
    info = DISEASE_INFO[disease]
    st.markdown("## 📖 Disease Explanation")
    st.write(info)

    # 🔥 GradCAM
    if st.button("Show Heatmap"):
        grad_model = load_gradcam_model()
        heatmap = generate_gradcam(arr, grad_model)

        heatmap = cv2.resize(heatmap, (224,224))
        heatmap[heatmap < 0.6] = 0
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(np.array(img_resized), 0.6, heatmap, 0.7, 0)
        st.image(overlay)

    # 🏥 Hospitals
    city = st.text_input("Enter City")
    if city in HOSPITALS:
        for h in HOSPITALS[city]:
            st.write(h)
