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
# SYMPTOMS
# -----------------------------
SYMPTOM_MAP = {
    "COVID19": ["Fever", "Cough", "Fatigue"],
    "PNEUMONIA": ["Fever", "Chest Pain", "Breathing Difficulty"],
    "TURBERCULOSIS": ["Weight Loss", "Night Sweats", "Cough"],
    "NORMAL": []
}

# -----------------------------
# HOSPITALS + SPECIALISTS
# -----------------------------
HOSPITALS = {
    "Chennai": [
        {"name": "Apollo Hospital", "specialist": "Dr. Ramesh Kumar (Pulmonologist)"},
        {"name": "MIOT International", "specialist": "Dr. Priya Sharma (Chest Specialist)"},
        {"name": "Fortis Malar", "specialist": "Dr. Arjun Singh (Respiratory Expert)"}
    ],
    "Madurai": [
        {"name": "Meenakshi Mission Hospital", "specialist": "Dr. Karthik (Pulmonologist)"},
        {"name": "Apollo Specialty", "specialist": "Dr. Lakshmi (Chest Physician)"}
    ],
    "Coimbatore": [
        {"name": "KG Hospital", "specialist": "Dr. Vignesh (Pulmonologist)"},
        {"name": "Ganga Hospital", "specialist": "Dr. Suresh (Respiratory Specialist)"}
    ],
    "Salem": [
        {"name": "Gokulam Hospital", "specialist": "Dr. Naveen (Pulmonologist)"},
        {"name": "Vinayaka Mission", "specialist": "Dr. Deepak (Chest Specialist)"}
    ],
    "Trichy": [
        {"name": "Kauvery Hospital", "specialist": "Dr. Mohan (Pulmonologist)"},
        {"name": "Apollo Trichy", "specialist": "Dr. Anand (Respiratory Specialist)"}
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
def get_maps_link(hospital, city):
    return f"https://www.google.com/maps/search/{hospital}+{city}"

def get_rating():
    return round(random.uniform(3.5, 5.0), 1)

def symptom_score(disease, symptoms):
    match = set(symptoms).intersection(SYMPTOM_MAP.get(disease, []))
    return len(match) / (len(SYMPTOM_MAP.get(disease, [])) + 1e-5)

def generate_gradcam(image, model):
    last_conv_layer = None
    for layer in reversed(model.layers):
        if "conv" in layer.name:
            last_conv_layer = layer.name
            break

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(image)
        class_idx = tf.argmax(predictions[0])
        class_idx = tf.cast(class_idx, tf.int32)
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
# SIDEBAR
# -----------------------------
st.sidebar.header("👤 Patient Info")
name = st.sidebar.text_input("Name")
age = st.sidebar.number_input("Age", 0, 120)

symptoms = st.sidebar.multiselect(
    "🤒 Symptoms",
    ["Fever","Cough","Chest Pain","Breathing Difficulty","Fatigue","Weight Loss","Night Sweats"]
)

uploaded_file = st.file_uploader("📤 Upload X-ray")

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

    model_scores = preds[0]
    final_scores = []

    for i, d in enumerate(CLASS_NAMES):
        final = (0.7 * model_scores[i]) + (0.3 * symptom_score(d, symptoms))
        final_scores.append(final)

    final_scores = np.array(final_scores)

    disease = CLASS_NAMES[np.argmax(final_scores)]
    conf = np.max(final_scores) * 100

    col1, col2 = st.columns(2)
    col1.metric("🧠 Disease", disease)
    col2.metric("📊 Confidence", f"{conf:.2f}%")

    col1, col2 = st.columns(2)

    with col1:
        st.image(img)

    with col2:
        fig, ax = plt.subplots()
        ax.bar(CLASS_NAMES, final_scores)
        st.pyplot(fig)

    # -----------------------------
    # GRADCAM
    # -----------------------------
    st.markdown("## 🔍 Explain Prediction")

    if st.button("Show Affected Lung Area"):
        grad_model = load_gradcam_model()
        heatmap = generate_gradcam(arr, grad_model)

        heatmap = cv2.resize(heatmap, (224,224))
        heatmap = heatmap / (np.max(heatmap) + 1e-8)

        threshold = 0.6
        heatmap[heatmap < threshold] = 0

        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(np.array(img_resized), 0.6, heatmap, 0.7, 0)
        st.image(overlay, caption="🔥 Affected Region")

    # -----------------------------
    # HOSPITALS
    # -----------------------------
    st.markdown("## 🏥 Nearby Hospitals")

    city = st.text_input("Enter your city")

    if city:
        hospitals = HOSPITALS.get(city.title())

        if hospitals:
            cols = st.columns(2)
            for i, h in enumerate(hospitals):
                with cols[i % 2]:
                    rating = get_rating()
                    link = get_maps_link(h["name"], city)

                    st.markdown(f"""
                    <div class="card">
                        <h4>🏥 {h["name"]}</h4>
                        👨‍⚕️ {h["specialist"]}<br>
                        ⭐ {rating}/5<br>
                        <a href="{link}" target="_blank">📍 View Map</a>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("No hospitals found")

    # -----------------------------
    # PDF
    # -----------------------------
    if name:
        file = "/tmp/report.pdf"
        doc = SimpleDocTemplate(file, pagesize=A4)
        styles = getSampleStyleSheet()

        elements = [
            Paragraph("AI MEDICAL REPORT", styles["Title"]),
            Spacer(1, 20),
            Paragraph(f"Name: {name}", styles["Normal"]),
            Paragraph(f"Age: {age}", styles["Normal"]),
            Paragraph(f"Disease: {disease}", styles["Normal"]),
            Paragraph(f"Confidence: {conf:.2f}%", styles["Normal"]),
            Paragraph(f"Symptoms: {', '.join(symptoms)}", styles["Normal"]),
        ]

        doc.build(elements)

        with open(file, "rb") as f:
            st.download_button("📄 Download Report", f)
