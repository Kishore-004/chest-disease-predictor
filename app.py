import streamlit as st
import numpy as np
from PIL import Image
import gdown, os, cv2
import tensorflow as tf
import random
import matplotlib.pyplot as plt

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Healthcare", layout="wide")

# TITLE
st.markdown("""
<h1 style='text-align: center; color: #1f77b4;'>
🩺 AI Powered Chest Disease Prediction and Hospital Suggestion
</h1>
""", unsafe_allow_html=True)

MODEL_PATH = "model.tflite"
FILE_ID = "1CBdRBXsze5YgdbRnC8H3GYtqLlydeF-j"

KERAS_PATH = "model.keras"
KERAS_ID = "1GRO5EwB9PDX61G1lZfIHChvCK7JkYe6v"

CLASS_NAMES = ['COVID19','NORMAL','PNEUMONIA','TURBERCULOSIS']

# ---------------- GLOBAL FONT FIX ----------------
st.markdown("""
<style>
html, body, [class*="css"] {
    font-size:16px !important;
    font-weight:600 !important;
}

/* Cards */
.card {
    background:white;
    padding:18px;
    border-radius:16px;
    box-shadow:0px 4px 12px rgba(0,0,0,0.08);
    margin-bottom:15px;
}

/* Highlight */
.highlight {
    background: linear-gradient(135deg,#ff758c,#ff7eb3);
    color:white;
    padding:20px;
    border-radius:12px;
    font-size:20px;
    font-weight:bold;
    text-align:center;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    font-size:16px !important;
    font-weight:600 !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------- SYMPTOMS LIST ----------------
SYMPTOMS_LIST = [
    "Fever", "Dry Cough", "Chronic Cough", "Chest Pain",
    "Breathing Difficulty", "Shortness of Breath",
    "Fatigue", "Weight Loss", "Night Sweats"
]

# ---------------- DISEASE DETAILS ----------------
DISEASE_DETAILS = {
    "COVID19": "COVID-19 is a viral respiratory disease affecting lungs and breathing patterns. It spreads through droplets and affects oxygen levels. Common symptoms include fever, dry cough, fatigue, and breathing issues. In severe cases, it may cause pneumonia and lung damage. Early diagnosis and isolation are critical. Vaccination and hygiene reduce risk.",
    "PNEUMONIA": "Pneumonia is a lung infection where air sacs fill with fluid or pus. It can be caused by bacteria, viruses, or fungi. Symptoms include fever, cough, chest pain, and difficulty breathing. It can be mild or life-threatening. Treatment includes antibiotics, rest, and oxygen therapy. Vaccination and early care help prevent complications.",
    "TURBERCULOSIS": "Tuberculosis is a bacterial infection affecting lungs and spreads through air. It causes persistent cough, weight loss, and night sweats. It can damage lung tissue if untreated. Diagnosis requires imaging and lab tests. Treatment involves long-term antibiotics for 6–9 months. Early detection prevents spread and complications.",
    "NORMAL": "No abnormalities detected in the lungs. The X-ray appears normal with no signs of infection or disease. Maintain a healthy lifestyle and regular checkups to ensure continued well-being."
}

# ---------------- HOSPITALS ----------------
HOSPITALS = {
    "Chennai":[
        {"name":"Apollo Hospital","doc":"Dr. Ramesh (Pulmonologist)"},
        {"name":"MIOT International","doc":"Dr. Priya (Chest Specialist)"}
    ],
    "Coimbatore":[
        {"name":"KG Hospital","doc":"Dr. Vignesh"},
        {"name":"Ganga Hospital","doc":"Dr. Suresh"}
    ]
}

def maps_link(name,city):
    return f"https://www.google.com/maps/search/{name}+{city}"

def rating():
    return round(random.uniform(3.8,5.0),1)

# ---------------- PDF ----------------
def generate_pdf(name, age, gender, symptoms, disease, conf, city):
    file = "report.pdf"
    doc = SimpleDocTemplate(file, pagesize=A4)
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("<b>AI Healthcare Diagnostic Report</b>", styles["Title"]))
    content.append(Spacer(1,10))

    content.append(Paragraph("<b>Patient Details</b>", styles["Heading2"]))
    content.append(Paragraph(f"Name: {name}", styles["Normal"]))
    content.append(Paragraph(f"Age: {age}", styles["Normal"]))
    content.append(Paragraph(f"Gender: {gender}", styles["Normal"]))
    content.append(Paragraph(f"Symptoms: {', '.join(symptoms)}", styles["Normal"]))
    content.append(Spacer(1,10))

    if os.path.exists("xray.png") and os.path.exists("gradcam.png"):
        content.append(RLImage("xray.png", width=250, height=250))
        content.append(RLImage("gradcam.png", width=250, height=250))

    content.append(Spacer(1,10))

    content.append(Paragraph("<b>Disease Information</b>", styles["Heading2"]))
    content.append(Paragraph(DISEASE_DETAILS[disease], styles["Normal"]))
    content.append(Spacer(1,10))

    content.append(Paragraph("<b>Recommended Hospitals</b>", styles["Heading2"]))
    if city in HOSPITALS:
        for h in HOSPITALS[city]:
            content.append(Paragraph(f"{h['name']} - {h['doc']} - Rating: {rating()}⭐", styles["Normal"]))
            content.append(Paragraph(maps_link(h['name'], city), styles["Normal"]))

    doc.build(content)
    return file

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH)
    model = tf.lite.Interpreter(model_path=MODEL_PATH)
    model.allocate_tensors()
    return model

@st.cache_resource
def load_grad():
    if not os.path.exists(KERAS_PATH):
        gdown.download(f"https://drive.google.com/uc?id={KERAS_ID}", KERAS_PATH)
    return tf.keras.models.load_model(KERAS_PATH, compile=False)

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---------------- SIDEBAR ----------------
st.sidebar.header("Patient Info")
name = st.sidebar.text_input("Name")
age = st.sidebar.number_input("Age",0,120)
gender = st.sidebar.selectbox("Gender", ["Male","Female","Other"])
symptoms = st.sidebar.multiselect("Select Symptoms", SYMPTOMS_LIST)
uploaded = st.sidebar.file_uploader("Upload X-ray")

# ---------------- MAIN ----------------
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    img_resized = img.resize((224,224))
    img_resized.save("xray.png")

    arr = np.expand_dims(np.array(img_resized)/255,0).astype('float32')

    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])

    disease = CLASS_NAMES[np.argmax(preds[0])]
    conf = np.max(preds[0])*100

    st.markdown(f'<div class="highlight">🧠 Predicted Disease: {disease} ({conf:.2f}%)</div>', unsafe_allow_html=True)

    col1,col2 = st.columns(2)

    with col1:
        st.image(img, use_container_width=True)
        fig, ax = plt.subplots(figsize=(5,4))
        ax.bar(CLASS_NAMES, preds[0])
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("### 📖 Detailed Explanation")
        st.write(DISEASE_DETAILS[disease])
        st.markdown('</div>', unsafe_allow_html=True)

    # -------- GRADCAM --------
    if st.button("Show GradCAM"):
        model = load_grad()
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv = layer.name
                break

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
        cv2.imwrite("gradcam.png", overlay)

        st.image(overlay, width=400)

    # -------- PDF --------
    if st.button("Download Report"):
        pdf = generate_pdf(name, age, gender, symptoms, disease, conf, "Chennai")
        with open(pdf, "rb") as f:
            st.download_button(
                "📄 Download PDF Report",
                data=f,
                file_name="Medical_Report.pdf",
                mime="application/pdf"
            )
