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

MODEL_PATH = "model.tflite"
FILE_ID = "1CBdRBXsze5YgdbRnC8H3GYtqLlydeF-j"

KERAS_PATH = "model.keras"
KERAS_ID = "1GRO5EwB9PDX61G1lZfIHChvCK7JkYe6v"

CLASS_NAMES = ['COVID19','NORMAL','PNEUMONIA','TURBERCULOSIS']

# ---------------- STYLE ----------------
st.markdown("""
<style>
.card {
    background:white;padding:18px;border-radius:16px;
    box-shadow:0px 4px 12px rgba(0,0,0,0.08);margin-bottom:15px;
}
.metric-card {
    background: linear-gradient(135deg,#4facfe,#00f2fe);
    padding:15px;border-radius:12px;color:white;text-align:center;font-weight:bold;
}
.title {font-size:32px;font-weight:800;}
.subtitle {color:gray;margin-bottom:20px;}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown('<div class="title">🩺 AI Healthcare System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI + Symptom Based Diagnosis</div>', unsafe_allow_html=True)

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
        {"name":"KG Hospital","doc":"Dr. Vignesh"},
        {"name":"Ganga Hospital","doc":"Dr. Suresh"}
    ]
}

def maps_link(name,city):
    return f"https://www.google.com/maps/search/{name}+{city}"

# ---------------- PDF FUNCTION ----------------
def generate_pdf(name, age, gender, symptoms, disease, conf, city):
    file = "report.pdf"
    doc = SimpleDocTemplate(file, pagesize=A4)
    styles = getSampleStyleSheet()
    content = []

    content.append(Paragraph("AI Healthcare Diagnostic Report", styles["Title"]))
    content.append(Spacer(1,10))

    content.append(Paragraph(f"Name: {name}", styles["Normal"]))
    content.append(Paragraph(f"Age: {age}", styles["Normal"]))
    content.append(Paragraph(f"Gender: {gender}", styles["Normal"]))
    content.append(Paragraph(f"Symptoms: {', '.join(symptoms)}", styles["Normal"]))
    content.append(Spacer(1,10))

    content.append(Paragraph(f"Disease: {disease}", styles["Heading2"]))
    content.append(Paragraph(f"Confidence: {conf:.2f}%", styles["Normal"]))

    if os.path.exists("xray.png"):
        content.append(RLImage("xray.png", width=250, height=250))

    if os.path.exists("gradcam.png"):
        content.append(RLImage("gradcam.png", width=250, height=250))

    content.append(Spacer(1,10))

    if city in HOSPITALS:
        for h in HOSPITALS[city]:
            content.append(Paragraph(f"{h['name']} - {h['doc']}", styles["Normal"]))
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
symptoms = st.sidebar.multiselect("Symptoms", ALL_SYMPTOMS)
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

    st.success(f"Prediction: {disease}")
    st.info(f"Confidence: {conf:.2f}%")

    col1,col2 = st.columns(2)

    with col1:
        st.image(img, use_container_width=True)

        fig, ax = plt.subplots(figsize=(5,4))
        ax.bar(CLASS_NAMES, preds[0])
        ax.set_xticklabels(CLASS_NAMES, rotation=30)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.write("Symptoms:", symptoms)

    # -------- GRADCAM --------
    if st.button("Show GradCAM"):
        model = load_grad()

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
            cv2.imwrite("gradcam.png", overlay)

            st.image(overlay, width=400)

    # -------- HOSPITALS --------
    city = st.text_input("Enter City")

    if city:
        city = city.strip().title()
        if city in HOSPITALS:
            for h in HOSPITALS[city]:
                st.write(h["name"], "-", h["doc"])
                st.write(maps_link(h["name"], city))

    # -------- PDF --------
    if st.button("Download Report"):
        pdf = generate_pdf(name, age, gender, symptoms, disease, conf, city)
        with open(pdf, "rb") as f:
            st.download_button("Download PDF", f)
