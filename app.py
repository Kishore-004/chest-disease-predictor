import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown, os, cv2, uuid
import matplotlib.pyplot as plt

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI Healthcare", layout="wide")

# -----------------------------
# BIG HEADER
# -----------------------------
st.markdown("""
<h1 style='text-align:center; font-size:60px;'>🩺 AI Healthcare System</h1>
<p style='text-align:center; font-size:28px;'>Chest Disease Detection</p>
<hr>
""", unsafe_allow_html=True)

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "final_chest_disease_model.keras"
FILE_ID = "1GRO5EwB9PDX61G1lZfIHChvCK7JkYe6v"
CLASS_NAMES = ['COVID19','NORMAL','PNEUMONIA','TURBERCULOSIS']

DISEASE_INFO = {
    "COVID19": "COVID-19 is a viral infection affecting lungs causing fever, cough and breathing issues.",
    "PNEUMONIA": "Pneumonia is a lung infection causing inflammation and difficulty breathing.",
    "TURBERCULOSIS": "Tuberculosis is a bacterial infection affecting lungs requiring long treatment.",
    "NORMAL": "No abnormalities detected."
}

DISEASE_SPECIALIST = {
    "COVID19": "Pulmonologist",
    "PNEUMONIA": "Pulmonologist",
    "TURBERCULOSIS": "Chest Specialist",
    "NORMAL": "General Physician"
}

FALLBACK_HOSPITALS = {
    "Chennai": ["Apollo Hospitals", "MIOT International", "Fortis Malar"],
    "Madurai": ["Meenakshi Mission Hospital"],
    "Trichy": ["Kauvery Hospital"]
}

# -----------------------------
# LOAD MODEL
# -----------------------------
if not os.path.exists(MODEL_PATH):
    gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH)

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.markdown("<h2>👤 Patient Details</h2>", unsafe_allow_html=True)

name = st.sidebar.text_input("Name")
age = st.sidebar.number_input("Age",0,120)

symptoms = st.sidebar.multiselect(
    "Symptoms",
    ["Fever","Cough","Chest Pain","Breathing Difficulty","Fatigue"]
)

# -----------------------------
# GRAD-CAM (FIXED)
# -----------------------------
def gradcam(img_array):
    layer = "conv5_block16_concat"

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(layer).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)

        if isinstance(predictions, list):
            predictions = predictions[0]

        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs * pooled_grads
    heatmap = tf.reduce_sum(heatmap, axis=-1)

    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)

    return heatmap

# -----------------------------
# PDF FUNCTION
# -----------------------------
def generate_pdf(name, age, disease, conf, symptoms, hospitals, specialist, description, grad_path):
    file = f"/tmp/report_{uuid.uuid4().hex}.pdf"
    doc = SimpleDocTemplate(file, pagesize=A4)
    styles = getSampleStyleSheet()

    elements = []

    elements.append(Paragraph("<b>AI MEDICAL REPORT</b>", styles["Title"]))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph(f"Name: {name}", styles["Normal"]))
    elements.append(Paragraph(f"Age: {age}", styles["Normal"]))
    elements.append(Spacer(1,10))

    elements.append(Paragraph("<b>Symptoms:</b>", styles["Heading2"]))
    elements.append(Paragraph(", ".join(symptoms) if symptoms else "None", styles["Normal"]))
    elements.append(Spacer(1,10))

    elements.append(Paragraph("<b>Diagnosis:</b>", styles["Heading2"]))
    elements.append(Paragraph(f"Disease: {disease}", styles["Normal"]))
    elements.append(Paragraph(f"Confidence: {conf:.2f}%", styles["Normal"]))
    elements.append(Paragraph(f"Specialist: {specialist}", styles["Normal"]))
    elements.append(Spacer(1,10))

    elements.append(Paragraph("<b>Disease Explanation:</b>", styles["Heading2"]))
    elements.append(Paragraph(description, styles["Normal"]))
    elements.append(Spacer(1,10))

    elements.append(Paragraph("<b>Recommended Hospitals:</b>", styles["Heading2"]))
    for h in hospitals:
        elements.append(Paragraph(f"• {h} ({specialist})", styles["Normal"]))

    elements.append(Spacer(1,15))

    if grad_path:
        elements.append(RLImage(grad_path, width=4*inch, height=4*inch))

    doc.build(elements)
    return file

# -----------------------------
# UPLOAD
# -----------------------------
st.markdown("<h2>📤 Upload Chest X-ray</h2>", unsafe_allow_html=True)

file = st.file_uploader("", type=["jpg","png","jpeg"])

# -----------------------------
# PREDICTION
# -----------------------------
if file:
    img = Image.open(file).convert("RGB")
    st.image(img, use_container_width=True)

    img_resized = img.resize((224,224))
    arr = np.expand_dims(np.array(img_resized)/255, axis=0)

    preds = model.predict(arr)
    disease = CLASS_NAMES[np.argmax(preds)]
    conf = np.max(preds)*100

    st.markdown(f"<h3>Prediction: {disease}</h3>", unsafe_allow_html=True)
    st.markdown(f"<h3>Confidence: {conf:.2f}%</h3>", unsafe_allow_html=True)

    # -----------------------------
    # SMALL CHART
    # -----------------------------
    fig, ax = plt.subplots(figsize=(4,3))
    ax.bar(CLASS_NAMES, preds[0])
    st.pyplot(fig)

    # -----------------------------
    # GRAD-CAM
    # -----------------------------
    heat = gradcam(arr)
    heat = heat.numpy() if hasattr(heat,"numpy") else heat

    heat = cv2.resize(heat,(224,224))
    heat = np.uint8(255*heat)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

    grad_img = heat*0.4 + np.array(img_resized)
    grad_img = np.uint8(grad_img)

    st.image(grad_img, caption="Grad-CAM")

    grad_path = f"/tmp/grad_{uuid.uuid4().hex}.jpg"
    cv2.imwrite(grad_path, grad_img)

    # -----------------------------
    # MAP
    # -----------------------------
    city = st.text_input("Enter City")

    if city:
        st.markdown(f"[🏥 Search Hospitals in {city}](https://www.google.com/maps/search/hospital+in+{city})")

    # -----------------------------
    # PDF DOWNLOAD
    # -----------------------------
    if name:
        hospitals = FALLBACK_HOSPITALS.get(city.title(), [])

        pdf = generate_pdf(
            name, age, disease, conf,
            symptoms,
            hospitals,
            DISEASE_SPECIALIST.get(disease),
            DISEASE_INFO.get(disease),
            grad_path
        )

        with open(pdf, "rb") as f:
            st.download_button(
                "📄 Download Full Medical Report",
                f,
                file_name="AI_Medical_Report.pdf",
                mime="application/pdf"
            )
