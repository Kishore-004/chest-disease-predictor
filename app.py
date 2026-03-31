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
# CONFIG
# -----------------------------
st.set_page_config(page_title="AI Healthcare", layout="wide")

MODEL_PATH = "final_chest_disease_model.keras"
FILE_ID = "1GRO5EwB9PDX61G1lZfIHChvCK7JkYe6v"
CLASS_NAMES = ['COVID19','NORMAL','PNEUMONIA','TURBERCULOSIS']

# -----------------------------
# HEADER
# -----------------------------
st.markdown("<h1 style='text-align:center;'>🩺 AI Healthcare System</h1>", unsafe_allow_html=True)

# -----------------------------
# DATA
# -----------------------------
DISEASE_INFO = {
    "COVID19": "COVID-19 is a viral infection affecting lungs causing fever, cough and breathing issues.",
    "PNEUMONIA": "Pneumonia is a lung infection causing inflammation and breathing difficulty.",
    "TURBERCULOSIS": "Tuberculosis is a bacterial lung infection requiring long treatment.",
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
name = st.sidebar.text_input("Name")
age = st.sidebar.number_input("Age",0,120)

symptoms = st.sidebar.multiselect(
    "Symptoms",
    ["Fever","Cough","Chest Pain","Breathing Difficulty","Fatigue"]
)

# -----------------------------
# GRAD-CAM
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
# PDF FUNCTION (FINAL)
# -----------------------------
def generate_pdf(name, age, disease, conf, symptoms, hospitals, specialist, description, grad_path, graph_path):
    file = f"/tmp/report_{uuid.uuid4().hex}.pdf"

    doc = SimpleDocTemplate(file, pagesize=A4)
    styles = getSampleStyleSheet()

    elements = []

    elements.append(Paragraph("AI MEDICAL REPORT", styles["Title"]))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph(f"Name: {name}", styles["Normal"]))
    elements.append(Paragraph(f"Age: {age}", styles["Normal"]))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("Symptoms:", styles["Heading2"]))
    elements.append(Paragraph(", ".join(symptoms) if symptoms else "None", styles["Normal"]))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("Diagnosis:", styles["Heading2"]))
    elements.append(Paragraph(f"Disease: {disease}", styles["Normal"]))
    elements.append(Paragraph(f"Confidence: {conf:.2f}%", styles["Normal"]))
    elements.append(Paragraph(f"Specialist: {specialist}", styles["Normal"]))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("Disease Explanation:", styles["Heading2"]))
    elements.append(Paragraph(description, styles["Normal"]))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("Recommended Hospitals:", styles["Heading2"]))
    for h in hospitals:
        elements.append(Paragraph(f"{h} ({specialist})", styles["Normal"]))

    elements.append(Spacer(1, 15))

    if os.path.exists(grad_path):
        elements.append(Paragraph("Grad-CAM Visualization:", styles["Heading2"]))
        elements.append(Spacer(1, 10))
        elements.append(RLImage(grad_path, width=4*inch, height=4*inch))

    elements.append(Spacer(1, 15))

    if os.path.exists(graph_path):
        elements.append(Paragraph("Prediction Graph:", styles["Heading2"]))
        elements.append(Spacer(1, 10))
        elements.append(RLImage(graph_path, width=4*inch, height=3*inch))

    doc.build(elements)
    return file

# -----------------------------
# UPLOAD
# -----------------------------
file = st.file_uploader("Upload X-ray")

if file:
    img = Image.open(file).convert("RGB")

    img_resized = img.resize((224,224))
    arr = np.expand_dims(np.array(img_resized)/255, axis=0)

    preds = model.predict(arr)
    disease = CLASS_NAMES[np.argmax(preds)]
    conf = np.max(preds)*100

    st.success(f"{disease} ({conf:.2f}%)")

    # IMAGE + GRAPH
    col1, col2 = st.columns(2)

    with col1:
        st.image(img)

    with col2:
        fig, ax = plt.subplots(figsize=(4,3))
        ax.bar(CLASS_NAMES, preds[0])
        st.pyplot(fig)

    graph_path = f"/tmp/graph_{uuid.uuid4().hex}.png"
    fig.savefig(graph_path)

    # GRAD-CAM
    heat = gradcam(arr)
    heat = heat.numpy() if hasattr(heat,"numpy") else heat

    heat = cv2.resize(heat,(224,224))
    heat = np.uint8(255*heat)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

    grad_img = heat*0.4 + np.array(img_resized)
    grad_img = np.uint8(grad_img)

    st.image(grad_img)

    grad_path = f"/tmp/grad_{uuid.uuid4().hex}.jpg"
    cv2.imwrite(grad_path, grad_img)

    city = st.text_input("Enter City")

    if name:
        hospitals = FALLBACK_HOSPITALS.get(city.title(), [])

        pdf = generate_pdf(
            name, age, disease, conf,
            symptoms,
            hospitals,
            DISEASE_SPECIALIST.get(disease),
            DISEASE_INFO.get(disease),
            grad_path,
            graph_path
        )

        # 🔥 FIXED DOWNLOAD
        with open(pdf, "rb") as f:
            pdf_bytes = f.read()

        st.download_button(
            "📄 Download PDF Report",
            data=pdf_bytes,
            file_name="AI_Medical_Report.pdf",
            mime="application/pdf"
        )
