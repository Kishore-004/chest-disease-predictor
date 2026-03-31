import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown, os, cv2
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
# LOAD MODEL
# -----------------------------
if not os.path.exists(MODEL_PATH):
    gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH)

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# -----------------------------
# SIDEBAR
# -----------------------------
st.sidebar.title("Patient Details")
name = st.sidebar.text_input("Name")
age = st.sidebar.number_input("Age",0,120)

symptoms = st.sidebar.multiselect(
    "Symptoms",
    ["Fever","Cough","Chest Pain","Breathing Difficulty","Fatigue"]
)

# -----------------------------
# GRAD-CAM (FINAL FIXED)
# -----------------------------
def gradcam(img_array):
    last_layer = "conv5_block16_concat"

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_layer).output, model.output]
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
def generate_pdf(name, age, disease, conf, grad_path):
    file="/tmp/report.pdf"
    doc=SimpleDocTemplate(file,pagesize=A4)
    styles=getSampleStyleSheet()

    elements=[]

    elements.append(Paragraph("AI Medical Report",styles["Title"]))
    elements.append(Spacer(1,20))

    elements.append(Paragraph(f"Name: {name}",styles["Normal"]))
    elements.append(Paragraph(f"Age: {age}",styles["Normal"]))
    elements.append(Paragraph(f"Disease: {disease}",styles["Normal"]))
    elements.append(Paragraph(f"Confidence: {conf:.2f}%",styles["Normal"]))

    elements.append(Spacer(1,20))
    elements.append(RLImage(grad_path,width=4*inch,height=4*inch))

    doc.build(elements)
    return file

# -----------------------------
# UPLOAD
# -----------------------------
file = st.file_uploader("Upload X-ray")

if file:
    img = Image.open(file).convert("RGB")
    st.image(img)

    img_resized = img.resize((224,224))
    arr = np.expand_dims(np.array(img_resized)/255,axis=0)

    preds = model.predict(arr)
    disease = CLASS_NAMES[np.argmax(preds)]
    conf = np.max(preds)*100

    st.success(f"{disease} ({conf:.2f}%)")

    # -----------------------------
    # CHART
    # -----------------------------
    fig = plt.figure()
    plt.bar(CLASS_NAMES, preds[0])
    plt.title("Prediction Confidence")
    st.pyplot(fig)

    # -----------------------------
    # GRAD-CAM
    # -----------------------------
    heat = gradcam(arr)
    heat = heat.numpy() if hasattr(heat, "numpy") else heat

    heat = cv2.resize(heat,(224,224))
    heat = np.uint8(255*heat)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

    grad_img = heat*0.4 + np.array(img_resized)
    grad_img = np.uint8(grad_img)

    st.image(grad_img, caption="Grad-CAM")

    grad_path="/tmp/grad.jpg"
    cv2.imwrite(grad_path, grad_img)

    # -----------------------------
    # MAP
    # -----------------------------
    city = st.text_input("Enter City")
    if city:
        st.markdown(f"[Search Hospitals in {city}](https://www.google.com/maps/search/hospital+in+{city})")

    # -----------------------------
    # PDF DOWNLOAD
    # -----------------------------
    if name:
        pdf = generate_pdf(name, age, disease, conf, grad_path)

        with open(pdf,"rb") as f:
            st.download_button("Download Report", f)
