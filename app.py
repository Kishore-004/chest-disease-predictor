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
# BIG UI HEADER
# -----------------------------
st.markdown("""
<h1 style='text-align:center; font-size:60px; font-weight:900;'>🩺 AI Healthcare System</h1>
<p style='text-align:center; font-size:28px;'>Chest Disease Detection</p>
<hr>
""", unsafe_allow_html=True)

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "final_chest_disease_model.keras"
FILE_ID = "1GRO5EwB9PDX61G1lZfIHChvCK7JkYe6v"
CLASS_NAMES = ['COVID19','NORMAL','PNEUMONIA','TURBERCULOSIS']

# -----------------------------
# LOAD MODEL
# -----------------------------
if not os.path.exists(MODEL_PATH):
    gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH)

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# -----------------------------
# SIDEBAR (BIG TEXT)
# -----------------------------
st.sidebar.markdown("<h2 style='font-size:30px;'>👤 Patient Details</h2>", unsafe_allow_html=True)

st.sidebar.markdown("<p style='font-size:24px;'>Name</p>", unsafe_allow_html=True)
name = st.sidebar.text_input("", key="name")

st.sidebar.markdown("<p style='font-size:24px;'>Age</p>", unsafe_allow_html=True)
age = st.sidebar.number_input("",0,120,key="age")

symptoms = st.sidebar.multiselect(
    "Symptoms",
    ["Fever","Cough","Chest Pain","Breathing Difficulty","Fatigue"]
)

# -----------------------------
# GRAD-CAM FIXED
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
# PDF FUNCTION (FIXED)
# -----------------------------
def generate_pdf(name, age, disease, conf, grad_path):
    file = f"/tmp/report_{uuid.uuid4().hex}.pdf"

    doc = SimpleDocTemplate(file, pagesize=A4)
    styles = getSampleStyleSheet()

    elements = []

    elements.append(Paragraph("<b>AI MEDICAL REPORT</b>", styles["Title"]))
    elements.append(Spacer(1,20))

    elements.append(Paragraph(f"Name: {name}", styles["Normal"]))
    elements.append(Paragraph(f"Age: {age}", styles["Normal"]))
    elements.append(Paragraph(f"Disease: {disease}", styles["Normal"]))
    elements.append(Paragraph(f"Confidence: {conf:.2f}%", styles["Normal"]))

    elements.append(Spacer(1,20))

    if os.path.exists(grad_path):
        elements.append(RLImage(grad_path, width=4*inch, height=4*inch))

    doc.build(elements)

    return file

# -----------------------------
# UPLOAD SECTION
# -----------------------------
st.markdown("<h2 style='font-size:40px;'>📤 Upload Chest X-ray</h2>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg","png","jpeg"])

# -----------------------------
# PREDICTION
# -----------------------------
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    img_resized = img.resize((224,224))
    arr = np.expand_dims(np.array(img_resized)/255, axis=0)

    preds = model.predict(arr)
    disease = CLASS_NAMES[np.argmax(preds)]
    conf = np.max(preds)*100

    st.markdown(f"""
    <div style="font-size:28px; font-weight:700;">
    🧠 Prediction: {disease} <br>
    🎯 Confidence: {conf:.2f}%
    </div>
    """, unsafe_allow_html=True)

    # -----------------------------
    # SMALL CHART (FIXED SIZE)
    # -----------------------------
    fig, ax = plt.subplots(figsize=(4,3))
    ax.bar(CLASS_NAMES, preds[0])
    ax.set_title("Confidence")
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
    # PDF DOWNLOAD (FIXED)
    # -----------------------------
    if name:
        pdf = generate_pdf(name, age, disease, conf, grad_path)

        with open(pdf, "rb") as f:
            st.download_button(
                "📄 Download PDF Report",
                f,
                file_name="AI_Report.pdf",
                mime="application/pdf"
            )
