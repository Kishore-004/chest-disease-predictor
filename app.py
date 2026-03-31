import streamlit as st
from supabase import create_client, Client
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
# SUPABASE CONFIG (IMPORTANT)
# -----------------------------
SUPABASE_URL = "https://zpizagggcjbfgbixnchx.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."  # paste full key

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# -----------------------------
# LOGIN SYSTEM
# -----------------------------
if "user" not in st.session_state:
    st.session_state.user = None

def login_page():
    st.markdown("<h1 style='text-align:center;'>🔐 Login</h1>", unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

    # LOGIN
    with tab1:
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            try:
                user = supabase.auth.sign_in_with_password({
                    "email": email,
                    "password": password
                })
                st.session_state.user = user
                st.success("Login successful")
                st.rerun()
            except Exception as e:
                st.error("Invalid credentials")

    # SIGNUP
    with tab2:
        new_email = st.text_input("New Email")
        new_password = st.text_input("New Password", type="password")

        if st.button("Sign Up"):
            try:
                supabase.auth.sign_up({
                    "email": new_email,
                    "password": new_password
                })
                st.success("Account created! Please login")
            except:
                st.error("Signup failed")

# BLOCK APP IF NOT LOGGED IN
if st.session_state.user is None:
    login_page()
    st.stop()

# -----------------------------
# LOGOUT
# -----------------------------
if st.sidebar.button("🚪 Logout"):
    st.session_state.user = None
    st.rerun()

# -----------------------------
# HEADER
# -----------------------------
st.markdown("<h1 style='text-align:center;'>🩺 AI Healthcare System</h1>", unsafe_allow_html=True)

# -----------------------------
# MODEL CONFIG
# -----------------------------
MODEL_PATH = "final_chest_disease_model.keras"
FILE_ID = "1GRO5EwB9PDX61G1lZfIHChvCK7JkYe6v"
CLASS_NAMES = ['COVID19','NORMAL','PNEUMONIA','TURBERCULOSIS']

if not os.path.exists(MODEL_PATH):
    gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH)

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# -----------------------------
# USER INPUT
# -----------------------------
name = st.sidebar.text_input("Patient Name")
age = st.sidebar.number_input("Age", 0, 120)

# -----------------------------
# GRAD-CAM FUNCTION
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
def generate_pdf(name, age, disease, conf, grad_path):
    file = f"/tmp/report_{uuid.uuid4().hex}.pdf"

    doc = SimpleDocTemplate(file, pagesize=A4)
    styles = getSampleStyleSheet()

    elements = []

    elements.append(Paragraph("AI MEDICAL REPORT", styles["Title"]))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph(f"Name: {name}", styles["Normal"]))
    elements.append(Paragraph(f"Age: {age}", styles["Normal"]))
    elements.append(Paragraph(f"Disease: {disease}", styles["Normal"]))
    elements.append(Paragraph(f"Confidence: {conf:.2f}%", styles["Normal"]))

    elements.append(Spacer(1, 20))

    if os.path.exists(grad_path):
        elements.append(RLImage(grad_path, width=4*inch, height=4*inch))

    doc.build(elements)
    return file

# -----------------------------
# UPLOAD IMAGE
# -----------------------------
file = st.file_uploader("Upload Chest X-ray")

if file:
    img = Image.open(file).convert("RGB")

    img_resized = img.resize((224,224))
    arr = np.expand_dims(np.array(img_resized)/255, axis=0)

    preds = model.predict(arr)
    disease = CLASS_NAMES[np.argmax(preds)]
    conf = np.max(preds)*100

    st.success(f"Prediction: {disease} ({conf:.2f}%)")

    # IMAGE + GRAPH
    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="X-ray")

    with col2:
        fig, ax = plt.subplots(figsize=(4,3))
        ax.bar(CLASS_NAMES, preds[0])
        st.pyplot(fig)

    # GRAD-CAM
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

    # PDF DOWNLOAD (FIXED)
    if name:
        pdf = generate_pdf(name, age, disease, conf, grad_path)

        with open(pdf, "rb") as f:
            pdf_bytes = f.read()

        st.download_button(
            "📄 Download PDF Report",
            data=pdf_bytes,
            file_name="AI_Report.pdf",
            mime="application/pdf"
        )
