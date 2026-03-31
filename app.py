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
st.write("✅ App started successfully")

# -----------------------------
# SUPABASE CONFIG
# -----------------------------
SUPABASE_URL = "https://zpizagggcjbfgbixnchx.supabase.co"
SUPABASE_KEY = "YOUR_NEW_SAFE_KEY"  # ⚠️ Replace with new key

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    st.write("✅ Supabase connected")
except Exception as e:
    st.error(f"❌ Supabase connection failed: {e}")
    st.stop()

# -----------------------------
# LOGIN SYSTEM
# -----------------------------
if "user" not in st.session_state:
    st.session_state.user = None

def login_page():
    st.title("🔐 Login")

    tab1, tab2 = st.tabs(["Login", "Sign Up"])

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
                st.error(f"Login failed: {e}")

    with tab2:
        new_email = st.text_input("New Email")
        new_password = st.text_input("New Password", type="password")

        if st.button("Sign Up"):
            try:
                supabase.auth.sign_up({
                    "email": new_email,
                    "password": new_password
                })
                st.success("Account created! Now login")
            except Exception as e:
                st.error(f"Signup failed: {e}")

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
st.title("🩺 AI Healthcare System")

# -----------------------------
# MODEL SETUP (FIXED)
# -----------------------------
MODEL_PATH = "model.keras"
FILE_ID = "1GRO5EwB9PDX61G1lZfIHChvCK7JkYe6v"
CLASS_NAMES = ['COVID19','NORMAL','PNEUMONIA','TUBERCULOSIS']

@st.cache_resource
def load_model():
    try:
        if not os.path.exists(MODEL_PATH):
            st.warning("⬇️ Downloading model...")
            gdown.download(id=FILE_ID, output=MODEL_PATH, quiet=False)

        st.write("📁 Model exists:", os.path.exists(MODEL_PATH))

        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model

    except Exception as e:
        st.error(f"❌ Model loading failed: {e}")
        return None

model = load_model()

if model is None:
    st.stop()
else:
    st.success("✅ Model loaded successfully")

# -----------------------------
# INPUT
# -----------------------------
name = st.sidebar.text_input("Patient Name")
age = st.sidebar.number_input("Age", 0, 120)

# -----------------------------
# SAFE GRAD-CAM
# -----------------------------
def gradcam(img_array):
    try:
        # automatically find last conv layer
        last_conv_layer = None
        for layer in reversed(model.layers):
            if "conv" in layer.name:
                last_conv_layer = layer.name
                break

        if last_conv_layer is None:
            return None

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            class_idx = tf.argmax(predictions[0])
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs * pooled_grads
        heatmap = tf.reduce_sum(heatmap, axis=-1)

        heatmap = np.maximum(heatmap, 0)
        heatmap /= (np.max(heatmap) + 1e-8)

        return heatmap

    except Exception as e:
        st.warning(f"Grad-CAM skipped: {e}")
        return None

# -----------------------------
# PDF FUNCTION
# -----------------------------
def generate_pdf(name, age, disease, conf, grad_path):
    file = f"/tmp/report_{uuid.uuid4().hex}.pdf"

    doc = SimpleDocTemplate(file, pagesize=A4)
    styles = getSampleStyleSheet()

    elements = []

    elements.append(Paragraph("AI Medical Report", styles["Title"]))
    elements.append(Spacer(1,20))

    elements.append(Paragraph(f"Name: {name}", styles["Normal"]))
    elements.append(Paragraph(f"Age: {age}", styles["Normal"]))
    elements.append(Paragraph(f"Disease: {disease}", styles["Normal"]))
    elements.append(Paragraph(f"Confidence: {conf:.2f}%", styles["Normal"]))

    elements.append(Spacer(1,20))

    if grad_path and os.path.exists(grad_path):
        elements.append(RLImage(grad_path, width=4*inch, height=4*inch))

    doc.build(elements)
    return file

# -----------------------------
# UPLOAD & PREDICTION
# -----------------------------
file = st.file_uploader("Upload Chest X-ray")

if file:
    try:
        img = Image.open(file).convert("RGB")

        img_resized = img.resize((224,224))
        arr = np.expand_dims(np.array(img_resized)/255, axis=0)

        st.write("📊 Input shape:", arr.shape)

        preds = model.predict(arr)

        disease = CLASS_NAMES[np.argmax(preds)]
        conf = np.max(preds)*100

        st.success(f"{disease} ({conf:.2f}%)")

        col1, col2 = st.columns(2)

        with col1:
            st.image(img)

        with col2:
            fig, ax = plt.subplots()
            ax.bar(CLASS_NAMES, preds[0])
            st.pyplot(fig)

        # Grad-CAM
        heat = gradcam(arr)
        grad_path = None

        if heat is not None:
            heat = cv2.resize(heat, (224,224))
            heat = np.uint8(255*heat)
            heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

            grad_img = heat*0.4 + np.array(img_resized)
            grad_img = np.uint8(grad_img)

            st.image(grad_img)

            grad_path = f"/tmp/grad_{uuid.uuid4().hex}.jpg"
            cv2.imwrite(grad_path, grad_img)

        # PDF
        if name:
            pdf = generate_pdf(name, age, disease, conf, grad_path)

            with open(pdf, "rb") as f:
                pdf_bytes = f.read()

            st.download_button(
                "📄 Download Report",
                data=pdf_bytes,
                file_name="AI_Report.pdf",
                mime="application/pdf"
            )

    except Exception as e:
        st.error(f"❌ Prediction error: {e}")
