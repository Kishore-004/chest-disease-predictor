import streamlit as st
import numpy as np
from PIL import Image
import gdown, os, cv2
import tensorflow as tf

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI Healthcare", layout="wide")

# -----------------------------
# GLOBAL CSS (FONT + BOLD)
# -----------------------------
st.markdown("""
<style>
html, body, [class*="css"]  {
    font-size: 16px !important;
    font-weight: 600;
}

.card {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# HEADER
# -----------------------------
st.title("🩺 AI Healthcare System")
st.markdown("### Disease Detection + Explanation + Hospital Suggestion")
st.markdown("---")

# -----------------------------
# MODEL CONFIG
# -----------------------------
MODEL_PATH = "model.tflite"
FILE_ID = "1CBdRBXsze5YgdbRnC8H3GYtqLlydeF-j"

CLASS_NAMES = ['COVID19','NORMAL','PNEUMONIA','TURBERCULOSIS']

# -----------------------------
# DISEASE INFO (STRUCTURED)
# -----------------------------
DISEASE_INFO = {
    "TURBERCULOSIS": {
        "description": "Tuberculosis is a serious infection that mainly affects the lungs.",
        "symptoms": "Long-term cough, weight loss, night sweats, fatigue.",
        "causes": "Caused by TB bacteria spread through air.",
        "treatment": "Requires antibiotics for 6–9 months continuously.",
        "precautions": "Early diagnosis, avoid close contact, complete treatment."
    },
    "PNEUMONIA": {
        "description": "Pneumonia is a lung infection causing fluid buildup.",
        "symptoms": "Fever, chest pain, cough with mucus.",
        "causes": "Bacteria, virus, or fungi.",
        "treatment": "Antibiotics and rest.",
        "precautions": "Vaccination and hygiene."
    },
    "COVID19": {
        "description": "COVID-19 is a viral respiratory disease.",
        "symptoms": "Fever, cough, breathing issues.",
        "causes": "Spread through droplets.",
        "treatment": "Rest and medical care.",
        "precautions": "Mask and vaccination."
    },
    "NORMAL": {
        "description": "Lungs appear normal.",
        "symptoms": "No symptoms.",
        "causes": "Healthy condition.",
        "treatment": "Not required.",
        "precautions": "Maintain healthy lifestyle."
    }
}

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH)
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -----------------------------
# SIDEBAR INPUTS
# -----------------------------
st.sidebar.header("👤 Patient Details")

name = st.sidebar.text_input("Name")
age = st.sidebar.number_input("Age", 0, 120)

symptoms = st.sidebar.multiselect(
    "Symptoms",
    ["Fever","Cough","Chest Pain","Breathing Difficulty","Fatigue","Weight Loss","Night Sweats"]
)

uploaded_file = st.sidebar.file_uploader("Upload X-ray")

# -----------------------------
# MAIN OUTPUT
# -----------------------------
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_resized = img.resize((224,224))

    arr = np.expand_dims(np.array(img_resized)/255, axis=0).astype('float32')

    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])

    disease = CLASS_NAMES[np.argmax(preds[0])]
    confidence = np.max(preds[0]) * 100

    # -----------------------------
    # RESULT
    # -----------------------------
    st.success(f"🧠 Prediction: {disease}")
    st.info(f"📊 Confidence: {confidence:.2f}%")

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Uploaded X-ray")

    # -----------------------------
    # 📖 CLEAN EXPLANATION
    # -----------------------------
    info = DISEASE_INFO[disease]

    with col2:
        st.markdown("## 📖 Disease Details")

        st.markdown(f"""
        <div class="card">
        <b>🦠 Description:</b><br>{info['description']}<br><br>
        
        <b>🤒 Symptoms:</b><br>{info['symptoms']}<br><br>
        
        <b>🧬 Causes:</b><br>{info['causes']}<br><br>
        
        <b>💊 Treatment:</b><br>{info['treatment']}<br><br>
        
        <b>🛡️ Precautions:</b><br>{info['precautions']}
        </div>
        """, unsafe_allow_html=True)
