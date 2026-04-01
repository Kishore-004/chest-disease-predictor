import streamlit as st
import numpy as np
from PIL import Image
import gdown, os, cv2
import tensorflow as tf
import random
import matplotlib.pyplot as plt

from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

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
html, body {
    font-size:16px;
    font-weight:600;
}
.card {
    background:white;
    padding:20px;
    border-radius:12px;
    margin-bottom:15px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.title("🩺 AI Healthcare System")
st.write("Disease Detection + Explanation + Hospital Suggestion")

# ---------------- DISEASE INFO ----------------
DISEASE_INFO = {
    "TURBERCULOSIS": {
        "desc": "Tuberculosis is a serious lung infection caused by bacteria that spreads through air.",
        "sym": "Long-term cough, weight loss, night sweats.",
        "cause": "TB bacteria (Mycobacterium tuberculosis).",
        "treat": "6–9 months antibiotics.",
        "prec": "Early diagnosis and complete treatment."
    },
    "PNEUMONIA": {
        "desc": "Pneumonia is a lung infection where air sacs fill with fluid.",
        "sym": "Fever, cough, chest pain.",
        "cause": "Bacteria or virus.",
        "treat": "Antibiotics and rest.",
        "prec": "Vaccination and hygiene."
    },
    "COVID19": {
        "desc": "COVID-19 is a viral respiratory infection affecting lungs.",
        "sym": "Fever, cough, breathing issues.",
        "cause": "Coronavirus.",
        "treat": "Rest and supportive care.",
        "prec": "Mask and vaccination."
    },
    "NORMAL": {
        "desc": "No lung abnormality detected.",
        "sym": "No symptoms.",
        "cause": "Healthy lungs.",
        "treat": "Not required.",
        "prec": "Healthy lifestyle."
    }
}

# ---------------- HOSPITALS ----------------
HOSPITALS = {
    "Chennai": [
        {"name":"Apollo Hospital","doc":"Dr. Ramesh (Pulmonologist)"},
        {"name":"MIOT International","doc":"Dr. Priya (Chest Specialist)"}
    ],
    "Madurai":[
        {"name":"Meenakshi Mission","doc":"Dr. Karthik"}
    ],
    "Coimbatore":[
        {"name":"KG Hospital","doc":"Dr. Vignesh"},
        {"name":"Ganga Hospital","doc":"Dr. Suresh"}
    ]
}

def maps_link(name,city):
    return f"https://www.google.com/maps/search/{name}+{city}"

def rating():
    return round(random.uniform(3.5,5.0),1)

# ---------------- LOAD MODELS ----------------
@st.cache_resource
def load_tflite():
    if not os.path.exists(MODEL_PATH):
        gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH)
    model = tf.lite.Interpreter(model_path=MODEL_PATH)
    model.allocate_tensors()
    return model

@st.cache_resource
def load_grad_model():
    if not os.path.exists(KERAS_PATH):
        gdown.download(f"https://drive.google.com/uc?id={KERAS_ID}", KERAS_PATH)
    return tf.keras.models.load_model(KERAS_PATH)

interpreter = load_tflite()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---------------- SIDEBAR ----------------
st.sidebar.header("👤 Patient Details")

name = st.sidebar.text_input("Name")
age = st.sidebar.number_input("Age",0,120)
uploaded = st.sidebar.file_uploader("Upload X-ray")

# ---------------- MAIN ----------------
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    img_r = img.resize((224,224))
    arr = np.expand_dims(np.array(img_r)/255,0).astype('float32')

    interpreter.set_tensor(input_details[0]['index'], arr)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])

    disease = CLASS_NAMES[np.argmax(preds[0])]
    conf = np.max(preds[0])*100

    st.success(f"🧠 Prediction: {disease}")
    st.info(f"📊 Confidence: {conf:.2f}%")

    col1,col2 = st.columns(2)

    with col1:
        st.image(img, caption="X-ray")

    # ---------------- EXPLANATION ----------------
    info = DISEASE_INFO[disease]

    with col2:
        st.markdown("## 📖 Disease Details")
        st.markdown(f"""
        <div class="card">
        🦠 Description: {info['desc']}<br><br>
        🤒 Symptoms: {info['sym']}<br><br>
        🧬 Causes: {info['cause']}<br><br>
        💊 Treatment: {info['treat']}<br><br>
        🛡️ Precautions: {info['prec']}
        </div>
        """, unsafe_allow_html=True)

    # ---------------- CHART ----------------
    st.subheader("📊 Prediction Chart")
    fig, ax = plt.subplots()
    ax.bar(CLASS_NAMES, preds[0])
    st.pyplot(fig)

    # ---------------- GRADCAM FIXED ----------------
    if st.button("🔥 Show Affected Area"):
        model = load_grad_model()

        last_conv = [l.name for l in model.layers if "conv" in l.name][-1]

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(arr)

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

        heatmap = cv2.resize(heatmap, (224,224))
        heatmap[heatmap < 0.6] = 0

        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        overlay = cv2.addWeighted(np.array(img_r), 0.6, heatmap, 0.7, 0)

        st.image(overlay, caption="🔥 Affected Lung Area")

    # ---------------- HOSPITAL FIXED ----------------
    st.markdown("## 🏥 Hospital Suggestions")
    city = st.text_input("Enter City")

    if city:
        city_clean = city.strip().title()

        if city_clean in HOSPITALS:
            for h in HOSPITALS[city_clean]:
                st.markdown(f"""
                <div class="card">
                🏥 {h['name']}<br>
                👨‍⚕️ {h['doc']}<br>
                ⭐ {rating()}/5<br>
                <a href="{maps_link(h['name'], city_clean)}" target="_blank">📍 View Map</a>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("❌ No hospitals found for this city")

    # ---------------- PDF ----------------
    if st.button("📄 Download Report"):
        file = "report.pdf"
        doc = SimpleDocTemplate(file)
        styles = getSampleStyleSheet()

        content = [
            Paragraph(f"Name: {name}", styles["Normal"]),
            Paragraph(f"Disease: {disease}", styles["Normal"]),
            Paragraph(f"Confidence: {conf:.2f}%", styles["Normal"])
        ]

        doc.build(content)

        with open(file,"rb") as f:
            st.download_button("Download PDF", f)
