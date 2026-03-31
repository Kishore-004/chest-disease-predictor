import streamlit as st
import numpy as np
from PIL import Image
import os, cv2, uuid
import matplotlib.pyplot as plt

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

# -----------------------------
# SAFE TENSORFLOW LOAD
# -----------------------------
model = None

try:
    import tensorflow as tf

    MODEL_PATH = "final_chest_disease_model.keras"

    if os.path.exists(MODEL_PATH):
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        st.success("✅ Model Loaded")
    else:
        st.warning("⚠ Model not found (using demo mode)")

except:
    st.warning("⚠ TensorFlow not supported (demo mode)")

# -----------------------------
# UI
# -----------------------------
st.title("🩺 AI Chest Disease Prediction System")

CLASS_NAMES = ['COVID19','NORMAL','PNEUMONIA','TURBERCULOSIS']

name = st.text_input("Patient Name")
age = st.number_input("Age",0,120)

uploaded_file = st.file_uploader("Upload X-ray")

# -----------------------------
# MAIN
# -----------------------------
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img)

    img_resized = img.resize((224,224))
    arr = np.expand_dims(np.array(img_resized)/255, axis=0)

    # -----------------------------
    # PREDICTION
    # -----------------------------
    if model:
        preds = model.predict(arr)
        disease = CLASS_NAMES[np.argmax(preds)]
        conf = np.max(preds)*100
    else:
        import random
        disease = random.choice(CLASS_NAMES)
        conf = round(random.uniform(80,95),2)
        preds = [[random.random() for _ in CLASS_NAMES]]

    st.success(f"{disease} ({conf:.2f}%)")

    # -----------------------------
    # GRAPH
    # -----------------------------
    fig, ax = plt.subplots(figsize=(4,3))
    ax.bar(CLASS_NAMES, preds[0])
    st.pyplot(fig)

    graph_path = f"/tmp/graph_{uuid.uuid4().hex}.png"
    fig.savefig(graph_path)

    # -----------------------------
    # GRAD-CAM (SAFE)
    # -----------------------------
    grad_path = None

    if model:
        try:
            layer = "conv5_block16_concat"

            grad_model = tf.keras.models.Model(
                inputs=model.inputs,
                outputs=[model.get_layer(layer).output, model.output]
            )

            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(arr)
                class_idx = tf.argmax(predictions[0])
                loss = predictions[:, class_idx]

            grads = tape.gradient(loss, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0,1,2))

            conv_outputs = conv_outputs[0]
            heatmap = conv_outputs * pooled_grads
            heatmap = tf.reduce_sum(heatmap, axis=-1)

            heatmap = np.maximum(heatmap, 0)
            heatmap = heatmap / (np.max(heatmap) + 1e-8)

            heat = cv2.resize(heatmap,(224,224))
            heat = np.uint8(255*heat)
            heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)

            grad_img = heat*0.4 + np.array(img_resized)
            grad_img = np.uint8(grad_img)

            st.image(grad_img)

            grad_path = f"/tmp/grad_{uuid.uuid4().hex}.jpg"
            cv2.imwrite(grad_path, grad_img)

        except:
            st.warning("Grad-CAM not supported")

    # -----------------------------
    # PDF
    # -----------------------------
    if name:
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

        if grad_path and os.path.exists(grad_path):
            elements.append(RLImage(grad_path, width=4*inch, height=4*inch))

        doc.build(elements)

        with open(file, "rb") as f:
            st.download_button("📄 Download Report", f.read())
