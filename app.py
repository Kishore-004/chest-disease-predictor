import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import gdown, os, cv2, time
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
st.markdown("""
<h1 style='text-align:center;font-size:50px;'>🩺 AI Healthcare System</h1>
<hr>
""", unsafe_allow_html=True)

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
    "Select Symptoms",
    ["Fever","Cough","Chest Pain","Breathing Difficulty","Fatigue"]
)

# -----------------------------
# UPLOAD
# -----------------------------
file = st.file_uploader("Upload X-ray")

# -----------------------------
# GRAD CAM
# -----------------------------
def gradcam(img_array):
    layer="conv5_block16_concat"
    grad_model=tf.keras.models.Model(
        [model.inputs],[model.get_layer(layer).output,model.output]
    )

    with tf.GradientTape() as tape:
        conv,preds=grad_model(img_array)
        loss=preds[:,np.argmax(preds[0])]

    grads=tape.gradient(loss,conv)
    pooled=tf.reduce_mean(grads,axis=(0,1,2))
    conv=conv[0]

    heat=conv*pooled
    heat=tf.reduce_sum(heat,axis=-1)

    heat=np.maximum(heat,0)/np.max(heat)
    return heat

# -----------------------------
# PDF FUNCTION
# -----------------------------
def generate_pdf(name,age,disease,conf,grad_path):
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
# PREDICTION
# -----------------------------
if file:
    img=Image.open(file).convert("RGB")
    st.image(img)

    img_r=img.resize((224,224))
    arr=np.expand_dims(np.array(img_r)/255,axis=0)

    pred=model.predict(arr)
    disease=CLASS_NAMES[np.argmax(pred)]
    conf=np.max(pred)*100

    st.success(f"{disease} ({conf:.2f}%)")

    # -----------------------------
    # CHART (ANALYTICS)
    # -----------------------------
    fig=plt.figure()
    plt.bar(CLASS_NAMES,pred[0])
    st.pyplot(fig)

    # -----------------------------
    # GRADCAM
    # -----------------------------
    heat=gradcam(arr)
    heat=cv2.resize(heat,(224,224))
    heat=np.uint8(255*heat)
    heat=cv2.applyColorMap(heat,cv2.COLORMAP_JET)

    grad_img=np.array(img_r)
    grad_img=heat*0.4+grad_img

    st.image(grad_img,caption="Grad-CAM")

    # save
    grad_path="/tmp/grad.jpg"
    cv2.imwrite(grad_path,grad_img)

    # -----------------------------
    # MAP (HOSPITAL LINKS)
    # -----------------------------
    city=st.text_input("Enter City")

    if city:
        st.write("Hospitals:")
        st.markdown(f"[Search Hospitals](https://www.google.com/maps/search/hospital+in+{city})")

    # -----------------------------
    # DOWNLOAD PDF
    # -----------------------------
    if name:
        pdf=generate_pdf(name,age,disease,conf,grad_path)

        with open(pdf,"rb") as f:
            st.download_button("Download Report",f)
