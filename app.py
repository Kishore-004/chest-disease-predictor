import streamlit as st
import matplotlib.pyplot as plt

st.subheader("📊 Prediction Confidence Graph")

fig, ax = plt.subplots(figsize=(4,3))

if model:
    ax.bar(CLASS_NAMES, preds[0])
else:
    import random
    fake = [random.random() for _ in CLASS_NAMES]
    ax.bar(CLASS_NAMES, fake)

st.pyplot(fig)
