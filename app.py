import streamlit as st

st.title("✅ App Working")

name = st.text_input("Enter your name")

if name:
    st.success(f"Hello {name}")
