st.markdown("""
<style>

/* FORCE APPLY FONT + SIZE EVERYWHERE */
html, body, [class*="st-"], [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    font-size: 20px !important;
}

/* HEADINGS */
h1 {
    font-size: 50px !important;
    font-weight: 700 !important;
}

h2 {
    font-size: 34px !important;
}

h3 {
    font-size: 28px !important;
}

/* SIDEBAR TEXT */
section[data-testid="stSidebar"] * {
    font-size: 20px !important;
}

/* INPUTS */
input, textarea {
    font-size: 18px !important;
}

/* BUTTON */
button {
    font-size: 18px !important;
}

/* FILE UPLOADER */
[data-testid="stFileUploader"] * {
    font-size: 18px !important;
}

/* ALERT BOXES */
[data-testid="stAlert"] * {
    font-size: 18px !important;
}

/* LABEL TEXT */
label {
    font-size: 20px !important;
}

/* PARAGRAPH */
p, span, div {
    font-size: 20px !important;
}

</style>
""", unsafe_allow_html=True)
