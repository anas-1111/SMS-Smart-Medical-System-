import streamlit as st
from PIL import Image
from io import BytesIO
import time
import google.generativeai as genai

# -------------------- GEMINI SETUP --------------------
# 🔹 Replace YOUR_API_KEY with your actual Gemini API key
genai.configure(api_key="AIzaSyCVlbh7euN0VtZIepD-NtI3PTcZtoQ95P4")

model = genai.GenerativeModel("models/gemini-2.5-flash")

# -------------------- IMPORT YOUR SCRAPER + LOGIC --------------------
from main.medicine_core import get_or_generate_medicine_report, model
# (replace 'your_module' with the filename where your scraping + fallback functions exist, e.g. medicine_core)

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="💊 Medicine Assistant", page_icon="💊", layout="centered")

st.title("💊 Smart Medicine Information Assistant")
st.markdown("""
Upload a photo **or** type a medicine name,  
and the app will fetch verified data from **Drugs.com**,  
or automatically generate a **bilingual report (English + Arabic)** using Gemini if not found.
""")

# -------------------- INPUTS --------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("📷 Upload Medicine Image")
    uploaded_img = st.file_uploader("Upload a clear photo", type=["jpg", "jpeg", "png"])

with col2:
    st.subheader("⌨️ Or Type Medicine Name")
    typed_name = st.text_input("Enter medicine name:", placeholder="e.g. Panadol, Gaviscon, Safetrium")

final_name = None

# -------------------- OCR FROM IMAGE --------------------
if uploaded_img:
    with st.spinner("🔍 Extracting medicine name from image using Gemini..."):
        try:
            image = Image.open(uploaded_img)
            response = model.generate_content([
                "Extract ONLY the brand/medicine name (biggest text) from this medicine package image. \
                with small description to know more about it, if ambitious.",
                image
                ])

            final_name = response.text.strip()
            st.success(f"✅ Detected medicine name: **{final_name}**")
        except Exception as e:
            st.error(f"⚠️ Could not extract text from image: {e}")

elif typed_name.strip():
    final_name = typed_name.strip()

# -------------------- GENERATE REPORT --------------------
if final_name:
    if st.button("🧾 Generate Medicine Report"):
        with st.spinner("⏳ Fetching or generating report..."):
            try:
                report = get_or_generate_medicine_report(final_name)
                st.markdown(report, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ Error generating report: {str(e)}")
else:
    st.info("💡 Please upload an image or type a medicine name to start.")

# -------------------- FOOTER --------------------
st.markdown("---")
st.caption("⚠️ Not medical advice — consult a doctor before use.")
