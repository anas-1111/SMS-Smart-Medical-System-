import os
import io
import streamlit as st
from PIL import Image
import pytesseract
import pdfplumber
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.tavily import TavilyTools

# ------------------ OCR Configuration ------------------
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
os.environ["TESSDATA_PREFIX"] = r"C:\\Program Files\\Tesseract-OCR\\tessdata"

# ------------------ API Keys ------------------
os.environ["GOOGLE_API_KEY"] = "AIzaSyAF41ddLrbz3A_iSxq9WanmswFm8Nep1pU"
os.environ["TAVILY_API_KEY"] = "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"

# ------------------ Tavily Search Tool ------------------
web_tool = TavilyTools()

# ------------------ Gemini Agent Initialization ------------------
medical_agent = Agent(
    model=Gemini(id="gemini-2.5-flash"),
    tools=[web_tool],
    markdown=True
)

# ------------------ OCR Functions ------------------
def ocr_from_image(pil_image):
    """Extract text from image using Tesseract OCR."""
    return pytesseract.image_to_string(pil_image, lang='eng+ara')

def ocr_from_pdf(file_bytes):
    """Extract text from all pages in a PDF file."""
    text = ""
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# ------------------ Streamlit UI ------------------
st.title("Smart Clinic — Web-Enhanced Lab Analysis Assistant")

# Language Selection
language = st.radio(
    "Select report language:",
    ("Arabic", "English")
)

# Report Type Selection
report_type = st.radio(
    "Select report type:",
    ("Detailed Medical Report", "Friendly Patient Report")
)

# File Upload
uploaded = st.file_uploader("Upload a lab report (Image or PDF):", type=["png", "jpg", "jpeg", "pdf"])

if uploaded:
    st.info("Processing file...")

    # Extract text from file
    if uploaded.type == "application/pdf":
        text = ocr_from_pdf(uploaded.getvalue())
    else:
        img = Image.open(uploaded)
        text = ocr_from_image(img)

    st.success("Text extracted successfully from the report.")

    # ------------------ Prompt Setup ------------------
    if language == "Arabic":
        if report_type == "Detailed Medical Report":
            prompt = f"""
قم بتحليل النص التالي من تقرير طبي، وابحث في الإنترنت (مثل Mayo Clinic، WebMD، NIH)
عن معلومات موثوقة لتفسير النتائج.

النص المستخرج من التحليل:
{text}

اكتب تقريرًا طبيًا باللغة العربية يحتوي على:
1. قائمة بكل التحاليل وقيمها.
2. تحديد إن كانت النتيجة منخفضة أو طبيعية أو مرتفعة.
3. تفسير علمي مختصر لكل تحليل (سطر أو سطرين).
4. مراجع قصيرة من مصادر موثوقة.
5. الإشارة إلى وجود أي خطورة في التحاليل أو أي نسب غير طبيعية مع توضيح تأثيرها استنادًا إلى البحث.
"""
        else:
            prompt = f"""
قم بقراءة هذا التقرير الطبي:

{text}

ثم ابحث في الإنترنت عن معلومات موثوقة (مثل Mayo Clinic أو WebMD)
واكتب تقريرًا مبسطًا باللغة العربية يتضمن:
- شرحًا مبسطًا وسهل الفهم للنتائج.
- نصائح صحية عامة بناءً على النتائج.
- تقييم للحالة الصحية من 1 إلى 10.
- رسالة ختامية إيجابية للمريض.
"""
    else:
        if report_type == "Detailed Medical Report":
            prompt = f"""
Analyze the following medical lab report and search trusted websites
(Mayo Clinic, WebMD, NIH) for accurate explanations.

Extracted text:
{text}

Write a detailed medical report in English including:
1. A list of all lab tests and their values.
2. Classification (Low / Normal / High).
3. Evidence-based explanations (1–2 lines each).
4. Short references to trusted medical sources.
5. Highlight any risks or abnormal results and explain their possible implications based on medical research.
"""
        else:
            prompt = f"""
Read the following lab report:

{text}

Search reliable medical sources (Mayo Clinic, WebMD, NIH)
and write a friendly English summary that includes:
- Simple explanations for the results.
- 2–3 practical health tips.
- A health score (1–10).
- A short positive closing message for the patient.
"""

    # ------------------ Generate Report ------------------
    st.info("Generating report using live web data...")
    model_out = medical_agent.run(prompt)

    # Display Final Report
    st.subheader("Final Report:")
    st.markdown(model_out.content)
    st.success("Report generated successfully.")
    