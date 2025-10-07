import os
from PIL import Image as PILImage
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.media import Image as AgnoImage
import streamlit as st

# ===========================
# API Key Setup
# ===========================
GOOGLE_API_KEY = "AIzaSyCqnC5HjMGRW6QXXY-PharEHZb9uYlAsRM"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

if not GOOGLE_API_KEY:
    raise ValueError("Please set your Google API Key in GOOGLE_API_KEY")

# ===========================
# Agent Initialization
# ===========================
medical_agent = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    tools=[DuckDuckGoTools()],
    markdown=True
)

# ===========================
# Prompt Builder
# ===========================
def build_prompt(language, report_type):
    if language == "🇬🇧 English":
        if report_type == "Professional Medical Report":
            return """
You are a highly skilled medical imaging expert with extensive knowledge in radiology and diagnostic imaging. Analyze the medical image and structure your response as follows:

### 1. Image Type & Region
- Identify imaging modality (X-ray/MRI/CT/Ultrasound/etc.).
- Specify anatomical region and positioning.
- Evaluate image quality and technical adequacy.

### 2. Key Findings
- Highlight primary observations systematically.
- Identify potential abnormalities with detailed descriptions.
- Include measurements and densities where relevant.

### 3. Diagnostic Assessment
- Provide primary diagnosis with confidence level.
- List differential diagnoses ranked by likelihood.
- Support each diagnosis with observed evidence.
- Highlight critical or urgent findings.

### 4. Patient-Friendly Explanation
- Simplify findings in clear, non-technical language.
- Avoid medical jargon or provide easy definitions.
- Include relatable analogies.

### 5. Research Context
- Use DuckDuckGo search to find recent medical literature.
- Search for standard treatment protocols.
- Provide 2–3 references supporting the analysis.

Ensure the response is structured, professional, and medically accurate using clean markdown formatting.
"""
        else:
            return """
You are a friendly medical assistant. Review the medical image and explain it in simple language.

Please:
- Describe the image (type and body part).
- Explain if it looks normal or shows any issues.
- Use clear and simple wording (avoid medical jargon).
- Add helpful health advice and end with a positive note.
"""
    else:
        # Arabic version
        if report_type == "تقرير طبي احترافي":
            return """
أنت خبير في الأشعة الطبية. قم بتحليل الصورة الطبية التالية وفق الهيكل التالي:

### 1. نوع الصورة والمنطقة
- حدد نوع الفحص (أشعة سينية / رنين مغناطيسي / مقطعية / موجات صوتية...).
- حدد المنطقة التشريحية ووضعية التصوير.
- قيّم جودة الصورة ودقتها التقنية.

### 2. الملاحظات الأساسية
- استعرض أهم الملاحظات بشكل منظم.
- وضّح أي تغيرات أو علامات غير طبيعية.
- أضف القياسات أو الكثافات إن وُجدت.

### 3. التقييم التشخيصي
- قدّم التشخيص الأساسي مع نسبة الثقة.
- أضف التشخيصات التفريقية مرتبة حسب الاحتمال.
- اربط كل تشخيص بالأدلة الظاهرة في الصورة.
- نوّه لأي ملاحظات حرجة أو عاجلة.

### 4. تبسيط النتائج للمريض
- اشرح النتائج بلغة مبسطة وواضحة.
- تجنّب المصطلحات الطبية أو فسّرها ببساطة.
- استخدم أمثلة أو تشبيهات لتسهيل الفهم.

### 5. المرجع العلمي
- استخدم أداة DuckDuckGo للبحث عن أحدث المراجع الطبية.
- اذكر البروتوكولات العلاجية الشائعة.
- أضف 2 إلى 3 مراجع علمية تدعم التحليل.

احرص أن يكون الرد منظمًا واحترافيًا بتنسيق Markdown واضح.
"""
        else:
            return """
أنت مساعد صحي ودود. انظر إلى الصورة الطبية وفسّرها بلغة بسيطة.

الرجاء:
- وصف نوع الصورة والمنطقة التي تُظهرها.
- ذكر ما إذا كانت الصورة طبيعية أو بها ملاحظات غير معتادة.
- استخدم لغة سهلة ومشجعة بدون مصطلحات طبية معقدة و لكن بتركيز.
- أضف نصيحة صحية بسيطة وختم بتوصية بمراجعة الطبيب المختص.
"""

# ===========================
# Image Analysis Function
# ===========================
def analyze_medical_image(image_path, language, report_type):
    image = PILImage.open(image_path)
    width, height = image.size
    aspect_ratio = width / height
    new_width = 500
    new_height = int(new_width / aspect_ratio)
    resized_image = image.resize((new_width, new_height))

    temp_path = "temp_resized_image.png"
    resized_image.save(temp_path)

    agno_image = AgnoImage(filepath=temp_path)
    query = build_prompt(language, report_type)

    try:
        response = medical_agent.run(query, images=[agno_image])
        return response.content
    except Exception as e:
        return f"Error: {e}"
    finally:
        os.remove(temp_path)

# ===========================
# Streamlit UI
# ===========================
st.set_page_config(page_title="Medical Image Analyzer", layout="centered")
st.title("Medical Image Analysis Tool")

st.markdown("""
Welcome to the AI-powered **Medical Image Analysis Tool**.
Upload a medical image (X-ray, MRI, CT, Ultrasound, etc.) to generate either a professional medical report or a simplified patient-friendly summary.
""")

# Language selection
language = st.radio("Select Language:", ("🇪🇬 العربية", "🇬🇧 English"))

# Report type selection
if language == "🇪🇬 العربية":
    report_type = st.radio("اختر نوع التقرير:", ("تقرير طبي احترافي", "تقرير مبسط للمريض"))
else:
    report_type = st.radio("Select Report Type:", ("Professional Medical Report", "Friendly Patient Report"))

# Image upload
uploaded_file = st.file_uploader("Upload a medical image", type=["jpg", "jpeg", "png", "bmp", "gif"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", width='stretch')
    if st.button("Start Analysis"):
        with st.spinner("Analyzing image..."):
            image_path = f"temp_image.{uploaded_file.type.split('/')[1]}"
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            report = analyze_medical_image(image_path, language, report_type)

            st.subheader("Analysis Result")
            st.markdown(report, unsafe_allow_html=True)

            os.remove(image_path)
else:
    st.info("Please upload a medical image to begin analysis.")
