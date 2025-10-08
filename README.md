
#  Smart Clinic — Web-Enhanced Lab Analysis Assistant

This project is an **AI-powered medical report analyzer** that uses **OCR, Google Gemini**, and **Tavily Web Search** to read and interpret lab reports (PDF or images).  
It generates either a **detailed medical report** or a **friendly summary** — in **Arabic or English**, based on user choice.

---

##  Features

-  Extracts text from PDF or image reports using **Tesseract OCR**  
-  Fetches live medical info using **Tavily Web Search API**
-  Analyzes data via **Google Gemini (2.5 Flash)**
-  Supports **Arabic** and **English**
-  Generates two types of reports:
  - **Detailed Medical Report**
  - **Friendly Patient Summary**

---

## ⚙️ Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/smart-clinic.git
cd smart-clinic
```

### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 3️⃣ Set Up Tesseract OCR
- Download from: https://github.com/UB-Mannheim/tesseract/wiki
- Default path (Windows):
  ```python
  pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
  ```

---

## 🔑 Environment Variables

| Variable | Description |
|-----------|-------------|
| `GOOGLE_API_KEY` | API key for Google Gemini |
| `TAVILY_API_KEY` | API key for Tavily Web Search |
| `TESSDATA_PREFIX` | Path to Tesseract data folder |

---

##  How It Works

1. Upload a **PDF** or **image** of a medical lab report.  
2. The system uses **OCR** to extract text.  
3. The **Gemini AI Agent** processes and interprets the data.  
4. The agent uses **Tavily Web Search** to find trusted medical information.  
5. Generates a **final AI-written report** (Detailed or Friendly).  

---

##  Project Structure

```
smart-clinic/
│
├── app.py                # Main Streamlit app
├── requirements.txt      # Python dependencies
├── README.md             # Documentation
└── /assets               # (Optional) Images or example reports
```

---

##  Technologies Used

- **Python 3.9+**
- **Streamlit**
- **Pytesseract + pdfplumber**
- **Google Gemini (via agno.models.google)**
- **TavilyTools (agno.tools.tavily)**

---

## 💻 Run the App

```bash
streamlit run app.py
```

Then open your browser at:
```
http://localhost:8501
```

---

## 🌐 Example Output

**Input:** A blood test report (PDF)  
**Output:**  
✅ Clean, structured interpretation of lab results  
✅ Summary of abnormal findings  
✅ Medical explanations from verified online sources  
✅ Friendly advice (if selected)

---

## 🧾 License

This project is released under the **MIT License**.

---

**Author:** Ahmed Elboos  
**Year:** 2025  

