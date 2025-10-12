from flask import Flask, request, jsonify, render_template, session, flash, redirect, url_for
from flask_cors import CORS
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
import pyodbc
import re
import base64
import face_recognition
import numpy as np
from io import BytesIO
from PIL import Image
import datetime  
from functools import wraps
from routes.medicine import medicine_bp
import os
import traceback
from utils.rate_limiter import exponential_backoff, RateLimitExceededError, parse_retry_delay
from agno.agent import Agent
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.media import Image as AgnoImage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryBufferMemory,
    CombinedMemory
)
from pydantic import SecretStr
from threading import Lock
import pytesseract
import pdfplumber
import io
_medical_agent = None
_lab_analysis_agent = None
_agent_lock = Lock()

# Configure Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
os.environ["TESSDATA_PREFIX"] = r"C:\\Program Files\\Tesseract-OCR\\tessdata"


# Path configurations
SYMPTOM_DB_PATH = "vectorstore_medibot/db_faiss"
FIRST_AID_DB_PATH = "backend/vectorstore/db_faiss"
custom_prompt = """
You are a friendly and knowledgeable virtual medical assistant chatbot.

Your goal is to help users understand their symptoms, offer basic first aid or home-care advice, and guide them on when to seek professional medical help.

Use the information provided in your context or knowledge base to answer questions. 
If the context does not contain enough relevant information, respond politely:

"I don't have enough verified medical information to answer that accurately. It's best to consult a qualified healthcare professional for proper evaluation."

Response Guidelines:
- Do NOT invent, assume, or speculate about medical facts or diagnoses.
- Provide clear, calm, and easy-to-follow steps or bullet points whenever possible.
- Always include a reminder that you are not a doctor and your guidance is not a substitute for professional medical advice.
- If the symptoms sound severe, urgent, or life-threatening, instruct the user to seek emergency care or call local emergency services immediately.
- Use empathetic and encouraging language — help the user feel supported, not alarmed.
- Explain medical terms in simple language.
- Keep responses concise, practical, and focused on what the user can safely do right now.

Tone & Style:
- Friendly, reassuring, and professional.
- Speak clearly, avoiding jargon unless you explain it.
- Structure responses with short paragraphs or bullet points for easy reading.

Previous discussion (summary):
{chat_summary}

Context:
{context}

Question:
{question}

Answer:
"""

# Initialize Google API Key for medical image analysis
GOOGLE_API_KEY = "your-google-api-key-here"      
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


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

def get_medical_agent():
    """Initialize and return the medical analysis agent with rate limiting"""
    global _medical_agent
    
    with _agent_lock:
        if _medical_agent is None:
            print("🔄 Initializing medical agent")
            _medical_agent = Agent(
                model=Gemini(id="gemini-2.0-flash-exp"),
                tools=[DuckDuckGoTools()],
                markdown=True
            )
        return _medical_agent

def get_lab_analysis_agent():
    """Initialize and return the lab analysis agent"""
    global _lab_analysis_agent
    
    with _agent_lock:
        if _lab_analysis_agent is None:
            print("🔄 Initializing lab analysis agent")
            web_tool = DuckDuckGoTools()
            _lab_analysis_agent = Agent(
                model=Gemini(id="gemini-2.5-flash"),
                tools=[web_tool],
                markdown=True
            )
        return _lab_analysis_agent

app = Flask(__name__, template_folder='templates')
CORS(app)

# Initialize Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'render_login'

# Register blueprints
app.register_blueprint(medicine_bp)

app.secret_key = 'your-secret-key-here'  
app.config['TEMPLATES_AUTO_RELOAD'] = True

# User class for Flask-Login
class User(UserMixin):
    def __init__(self, user_id):
        self.id = user_id

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

# Login required decorator
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            flash('Please log in first.', 'warning')
            return redirect(url_for('render_login'))
        return f(*args, **kwargs)
    return decorated_function

# ===============================
# Template Routes
# ===============================

@app.route('/')
def splash():
    if 'logged_in' in session:
        return redirect(url_for('render_home'))
    return render_template('splash.html')

@app.route('/home')
@login_required
def render_home():
    return render_template('home.html')

@app.route('/login')
def render_login():
    return render_template('login.html')

@app.route('/register')
def render_register():
    return render_template('register.html')

@app.route('/profile')
@login_required
def render_profile():
    patient_id = session.get('patient_id')
    if not patient_id:
        return redirect(url_for('render_login'))

    # Get profile data
    profile_response = get_patient_profile(patient_id)
    profile_data = profile_response[0].get_json() if profile_response[1] == 200 else None

    # Get medical history
    history_response = get_medical_history(patient_id)
    history_data = history_response[0].get_json() if history_response[1] == 200 else None

    return render_template('profile.html', 
                         profile=profile_data,
                         medical_history=history_data)

@app.route('/get_analysis/<int:analysis_id>')
@login_required
def get_analysis(analysis_id):
    patient_id = session.get('patient_id')
    if not patient_id:
        return jsonify({"error": "User not logged in"}), 401
        
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get the analysis details
        cursor.execute("""
            SELECT analysis_id, analysis_type, language, analysis_date,
                   image_type, anatomical_region, key_findings,
                   primary_diagnosis, diagnostic_assessment,
                   patient_explanation, research_context,
                   CAST(image_data AS VARBINARY(MAX)) as image_data
            FROM image_analysis 
            WHERE analysis_id = ? AND patient_id = ?
        """, (analysis_id, patient_id))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return jsonify({"error": "Analysis not found"}), 404
            
        # Convert image data to base64
        image_base64 = base64.b64encode(row[11]).decode('utf-8') if row[11] else None
        
        return jsonify({
            "analysis_id": row[0],
            "analysis_type": row[1],
            "language": row[2],
            "date": row[3].strftime("%Y-%m-%d %H:%M:%S"),
            "image_type": row[4],
            "anatomical_region": row[5],
            "key_findings": row[6],
            "primary_diagnosis": row[7],
            "diagnostic_assessment": row[8],
            "patient_explanation": row[9],
            "research_context": row[10],
            "image_data": f"data:image/png;base64,{image_base64}" if image_base64 else None
        }), 200
        
    except Exception as e:
        print(f"Error fetching analysis: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/appointments', methods=['GET'])
@login_required
def render_appointments():
    return render_template('appointments.html')

@app.route('/medical-image-analyzer')
@login_required
def render_medical_analyzer():
    return render_template('medical_analyzer.html')

@app.route('/lab-analyzer')
@login_required
def render_lab_analyzer():
    return render_template('lab_analyzer.html')

@app.route('/medical-history')
@login_required
def render_medical_history():
    patient_id = session.get('patient_id')
    if not patient_id:
        return redirect(url_for('render_login'))
        
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Initialize history data
        history_data = None
        image_analyses = []
        lab_analyses = []
        
        # Get medical history data
        history_response = get_medical_history(patient_id)
        if history_response and len(history_response) == 2:
            history_data = history_response[0].get_json() if history_response[1] == 200 else None
        
        # Get image analyses
        cursor.execute("""
            SELECT analysis_id, analysis_type, language, analysis_date,
                   image_type, anatomical_region, primary_diagnosis
            FROM image_analysis 
            WHERE patient_id = ?
            ORDER BY analysis_date DESC
        """, (patient_id,))
        
        for row in cursor.fetchall():
            image_analyses.append({
                "analysis_id": row[0],
                "analysis_type": row[1],
                "language": row[2],
                "date": row[3].strftime("%Y-%m-%d %H:%M:%S"),
                "image_type": row[4],
                "anatomical_region": row[5],
                "primary_diagnosis": row[6]
            })

        # Get lab report analyses
        cursor.execute("""
            SELECT analysis_id, analysis_type, language, analysis_date,
                   file_type, result_text
            FROM lab_report_analysis 
            WHERE patient_id = ?
            ORDER BY analysis_date DESC
        """, (patient_id,))
        
        for row in cursor.fetchall():
            lab_analyses.append({
                "analysis_id": row[0],
                "analysis_type": row[1],
                "language": row[2],
                "analysis_date": row[3].strftime("%Y-%m-%d %H:%M:%S"),
                "file_type": row[4],
                "result_text": row[5]
            })
            
        conn.close()
        return render_template('medical_history.html', 
                            medical_history=history_data,
                            image_analyses=image_analyses,
                            lab_analyses=lab_analyses)
                            
    except Exception as e:
        print(f"Error fetching medical history: {str(e)}")
        return render_template('medical_history.html', 
                            medical_history=None,
                            image_analyses=None,
                            lab_analyses=None,
                            error="Error loading analyses")

@app.route('/logout')
def logout():
    session.clear()
    flash('You have been logged out.', 'info')
    return redirect(url_for('render_login'))

# ===============================
# Database connection
# ===============================
def get_db_connection():
    conn = pyodbc.connect(
        'DRIVER={ODBC Driver 17 for SQL Server};'
        'SERVER=localhost;'
        'DATABASE=smart_Clinic;'
        'Trusted_Connection=yes;'
    )
    return conn

# ===============================
# Helper Functions
# ===============================
def validate_password(password):
    """Validate password according to database constraints"""
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    if not re.search(r'[0-9]', password):
        return False, "Password must contain at least one number"
    if not re.search(r'[A-Za-z]', password):
        return False, "Password must contain at least one letter"
    return True, ""

def get_face_encoding_from_base64(image_b64):
    """Extract face encoding from base64 image"""
    try:
        # Decode base64 image
        image_data = base64.b64decode(image_b64)
        image = Image.open(BytesIO(image_data))
        
        # Convert to numpy array (RGB)
        image_np = np.array(image)
        
        # If image is grayscale, convert to RGB
        if len(image_np.shape) == 2:
            image_np = np.stack([image_np] * 3, axis=-1)
        
        print(f"✅ Image shape: {image_np.shape}")
        
        # Find face locations
        face_locations = face_recognition.face_locations(image_np)
        print(f"✅ Found {len(face_locations)} face(s) in image")
        
        if not face_locations:
            return None
        
        # Get face encodings
        face_encodings = face_recognition.face_encodings(image_np, face_locations)
        print(f"✅ Extracted {len(face_encodings)} face encoding(s)")
        
        if not face_encodings:
            return None
        
        return face_encodings[0]  # Return first face encoding
        
    except Exception as e:
        print(f"❌ Error in face encoding: {e}")
        return None

def compare_faces(encoding1, encoding2, tolerance=0.6):
    """Compare two face encodings"""
    if encoding1 is None or encoding2 is None:
        return False
    
    # Calculate face distance
    face_distance = face_recognition.face_distance([encoding1], encoding2)[0]
    print(f"🔍 Face distance: {face_distance}, Tolerance: {tolerance}")
    
    return face_distance <= tolerance

# ===============================
# ROUTES
# ===============================

# -----------------------------------
# Route: Login with face (UPDATED)
# -----------------------------------
@app.route('/login_face', methods=['POST'])
def login_face():
    data = request.get_json()
    face_image_b64 = data.get('face_image')

    if not face_image_b64:
        return jsonify({"error": "No face image provided"}), 400

    try:
        print("🔍 Processing face login...")
        
        current_face_encoding = get_face_encoding_from_base64(face_image_b64)
        if current_face_encoding is None:
            return jsonify({
                "status": "fail", 
                "message": "No face detected in the uploaded image"
            }), 400

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT patient_id, username, first_name, last_name, face_data, 
                   LEN(face_data) as face_data_length 
            FROM patients 
            WHERE face_data IS NOT NULL
        """)
        patients = cursor.fetchall()
        conn.close()

        best_match = None
        best_distance = float('inf')
        
        for patient in patients:
            patient_id, username, first_name, last_name, stored_face_data, _ = patient
            
            if stored_face_data:
                try:
                    stored_encoding = np.frombuffer(stored_face_data, dtype=np.float64)
                    face_distance = face_recognition.face_distance([stored_encoding], current_face_encoding)[0]
                    
                    if face_distance < best_distance:
                        best_distance = face_distance
                        best_match = patient
                        
                except Exception as e:
                    continue

        tolerance = 0.6
        if best_match and best_distance <= tolerance:
            patient_id, username, first_name, last_name, _, _ = best_match
            
            # Set session
            session['logged_in'] = True
            session['patient_id'] = patient_id
            session['username'] = username
            session['name'] = f"{first_name} {last_name}"

            return jsonify({
                "status": "success",
                "message": f"Face recognition successful (distance: {best_distance:.4f})",
                "patient_id": patient_id,
                "username": username,
                "name": f"{first_name} {last_name}",
                "confidence": 1 - best_distance
            }), 200
        else:
            closest = f" (closest: {best_distance:.4f})" if best_match else ""
            return jsonify({
                "status": "fail", 
                "message": f"Face not recognized{closest}"
            }), 404

    except Exception as e:
        print(f"❌ Error in face login: {e}")
        return jsonify({"error": str(e)}), 500

# -----------------------------------
# Route: Register new patient (UPDATED)
# -----------------------------------
@app.route('/register', methods=['POST'])
def register():
    data = request.get_json()

    # Extract fields
    username = data.get('username')
    password = data.get('password')
    first_name = data.get('first_name')
    last_name = data.get('last_name')
    gender = data.get('gender')
    date_of_birth = data.get('date_of_birth')
    phone = data.get('phone')
    email = data.get('email')
    national_id = data.get('national_id')
    address = data.get('address')
    blood_type = data.get('blood_type')
    emergency_contact = data.get('emergency_contact', '')
    emergency_phone = data.get('emergency_phone', '')
    face_image_b64 = data.get('face_image')

    # Validate required fields
    required_fields = ['username', 'password', 'first_name', 'last_name', 'date_of_birth']
    for field in required_fields:
        if not data.get(field):
            return jsonify({"error": f"Missing required field: {field}"}), 400

    # Validate password
    is_valid, msg = validate_password(password)
    if not is_valid:
        return jsonify({"error": msg}), 400

    try:
        # Extract face encoding and store as binary
        face_encoding = None
        if face_image_b64:
            print("🔍 Extracting face encoding for registration...")
            face_encoding_obj = get_face_encoding_from_base64(face_image_b64)
            if face_encoding_obj is not None:
                print(f"✅ Face encoding shape: {face_encoding_obj.shape}")
                print(f"✅ Face encoding size: {face_encoding_obj.nbytes} bytes")
                face_encoding = face_encoding_obj.tobytes()
                print(f"✅ Face data binary length: {len(face_encoding)} bytes")
            else:
                return jsonify({"error": "No face detected in the provided image"}), 400

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO patients (username, password, first_name, last_name, gender, 
                                date_of_birth, phone, email, national_id, address, 
                                blood_type, emergency_contact, emergency_phone, face_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (username, password, first_name, last_name, gender, date_of_birth, 
              phone, email, national_id, address, blood_type, emergency_contact, 
              emergency_phone, face_encoding))
        conn.commit()
        conn.close()
        
        return jsonify({
            "status": "success",
            "message": "✅ Patient registered successfully!"
        }), 201

    except pyodbc.IntegrityError as e:
        if 'UNIQUE' in str(e):
            return jsonify({"error": "Username or National ID already exists"}), 400
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        print(f"❌ Registration error: {e}")
        return jsonify({"error": str(e)}), 500

# -----------------------------------
# Route: Login existing patient
# -----------------------------------
@app.route('/login', methods=['POST'])
def login():
    data = request.get_json() if request.is_json else request.form
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT patient_id, username, first_name, last_name 
            FROM patients 
            WHERE username=? AND password=?
        """, (username, password))
        user = cursor.fetchone()
        conn.close()

        if user:
            session['logged_in'] = True
            session['patient_id'] = user[0]
            session['username'] = user[1]
            session['name'] = f"{user[2]} {user[3]}"

            if request.is_json:
                return jsonify({
                    "status": "success",
                    "message": "✅ Login successful", 
                    "patient_id": user[0],
                    "username": user[1],
                    "name": f"{user[2]} {user[3]}"
                }), 200
            else:
                flash('Login successful!', 'success')
                return redirect(url_for('render_home'))
        else:
            if request.is_json:
                return jsonify({
                    "status": "fail",
                    "error": "❌ Invalid username or password"
                }), 401
            else:
                flash('Invalid username or password', 'danger')
                return redirect(url_for('render_login'))

    except Exception as e:
        if request.is_json:
            return jsonify({"error": str(e)}), 500
        else:
            flash('An error occurred during login', 'danger')
            return redirect(url_for('render_login'))

# -----------------------------------
# Route: Debug faces endpoint
# -----------------------------------
@app.route('/debug_faces', methods=['GET'])
def debug_faces():
    """Debug endpoint to check stored face data"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT patient_id, username, first_name, last_name, 
                   LEN(face_data) as data_length,
                   CASE WHEN face_data IS NULL THEN 'NULL' ELSE 'NOT NULL' END as data_status
            FROM patients
            ORDER BY patient_id
        """)
        patients = cursor.fetchall()
        conn.close()
        
        result = []
        for patient in patients:
            patient_data = {
                "patient_id": patient[0],
                "username": patient[1],
                "name": f"{patient[2]} {patient[3]}",
                "face_data_length": patient[4],
                "face_data_status": patient[5]
            }
            result.append(patient_data)
        
        return jsonify({"patients": result}), 200
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------------
# Route: Check if username exists
# -----------------------------------
@app.route('/check_username/<username>', methods=['GET'])
def check_username(username):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT username FROM patients WHERE username=?", (username,))
        user = cursor.fetchone()
        conn.close()

        if user:
            return jsonify({"exists": True}), 200
        else:
            return jsonify({"exists": False}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------------
# Route: Update face data for existing user
# -----------------------------------
@app.route('/update_face', methods=['POST'])
def update_face():
    data = request.get_json()
    username = data.get('username')
    face_image_b64 = data.get('face_image')

    if not username or not face_image_b64:
        return jsonify({"error": "Username and face image are required"}), 400

    try:
        # Extract face encoding
        face_encoding = get_face_encoding_from_base64(face_image_b64)
        if face_encoding is None:
            return jsonify({"error": "No face detected in the image"}), 400

        face_data = face_encoding.tobytes()

        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE patients 
            SET face_data = ? 
            WHERE username = ?
        """, (face_data, username))
        conn.commit()
        conn.close()

        if cursor.rowcount > 0:
            return jsonify({
                "status": "success",
                "message": "✅ Face data updated successfully!"
            }), 200
        else:
            return jsonify({"error": "User not found"}), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

# -----------------------------------
# Route: Get patient profile
# -----------------------------------
@app.route('/profile/<int:patient_id>', methods=['GET'])
def get_patient_profile(patient_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT patient_id, username, first_name, last_name, gender, 
                   date_of_birth, phone, email, national_id, address, 
                   blood_type, emergency_contact, emergency_phone, created_at
            FROM patients 
            WHERE patient_id=?
        """, (patient_id,))
        
        patient = cursor.fetchone()
        conn.close()

        if patient:
            return jsonify({
                "patient_id": patient[0],
                "username": patient[1],
                "first_name": patient[2],
                "last_name": patient[3],
                "gender": patient[4],
                "date_of_birth": str(patient[5]),
                "phone": patient[6],
                "email": patient[7],
                "national_id": patient[8],
                "address": patient[9],
                "blood_type": patient[10],
                "emergency_contact": patient[11],
                "emergency_phone": patient[12],
                "created_at": str(patient[13])
            }), 200
        else:
            return jsonify({"error": "Patient not found"}), 404

    except Exception as e:
        print(f"❌ Error getting patient profile: {e}")
        return jsonify({"error": str(e)}), 500
# -----------------------------------
# Route: Save medical history
# -----------------------------------
@app.route('/medical-history', methods=['POST'])
def save_medical_history():
    data = request.get_json()
    
    patient_id = data.get('patient_id')
    chronic_diseases = data.get('chronic_diseases', '')
    allergies = data.get('allergies', '')
    current_medications = data.get('current_medications', '')
    past_surgeries = data.get('past_surgeries', '')
    family_history = data.get('family_history', '')
    notes = data.get('notes', '')

    if not patient_id:
        return jsonify({"error": "Patient ID is required"}), 400

    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if medical history already exists
        cursor.execute("SELECT history_id FROM medical_history WHERE patient_id=?", (patient_id,))
        existing_history = cursor.fetchone()
        
        if existing_history:
            # Update existing history
            cursor.execute("""
                UPDATE medical_history 
                SET chronic_diseases=?, allergies=?, current_medications=?, 
                    past_surgeries=?, family_history=?, notes=?, last_updated=GETDATE()
                WHERE patient_id=?
            """, (chronic_diseases, allergies, current_medications, 
                  past_surgeries, family_history, notes, patient_id))
        else:
            # Insert new history
            cursor.execute("""
                INSERT INTO medical_history 
                (patient_id, chronic_diseases, allergies, current_medications, 
                 past_surgeries, family_history, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (patient_id, chronic_diseases, allergies, current_medications,
                  past_surgeries, family_history, notes))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            "status": "success",
            "message": "✅ Medical history saved successfully!"
        }), 200

    except Exception as e:
        print(f"❌ Error saving medical history: {e}")
        return jsonify({"error": str(e)}), 500

# -----------------------------------
# Route: Get medical history
# -----------------------------------
@app.route('/medical-history/<int:patient_id>', methods=['GET'])
def get_medical_history(patient_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT chronic_diseases, allergies, current_medications, 
                   past_surgeries, family_history, notes, last_updated
            FROM medical_history 
            WHERE patient_id=?
        """, (patient_id,))
        
        history = cursor.fetchone()

        # Get lab report analyses
        cursor.execute("""
            SELECT analysis_id, analysis_type, language, analysis_date,
                   file_type, result_text
            FROM lab_report_analysis 
            WHERE patient_id = ?
            ORDER BY analysis_date DESC
        """, (patient_id,))
        
        lab_analyses = []
        for row in cursor.fetchall():
            lab_analyses.append({
                "analysis_id": row[0],
                "analysis_type": row[1],
                "language": row[2],
                "date": row[3].strftime("%Y-%m-%d %H:%M:%S"),
                "file_type": row[4],
                "summary": row[5][:200] + "..." if row[5] and len(row[5]) > 200 else row[5]
            })

        conn.close()

        if history:
            response_data = {
                "chronic_diseases": history[0],
                "allergies": history[1],
                "current_medications": history[2],
                "past_surgeries": history[3],
                "family_history": history[4],
                "notes": history[5],
                "last_updated": history[6],
                "lab_analyses": lab_analyses
            }
            return jsonify(response_data), 200
        else:
            return jsonify({
                "message": "No medical history found",
                "lab_analyses": lab_analyses
            }), 404

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
# ===============================
# APPOINTMENT ROUTES
# ===============================

# -----------------------------------
# Route: Get all specialties
# -----------------------------------
@app.route('/specialties', methods=['GET'])
def get_specialties():
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT specialty_id, name FROM specialties ORDER BY name")
        specialties = cursor.fetchall()
        conn.close()

        specialties_list = []
        for specialty in specialties:
            specialties_list.append({
                "specialty_id": specialty[0],
                "name": specialty[1]
            })

        return jsonify({"specialties": specialties_list}), 200

    except Exception as e:
        print(f"❌ Error getting specialties: {e}")
        return jsonify({"error": str(e)}), 500

# -----------------------------------
# Route: Get doctors by specialty
# -----------------------------------
@app.route('/doctors/<int:specialty_id>', methods=['GET'])
def get_doctors_by_specialty(specialty_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT d.doctor_id, d.name, s.name as specialty_name
            FROM doctors d
            JOIN specialties s ON d.specialty_id = s.specialty_id
            WHERE d.specialty_id = ?
        """, (specialty_id,))
        doctors = cursor.fetchall()
        conn.close()

        doctors_list = []
        for doctor in doctors:
            doctors_list.append({
                "doctor_id": doctor[0],
                "name": doctor[1],
                "specialty": doctor[2]
            })

        return jsonify({"doctors": doctors_list}), 200

    except Exception as e:
        print(f"❌ Error getting doctors: {e}")
        return jsonify({"error": str(e)}), 500

# -----------------------------------
# Route: Get available appointments for doctor
# -----------------------------------
@app.route('/doctor_appointments/<int:doctor_id>', methods=['GET'])
def get_doctor_available_appointments(doctor_id):
    """إرجاع المواعيد المتاحة للدكتور (بدون مراعاة patient محدد)"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get doctor's schedule
        cursor.execute("""
            SELECT available_day, time_from, time_to 
            FROM schedules 
            WHERE doctor_id = ?
        """, (doctor_id,))
        schedules = cursor.fetchall()

        available_slots = []
        
        for schedule in schedules:
            day = schedule[0]
            time_from = schedule[1]
            time_to = schedule[2]
            
            # تحقق من البيانات
            if not all([day, time_from, time_to]):
                continue
                
            # تحويل الأوقات لـ datetime objects
            try:
                start_dt = datetime.datetime.combine(datetime.date.today(), time_from)
                end_dt = datetime.datetime.combine(datetime.date.today(), time_to)
            except Exception as e:
                print(f"❌ Error converting time: {e}")
                continue

            # Generate time slots كل ساعة
            current_dt = start_dt
            while current_dt < end_dt:
                current_time = current_dt.time()
                
                # Check available slots
                cursor.execute("""
                    SELECT COUNT(*) 
                    FROM appointments 
                    WHERE doctor_id = ? 
                    AND appointment_day = ? 
                    AND DATEPART(HOUR, appointment_time) = DATEPART(HOUR, ?)
                    AND status != 'Cancelled'
                """, (doctor_id, day, current_time))
                
                result = cursor.fetchone()
                appointment_count = result[0] if result else 0
                
                if appointment_count < 4:
                    available_slots.append({
                        "day": day,
                        "time": current_time.strftime('%H:%M'),
                        "available_slots": 4 - appointment_count
                    })
                
                # الانتقال للساعة التالية
                current_dt += datetime.timedelta(hours=1)

        conn.close()
        
        print(f"✅ Found {len(available_slots)} available slots for doctor {doctor_id}")
        return jsonify({"available_appointments": available_slots}), 200

    except Exception as e:
        print(f"❌ Error getting appointments: {e}")
        return jsonify({"error": str(e)}), 500

# -----------------------------------
# Route: Get available appointments with patient constraints
# -----------------------------------
@app.route('/patient_appointments/<int:doctor_id>/<int:patient_id>', methods=['GET'])
def get_patient_available_appointments(doctor_id, patient_id):
    """إرجاع المواعيد المتاحة للدكتور مع مراعاة قيود المريض المحدد"""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # ===========================================
        # التحقق الجديد: هل المريض عنده موعد مع نفس الدكتور في أي يوم؟
        # ===========================================
        cursor.execute("""
            SELECT COUNT(*) 
            FROM appointments 
            WHERE patient_id = ? 
            AND doctor_id = ? 
            AND status != 'Cancelled'
        """, (patient_id, doctor_id))
        
        existing_doctor_appointment_result = cursor.fetchone()
        existing_doctor_count = existing_doctor_appointment_result[0] if existing_doctor_appointment_result else 0

        # إذا المريض عنده موعد مع الدكتور بالفعل، ما نعرضش أي مواعيد
        if existing_doctor_count > 0:
            conn.close()
            print(f"❌ Patient {patient_id} already has an appointment with doctor {doctor_id}")
            return jsonify({"available_appointments": []}), 200
        
        # Get doctor's schedule
        cursor.execute("""
            SELECT available_day, time_from, time_to 
            FROM schedules 
            WHERE doctor_id = ?
        """, (doctor_id,))
        schedules = cursor.fetchall()

        available_slots = []
        
        for schedule in schedules:
            day = schedule[0]
            time_from = schedule[1]
            time_to = schedule[2]
            
            if time_from is None or time_to is None:
                continue
                
            try:
                start_dt = datetime.datetime.combine(datetime.date.today(), time_from)
                end_dt = datetime.datetime.combine(datetime.date.today(), time_to)
            except Exception as e:
                print(f"❌ Error converting time: {e}")
                continue

            current_dt = start_dt
            while current_dt < end_dt:
                current_time = current_dt.time()
                
                # Check doctor availability
                cursor.execute("""
                    SELECT COUNT(*) 
                    FROM appointments 
                    WHERE doctor_id = ? 
                    AND appointment_day = ? 
                    AND DATEPART(HOUR, appointment_time) = DATEPART(HOUR, ?)
                    AND status != 'Cancelled'
                """, (doctor_id, day, current_time))
                
                doctor_count_result = cursor.fetchone()
                doctor_appointment_count = doctor_count_result[0] if doctor_count_result else 0

                # Check patient conflicts
                current_time_str = current_time.strftime('%H:%M:%S')
                current_end = (current_dt + datetime.timedelta(hours=1)).time().strftime('%H:%M:%S')
                
                cursor.execute("""
                    SELECT COUNT(*) 
                    FROM appointments 
                    WHERE patient_id = ? 
                    AND appointment_day = ? 
                    AND status != 'Cancelled'
                    AND (
                        (appointment_time <= ? AND 
                         DATEADD(HOUR, 1, CAST(appointment_time AS datetime)) > CAST(? AS datetime))
                        OR
                        (appointment_time < ? AND
                         DATEADD(HOUR, 1, CAST(appointment_time AS datetime)) >= CAST(? AS datetime))
                    )
                """, (patient_id, day, 
                      current_time_str, current_time_str,
                      current_end, current_end))
                
                patient_conflict_result = cursor.fetchone()
                patient_conflict_count = patient_conflict_result[0] if patient_conflict_result else 0

                # الموعد متاح إذا:
                if (doctor_appointment_count < 4 and 
                    patient_conflict_count == 0):
                    
                    available_slots.append({
                        "day": day,
                        "time": current_time.strftime('%H:%M'),
                        "available_slots": 4 - doctor_appointment_count
                    })
                
                current_dt += datetime.timedelta(hours=1)

        conn.close()
        
        print(f"✅ Found {len(available_slots)} available slots for doctor {doctor_id} (patient {patient_id})")
        return jsonify({"available_appointments": available_slots}), 200

    except Exception as e:
        print(f"❌ Error getting appointments: {e}")
        return jsonify({"error": str(e)}), 500

# -----------------------------------
# Route: Book appointment
# -----------------------------------
@app.route('/book_appointment', methods=['POST'])
def book_appointment():
    data = request.get_json()
    
    patient_id = data.get('patient_id')
    doctor_id = data.get('doctor_id')
    appointment_day = data.get('appointment_day')
    appointment_time = data.get('appointment_time')

    print(f"🔍 Booking attempt - Patient: {patient_id}, Doctor: {doctor_id}, Day: {appointment_day}, Time: {appointment_time}")

    if not all([patient_id, doctor_id, appointment_day, appointment_time]):
        print("❌ Missing required fields")
        return jsonify({"error": "All fields are required"}), 400

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # Convert time string to time object
        appointment_time_obj = datetime.datetime.strptime(appointment_time, '%H:%M').time()
        print(f"✅ Time parsed: {appointment_time_obj}")

        # ===========================================
        # التحقق 1: منع الحجز المكرر لنفس الدكتور في نفس اليوم
        # ===========================================
        cursor.execute("""
            SELECT COUNT(*) 
            FROM appointments 
            WHERE patient_id = ? 
            AND doctor_id = ? 
            AND appointment_day = ? 
            AND status != 'Cancelled'
        """, (patient_id, doctor_id, appointment_day))
        
        same_doctor_same_day_result = cursor.fetchone()
        same_doctor_count = same_doctor_same_day_result[0] if same_doctor_same_day_result else 0

        print(f"🔍 Same doctor same day check: {same_doctor_count} appointments found")

        if same_doctor_count > 0:
            conn.close()
            return jsonify({
                "status": "fail",
                "message": "❌ You already have an appointment with this doctor on the same day. Please choose another day or doctor."
            }), 400

        # ===========================================
        # التحقق 2: منع الحجز المكرر لنفس الدكتور في نفس الوقت
        # ===========================================
        cursor.execute("""
            SELECT COUNT(*) 
            FROM appointments 
            WHERE patient_id = ? 
            AND doctor_id = ? 
            AND appointment_day = ? 
            AND appointment_time = ?
            AND status != 'Cancelled'
        """, (patient_id, doctor_id, appointment_day, appointment_time_obj))
        
        duplicate_booking_result = cursor.fetchone()
        duplicate_count = duplicate_booking_result[0] if duplicate_booking_result else 0

        print(f"🔍 Exact duplicate check: {duplicate_count} duplicates found")

        if duplicate_count > 0:
            conn.close()
            return jsonify({
                "status": "fail",
                "message": "❌ You already have an appointment with this doctor at the same time."
            }), 400

        # ===========================================
        # التحقق 3: منع المواعيد المتداخلة لنفس المريض
        # ===========================================
        appointment_datetime = datetime.datetime.combine(datetime.date.today(), appointment_time_obj)
        appointment_end = appointment_datetime + datetime.timedelta(hours=1)
        
        appointment_start_str = appointment_time_obj.strftime('%H:%M:%S')
        appointment_end_str = appointment_end.time().strftime('%H:%M:%S')

        cursor.execute("""
            SELECT COUNT(*) 
            FROM appointments 
            WHERE patient_id = ? 
            AND appointment_day = ? 
            AND status != 'Cancelled'
            AND (
                (appointment_time <= ? AND 
                 DATEADD(HOUR, 1, CAST(appointment_time AS datetime)) > CAST(? AS datetime))
                OR
                (appointment_time < ? AND
                 DATEADD(HOUR, 1, CAST(appointment_time AS datetime)) >= CAST(? AS datetime))
            )
        """, (patient_id, appointment_day, 
              appointment_start_str, appointment_start_str,
              appointment_end_str, appointment_end_str))
        
        overlapping_result = cursor.fetchone()
        overlapping_count = overlapping_result[0] if overlapping_result else 0

        print(f"🔍 Overlapping check: {overlapping_count} overlapping appointments found")

        if overlapping_count > 0:
            conn.close()
            return jsonify({
                "status": "fail",
                "message": "❌ You already have another appointment at this time. Please choose a different time."
            }), 400

        # ===========================================
        # التحقق 4: التأكد من توفر الموعد عند الدكتور
        # ===========================================
        cursor.execute("""
            SELECT COUNT(*) 
            FROM appointments 
            WHERE doctor_id = ? 
            AND appointment_day = ? 
            AND DATEPART(HOUR, appointment_time) = DATEPART(HOUR, ?)
            AND status != 'Cancelled'
        """, (doctor_id, appointment_day, appointment_time_obj))
        
        appointment_count_result = cursor.fetchone()
        appointment_count = appointment_count_result[0] if appointment_count_result else 0

        print(f"🔍 Doctor availability: {appointment_count}/4 appointments booked")

        if appointment_count >= 4:
            conn.close()
            return jsonify({
                "status": "fail",
                "message": "❌ Sorry, this time slot is fully booked. Please choose another time."
            }), 400

        # ===========================================
        # إذا كل التحققات نجحت - إجراء الحجز
        # ===========================================
        print("✅ All checks passed - proceeding with booking...")
        
        cursor.execute("""
            INSERT INTO appointments (patient_id, doctor_id, appointment_day, appointment_time, status)
            VALUES (?, ?, ?, ?, 'Booked')
        """, (patient_id, doctor_id, appointment_day, appointment_time_obj))
        
        conn.commit()

        # Get the new appointment ID
        cursor.execute("""
            SELECT appointment_id 
            FROM appointments 
            WHERE patient_id = ? 
            AND doctor_id = ? 
            AND appointment_day = ? 
            AND appointment_time = ?
            AND status = 'Booked'
            ORDER BY appointment_id DESC
        """, (patient_id, doctor_id, appointment_day, appointment_time_obj))

        appointment_id_result = cursor.fetchone()
        appointment_id = appointment_id_result[0] if appointment_id_result else None
        
        conn.close()

        print(f"✅ Booking successful! Appointment ID: {appointment_id}")

        return jsonify({
            "status": "success",
            "message": "✅ Appointment booked successfully!",
            "appointment_id": appointment_id if appointment_id else "retrieved"
        }), 201

    except ValueError as e:
        print(f"❌ Error parsing time: {e}")
        return jsonify({"error": "Invalid time format. Please use HH:MM format"}), 400
    except Exception as e:
        print(f"❌ Error booking appointment: {e}")
        return jsonify({"error": str(e)}), 500

# -----------------------------------
# Route: Get patient's appointments
# -----------------------------------
@app.route('/my_appointments/<int:patient_id>', methods=['GET'])
def get_my_appointments(patient_id):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT a.appointment_id, d.name as doctor_name, s.name as specialty_name,
                   a.appointment_day, a.appointment_time, a.status
            FROM appointments a
            JOIN doctors d ON a.doctor_id = d.doctor_id
            JOIN specialties s ON d.specialty_id = s.specialty_id
            WHERE a.patient_id = ?
            ORDER BY 
                CASE WHEN a.status = 'Booked' THEN 1 ELSE 2 END,
                a.appointment_day, 
                a.appointment_time
        """, (patient_id,))
        
        appointments = cursor.fetchall()
        conn.close()

        appointments_list = []
        for appointment in appointments:
            appointments_list.append({
                "appointment_id": appointment[0],
                "doctor_name": appointment[1],
                "specialty": appointment[2],
                "day": appointment[3],
                "time": appointment[4].strftime('%H:%M'),
                "status": appointment[5]
            })

        return jsonify({"appointments": appointments_list}), 200

    except Exception as e:
        print(f"❌ Error getting appointments: {e}")
        return jsonify({"error": str(e)}), 500
    
# -----------------------------------
# Route: Cancel appointment
# -----------------------------------
@app.route('/cancel_appointment', methods=['POST'])
def cancel_appointment():
    data = request.get_json()
    
    appointment_id = data.get('appointment_id')
    patient_id = data.get('patient_id')

    if not appointment_id or not patient_id:
        return jsonify({"error": "Appointment ID and Patient ID are required"}), 400

    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        # التحقق من أن الموعد موجود وينتمي للمريض
        cursor.execute("""
            SELECT appointment_id, status 
            FROM appointments 
            WHERE appointment_id = ? AND patient_id = ?
        """, (appointment_id, patient_id))
        
        appointment_result = cursor.fetchone()
        
        if not appointment_result:
            conn.close()
            return jsonify({
                "status": "fail",
                "message": "❌ Appointment not found or you don't have permission to cancel this appointment."
            }), 404

        current_status = appointment_result[1]
        
        # التحقق من أن الموعد ممكن يلغى (مش ملغى بالفعل)
        if current_status == 'Cancelled':
            conn.close()
            return jsonify({
                "status": "fail",
                "message": "❌ This appointment is already cancelled."
            }), 400

        cursor.execute("""
            UPDATE appointments 
            SET status = 'Cancelled' 
            WHERE appointment_id = ? AND patient_id = ?
        """, (appointment_id, patient_id))
        
        conn.commit()
        conn.close()

        return jsonify({
            "status": "success",
            "message": "✅ Appointment cancelled successfully!"
        }), 200

    except Exception as e:
        print(f"❌ Error cancelling appointment: {e}")
        return jsonify({"error": str(e)}), 500


# -----------------------------------
# Route: Analyze medical image
# -----------------------------------
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

@app.route('/analyze_image', methods=['POST'])
@login_required
def analyze_image():
    """Endpoint to analyze medical images using Gemini Vision"""
    from utils.rate_limiter import exponential_backoff, RateLimitExceededError, parse_retry_delay
    import tempfile
    import shutil
    
    print("\n=== Starting image analysis ===")
    temp_dir = None
    temp_file = None
    
    # Validate request
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400
        
    file = request.files['image']
    language = request.form.get('language')
    report_type = request.form.get('report_type')
    
    # Validate required fields
    if not all([file, language, report_type]):
        return jsonify({
            "error": "Missing required fields",
            "details": {
                "file": bool(file),
                "language": bool(language),
                "report_type": bool(report_type)
            }
        }), 400
        
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400
        
    try:
        # Create temporary directory and file
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, "temp_image.png")
        print(f"✅ Temp path created: {temp_path}")
        
        # Read and validate original image data
        file_data = file.read()
        if not file_data:
            shutil.rmtree(temp_dir)
            return jsonify({"error": "Empty file uploaded"}), 400
        print(f"✅ Read {len(file_data)} bytes of image data")
        
        # Store original image as base64
        original_image_base64 = base64.b64encode(file_data).decode('utf-8')
        
        # Reset file pointer and save temp file
        file.seek(0)
        file.save(temp_path)
        print("✅ Image saved to temp file")
        
        # Process image for analysis
        try:
            image = Image.open(temp_path)
            
            # Get original dimensions
            width, height = image.size
            print(f"📊 Original image size: {width}x{height}")
            
            # Initialize new dimensions
            new_width = width
            new_height = height
            
            # Resize while maintaining aspect ratio if needed
            max_size = 800
            if width > max_size or height > max_size:
                if width > height:
                    new_width = max_size
                    new_height = int((height/width) * max_size)
                else:
                    new_height = max_size
                    new_width = int((width/height) * max_size)
                    
                resized_image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
                resized_image.save(temp_path)
                print(f"✅ Image resized to {new_width}x{new_height}")
            else:
                print("ℹ️ Image within size limits, no resize needed")
                
        except Exception as img_error:
            print(f"❌ Error processing image: {str(img_error)}")
            return jsonify({"error": "Invalid image file or format"}), 400
        
            # Initialize and run analysis with rate limiting
        from utils.rate_limiter import exponential_backoff, RateLimitExceededError, parse_retry_delay
            
        medical_agent = get_medical_agent()
            
        @exponential_backoff(max_retries=3)
        def run_analysis(agent, image_path, analysis_query):
            """Run analysis with retry logic"""
            try:
                # Verify file exists before creating AgnoImage
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image file {image_path} does not exist")
                
                # Load and verify image
                image = Image.open(image_path)
                image.verify()  # Basic image file verification
                
                agno_image = AgnoImage(filepath=image_path)
                print("🔍 Starting Gemini analysis...")
                
                # Run analysis
                response = agent.run(analysis_query, images=[agno_image])
                if not response or not response.content:
                    raise ValueError("Empty response from Gemini API")
                    
                return response.content
                
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    # Parse retry delay from error response if available
                    retry_after = parse_retry_delay(str(e))
                    raise RateLimitExceededError(
                        retry_after=retry_after or 30, # Default 30 seconds if no retry delay found
                        message="API rate limit exceeded. Please try again in a few seconds."
                    )
                raise

        # Build prompt and run analysis
        query = build_prompt(language, report_type)
        
        try:
            # Get a fresh instance of the medical agent
            medical_agent = get_medical_agent()
            
            # Run analysis
            analysis_text = run_analysis(medical_agent, temp_path, query)
            if not analysis_text:
                raise ValueError("Analysis produced no output")
                
            print("✅ Analysis completed successfully")
            
            # Parse analysis into sections
            sections = parse_analysis_sections(analysis_text)
            print("✅ Analysis parsed into sections")
            
        except RateLimitExceededError as rate_error:
            print(f"❌ Rate limit error: {rate_error.message}")
            return jsonify({
                "error": "Service temporarily unavailable",
                "message": rate_error.message,
                "retry_after": rate_error.retry_after
            }), 429
            
        except Exception as analysis_error:
            print(f"❌ Analysis error: {str(analysis_error)}")
            return jsonify({
                "error": "Analysis failed",
                "message": "An error occurred during image analysis. Please try again."
            }), 500
            
        # Prepare response data
        try:
            # Format base64 strings with data URL prefix
            image_data_url = f"data:image/png;base64,{original_image_base64}"
            
            # Get analyzed (possibly resized) image
            with open(temp_path, 'rb') as img_file:
                analyzed_image_data = img_file.read()
            analyzed_image_base64 = base64.b64encode(analyzed_image_data).decode('utf-8')
            analyzed_image_url = f"data:image/png;base64,{analyzed_image_base64}"
            
            response_data = {
                "status": "success",
                "result": analysis_text,
                "parsed_sections": sections,
                "image_data": image_data_url,
                "analysis_image": analyzed_image_url,
                "language": language,
                "report_type": report_type,
                "metadata": {
                    "original_size": f"{width}x{height}",
                    "analyzed_size": f"{new_width}x{new_height}" if 'new_width' in locals() else f"{width}x{height}"
                }
            }
            
            print("✅ Response data prepared successfully")
            return jsonify(response_data), 200
            
        except Exception as response_error:
            print(f"❌ Error preparing response: {str(response_error)}")
            return jsonify({
                "error": "Failed to prepare analysis response",
                "details": str(response_error)
            }), 500
            
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": "An unexpected error occurred",
            "details": str(e)
        }), 500
        
    finally:
        # Clean up temp directory and all its contents
        try:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print("🧹 Cleaned up temporary directory")
        except Exception as cleanup_error:
            print(f"⚠️ Warning: Failed to clean up temp directory: {str(cleanup_error)}")

def parse_analysis_sections(analysis_text):
    """Parse the analysis text into structured sections, supporting multiple languages and formats"""
    sections = {
        "image_type": "",
        "anatomical_region": "",
        "key_findings": "",
        "primary_diagnosis": "",
        "diagnostic_assessment": "",
        "patient_explanation": "",
        "research_context": ""
    }
    
    # Initialize parsing variables
    current_section = "patient_explanation"  # Default section if no headers found
    current_text = []
    
    def extract_image_info(text):
        """Extract image type and region from text"""
        image_type = ""
        anatomical_region = ""
        
        # Common imaging terms to look for
        imaging_terms = [
            "x-ray", "mri", "ct", "ultrasound", "xray", "scan", "radiograph",
            "أشعة", "رنين", "مقطعية", "موجات", "تصوير"
        ]
        
        # Common anatomical terms
        anatomical_terms = [
            "brain", "chest", "abdomen", "spine", "knee", "shoulder", "head",
            "neck", "lung", "heart", "bone", "joint", "skull", "المخ", "الرأس",
            "الصدر", "البطن", "الركبة", "الكتف", "الرقبة", "الرئة", "القلب"
        ]
        
        for line in text.lower().split('\n'):
            # Look for image type
            if any(term in line.lower() for term in imaging_terms):
                image_type = line.strip()
                
            # Look for anatomical region
            if any(term in line.lower() for term in anatomical_terms):
                anatomical_region = line.strip()
                
            # Special case for lines that contain both
            if image_type and anatomical_region:
                break
                
        return image_type, anatomical_region
    
    # Process different report formats
    if "### 1" in analysis_text or "Image Type & Region" in analysis_text:
        # Professional report format
        section_markers = {
            "image_type": [
                "Image Type & Region", "نوع الصورة والمنطقة",
                "### 1", "###1", "#1"
            ],
            "key_findings": [
                "Key Findings", "الملاحظات الأساسية",
                "### 2", "###2", "#2"
            ],
            "diagnostic_assessment": [
                "Diagnostic Assessment", "التقييم التشخيصي",
                "### 3", "###3", "#3"
            ],
            "patient_explanation": [
                "Patient-Friendly Explanation", "تبسيط النتائج للمريض",
                "### 4", "###4", "#4"
            ],
            "research_context": [
                "Research Context", "المرجع العلمي",
                "### 5", "###5", "#5"
            ]
        }
        
        current_section = None
        current_text = []
        
        for line in analysis_text.split('\n'):
            line = line.strip()
            if not line:
                continue
                
            # Check for section markers
            found_section = None
            for section, markers in section_markers.items():
                if any(marker.lower() in line.lower() for marker in markers):
                    found_section = section
                    break
                    
            if found_section:
                # Save previous section
                if current_section and current_text:
                    sections[current_section] = '\n'.join(current_text).strip()
                current_section = found_section
                current_text = []
                continue
                
            if current_section:
                current_text.append(line)
                
        # Save last section
        # Save any remaining content from previous sections
    try:
        if 'current_section' in locals() and 'current_text' in locals() and current_section and current_text:
            sections[current_section] = '\n'.join(current_text).strip()
    except:
        pass
            
        # Extract image type and region from the image type section
        if sections["image_type"]:
            image_type, region = extract_image_info(sections["image_type"])
            if image_type:
                sections["image_type"] = image_type
            if region:
                sections["anatomical_region"] = region
    else:
        # Friendly report format - extract information from the full text
        # Split text into paragraphs
        paragraphs = [p.strip() for p in analysis_text.split('\n\n') if p.strip()]
        
        if paragraphs:
            # First paragraph usually contains image type and region
            image_type, region = extract_image_info(paragraphs[0])
            sections["image_type"] = image_type if image_type else paragraphs[0]
            sections["anatomical_region"] = region if region else ""
            
            # Key findings and assessment are usually in subsequent paragraphs
            if len(paragraphs) > 1:
                sections["key_findings"] = paragraphs[1]
            
            # Use the full text as patient explanation since it's already in friendly format
            sections["patient_explanation"] = analysis_text
            
            # Set a basic primary diagnosis if we can find one
            for paragraph in paragraphs:
                if any(word in paragraph.lower() for word in ["normal", "abnormal", "finding", "shows", "indicates"]):
                    sections["primary_diagnosis"] = paragraph
                    break
    
    # Save any remaining content in the current section
    if current_text and current_section is not None:
        sections[current_section] = '\n'.join(current_text).strip()
    
    # Post-process the sections
    for key in sections:
        if not sections[key]:
            if key == "patient_explanation" and analysis_text:
                # If no structured sections were found, use the entire text as patient explanation
                sections[key] = analysis_text.strip()
            else:
                sections[key] = "N/A"  # Default value for empty sections
        else:
            sections[key] = sections[key].strip()
    
    return sections

# -----------------------------------
# Route: Save analysis to medical history
# -----------------------------------
@app.route('/save_analysis_to_history', methods=['POST'])
@login_required
def save_analysis_to_history():
    """Save medical image analysis results to patient history"""
    print("\n=== Starting save_analysis_to_history function ===")
    
    try:
        patient_id = session.get('patient_id')
        if not patient_id:
            return jsonify({"error": "User not logged in"}), 401
            
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        # Get required fields
        analysis_result = data.get('analysis_result') or data.get('result')
        image_data = data.get('image_data', '')
        parsed_sections = data.get('parsed_sections')
        
        if not analysis_result:
            return jsonify({"error": "Analysis result is required"}), 400
        if not image_data:
            return jsonify({"error": "Image data is required"}), 400
        if not parsed_sections or not isinstance(parsed_sections, dict):
            return jsonify({"error": "Invalid parsed sections"}), 400
            
        # Process image data
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        image_binary = base64.b64decode(image_data)
        
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Insert analysis
        cursor.execute("""
            INSERT INTO image_analysis 
            (patient_id, analysis_type, language, image_data,
            image_type, anatomical_region, key_findings,
            primary_diagnosis, diagnostic_assessment,
            patient_explanation, research_context)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            patient_id,
            parsed_sections.get('analysis_type', 'General Analysis'),
            data.get('language', 'English'),
            image_binary,
            parsed_sections.get('image_type', ''),
            parsed_sections.get('anatomical_region', ''),
            parsed_sections.get('key_findings', ''),
            parsed_sections.get('primary_diagnosis', ''),
            parsed_sections.get('diagnostic_assessment', ''),
            parsed_sections.get('patient_explanation', ''),
            parsed_sections.get('research_context', '')
        ))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            "status": "success",
            "message": "✅ Analysis saved to medical history!"
        }), 200
        
    except Exception as e:
        print(f"❌ Error saving analysis: {e}")
        return jsonify({"error": str(e)}), 500

# -----------------------------------
# Route: Analyze lab report
# REMOVED DUPLICATE CHAT ROUTE

# REMOVED DUPLICATE CODE - Using the version in the main chat route

@app.route('/analyze_lab_report', methods=['POST'])
@login_required
def analyze_lab_report():
    """Analyze lab report using OCR and Gemini"""
    print("\n=== Starting lab report analysis ===")
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        file_data = data.get('file_data')
        report_type = data.get('report_type')
        language = data.get('language')
        
        if not all([file_data, report_type, language]):
            return jsonify({"error": "Missing required fields"}), 400
            
        # Extract file data
        if ',' in file_data:
            file_data = file_data.split(',')[1]
        file_binary = base64.b64decode(file_data)
        
        # Extract text based on file type
        extracted_text = ""
        if file_data.startswith('JVBERi0'):  # PDF
            extracted_text = ocr_from_pdf(file_binary)
        else:  # Image
            pil_image = Image.open(BytesIO(file_binary))
            extracted_text = ocr_from_image(pil_image)
            
        print(f"✅ Extracted {len(extracted_text)} characters of text")
        
        # Build analysis prompt
        if language == "🇬🇧 English":
            prompt = f"""Analyze this {report_type} report and provide a clear summary. Follow this structure:

1. Test Category: Identify the type of lab test(s)
2. Test Results: List key measurements and their values
3. Interpretation: Explain what the results mean
4. Normal/Abnormal: Flag any results outside normal ranges
5. Recommendations: Suggest follow-up actions if needed

Original Text:
{extracted_text}"""
        else:  # Arabic
            prompt = f"""قم بتحليل نتائج {report_type} وتقديم ملخص واضح. اتبع الهيكل التالي:

١. نوع الفحص: حدد نوع التحليل المخبري
٢. النتائج: اذكر القياسات الرئيسية وقيمها
٣. التفسير: اشرح معنى النتائج
٤. طبيعي/غير طبيعي: حدد أي نتائج خارج النطاق الطبيعي
٥. التوصيات: اقترح الخطوات التالية إن وجدت

النص الأصلي:
{extracted_text}"""

        # Get agent response
        lab_agent = get_lab_analysis_agent()
        response = lab_agent.run(prompt)
        
        if not response or not response.content:
            return jsonify({"error": "Failed to analyze report"}), 500
            
        return jsonify({
            "status": "success",
            "analysis": response.content,
            "extracted_text": extracted_text
        }), 200
        
    except Exception as e:
        print(f"❌ Error analyzing report: {str(e)}")
        return jsonify({"error": str(e)}), 500

# -----------------------------------
# Route: Save lab report analysis
# -----------------------------------
@app.route('/save_analysis', methods=['POST'])
@login_required
def save_lab_analysis():
    print("\n=== Starting save_lab_analysis function ===")
    conn = None

    try:
        patient_id = session.get('patient_id')
        if not patient_id:
            print("❌ Error: User not logged in")
            return jsonify({"error": "User not logged in"}), 401

        data = request.get_json()
        if not data:
            print("❌ Error: No data provided")
            return jsonify({"error": "No data provided"}), 400

        print("📝 Received data fields:", list(data.keys()))

        analysis_text = data.get('analysis')
        status = data.get('status', 'Pending')

        if not analysis_text:
            print("❌ Error: Analysis text is required")
            return jsonify({"error": "Analysis text is required"}), 400

        print("✅ All required fields present")

        # ✅ Default values for required columns that are NOT in request
        analysis_type = 'General'
        language = 'EN'

        conn = get_db_connection()
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO lab_report_analysis 
            (patient_id, result_text, status, analysis_type, language, analysis_date)
            VALUES (?, ?, ?, ?, ?, GETDATE());
        """, (patient_id, analysis_text, status, analysis_type, language))

        conn.commit()

        cursor.execute("SELECT @@IDENTITY")
        result = cursor.fetchone()
        analysis_id = result[0] if result else None

        conn.close()

        return jsonify({
            "status": "success",
            "message": "✅ Lab analysis saved successfully!",
            "analysis_id": analysis_id
        }), 201

    except Exception as e:
        if conn:
            try:
                conn.rollback()
                conn.close()
            except:
                pass
        print(f"❌ Error saving lab analysis: {str(e)}")
        return jsonify({"error": str(e)}), 500

def clean_analysis_text(text):
    # Remove unwanted sections (case-insensitive)
    text = re.sub(r"Analysis Type\s*Not specified", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Language\s*Not specified", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Original Report\s*No file available", "", text, flags=re.IGNORECASE)
    
    # Remove leftover blank lines
    text = re.sub(r'\n\s*\n+', '\n\n', text.strip())
    return text
# -----------------------------------
# Route: Get specific lab analysis
# -----------------------------------
@app.route('/get_lab_analysis/<int:analysis_id>')
@login_required
def get_lab_analysis(analysis_id):
    """Get a specific lab report analysis"""
    patient_id = session.get('patient_id')
    if not patient_id:
        return jsonify({"error": "User not logged in"}), 401
        
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get the analysis details
        cursor.execute("""
            SELECT analysis_id, analysis_type, language, analysis_date,
                   file_type, result_text, file_data
            FROM lab_report_analysis 
            WHERE analysis_id = ? AND patient_id = ?
        """, (analysis_id, patient_id))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return jsonify({"error": "Analysis not found"}), 404
            
        # Convert file data to base64
        file_data_base64 = base64.b64encode(row[6]).decode('utf-8') if row[6] else None
        
        # Determine content type for the data URL
        content_type = 'application/pdf' if row[4] == 'pdf' else 'image/png'
        
        return jsonify({
            "analysis_id": row[0],
            "analysis_type": row[1],
            "language": row[2],
            "date": row[3].strftime("%Y-%m-%d %H:%M:%S"),
            "file_type": row[4],
            "result_text": row[5],
            "file_data": f"data:{content_type};base64,{file_data_base64}" if file_data_base64 else None
        }), 200
        
    except Exception as e:
        print(f"Error fetching lab analysis: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
    patient_id = session.get('patient_id')
    if not patient_id:
        return jsonify({"error": "User not logged in"}), 401

    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    analysis_text = data.get('analysis')
    file_data = data.get('file_data')  # Base64 encoded file
    report_type = data.get('reportType')
    language = data.get('language', 'English')

    if not all([analysis_text, file_data, report_type]):
        return jsonify({
            "error": "Missing required fields",
            "details": {
                "analysis": bool(analysis_text),
                "file_data": bool(file_data),
                "report_type": bool(report_type)
            }
        }), 400

    try:
        # Process file data
        file_binary = None
        file_type = None
        
        if ',' in file_data:  # Data URL format
            file_data = file_data.split(',')[1]
        
        if file_data.startswith('JVBERi0'):  # PDF file
            file_type = 'pdf'
        else:  # Image file
            file_type = 'image'
            
        file_binary = base64.b64decode(file_data)
        
        # Save to database
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Insert the analysis
        cursor.execute("""
            INSERT INTO lab_report_analysis 
            (patient_id, analysis_type, language, result_text, file_data, file_type)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (patient_id, report_type, language, analysis_text, file_binary, file_type))
        
        # Get the analysis ID
        cursor.execute("SELECT @@IDENTITY")
        analysis_id = cursor.fetchone()[0]
        
        # Update medical history
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        new_entry = [
            f"\n=== Lab Report Analysis Summary ({current_time}) ===",
            f"Type: {report_type}",
            f"Language: {language}",
            "Key Findings: " + analysis_text[:200] + "..." if len(analysis_text) > 200 else analysis_text,
            f"Reference ID: {analysis_id} (Full report available in Lab Analysis section)"
        ]
        new_entry_text = '\n'.join(new_entry)
        
        # Check if medical history exists
        cursor.execute("SELECT notes FROM medical_history WHERE patient_id = ?", (patient_id,))
        existing_history = cursor.fetchone()
        
        if existing_history:
            cursor.execute("""
                UPDATE medical_history 
                SET notes = CASE 
                        WHEN notes IS NULL THEN ? 
                        ELSE notes + ?
                    END,
                    last_updated = GETDATE()
                WHERE patient_id = ?
            """, (new_entry_text, new_entry_text, patient_id))
        else:
            cursor.execute("""
                INSERT INTO medical_history 
                (patient_id, notes, last_updated)
                VALUES (?, ?, GETDATE())
            """, (patient_id, new_entry_text))
        
        conn.commit()
        conn.close()
        
        return jsonify({
            "status": "success",
            "message": "Analysis saved successfully!",
            "analysis_id": analysis_id
        }), 200
        
    except Exception as e:
        print(f"❌ Error saving lab analysis: {str(e)}")
        if 'conn' in locals():
            try:
                conn.rollback()
                conn.close()
            except:
                pass
        return jsonify({"error": str(e)}), 500
            
    def save_analysis_record(cursor, patient_id, data, image_binary):
        """Save the analysis record to database"""
        try:
            # For friendly reports, ensure we have proper content in sections
            parsed_sections = data['parsed_sections']
            if data['analysis_type'].lower() == 'friendly medical report':
                # Extract basic info from the full analysis result
                lines = data['analysis_result'].split('\n')
                image_info = lines[0] if lines else ''
                key_findings = '\n'.join(lines[1:]) if len(lines) > 1 else ''
                
                parsed_sections = {
                    'image_type': image_info,
                    'anatomical_region': image_info,  # In friendly reports, these are often combined
                    'key_findings': key_findings,
                    'primary_diagnosis': key_findings,  # Use the same content as it's simplified
                    'diagnostic_assessment': '',  # Not used in friendly reports
                    'patient_explanation': data['analysis_result'],  # Store full text
                    'research_context': ''  # Not used in friendly reports
                }
            
            # Insert with OUTPUT clause to get the ID immediately
            cursor.execute("""
                INSERT INTO image_analysis 
                (patient_id, analysis_type, language, result_text, 
                 image_data, image_type, anatomical_region, key_findings,
                 primary_diagnosis, diagnostic_assessment, patient_explanation,
                 research_context)
                OUTPUT INSERTED.analysis_id
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                patient_id, 
                data['analysis_type'],
                data['language'], 
                data['analysis_result'],
                image_binary,
                parsed_sections.get('image_type', '')[:500],  # Limit length to prevent truncation
                parsed_sections.get('anatomical_region', '')[:500],
                parsed_sections.get('key_findings', ''),
                parsed_sections.get('primary_diagnosis', ''),
                parsed_sections.get('diagnostic_assessment', ''),
                parsed_sections.get('patient_explanation', ''),
                parsed_sections.get('research_context', '')
            ))
            
            # Fetch the outputted ID
            row = cursor.fetchone()
            if not row:
                # If OUTPUT didn't work, try getting the ID using a separate query
                cursor.execute("SELECT TOP 1 analysis_id FROM image_analysis WHERE patient_id = ? ORDER BY analysis_id DESC", (patient_id,))
                row = cursor.fetchone()
                
            if not row:
                raise ValueError("Failed to get analysis ID after insert")
            
            analysis_id = row[0]
            return analysis_id
            
        except Exception as e:
            raise ValueError(f"Database error saving analysis: {str(e)}")
            
    def update_medical_history(cursor, patient_id, analysis_id, parsed_sections):
        """Update patient's medical history with analysis summary"""
        try:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Build new history entry
            new_entry = [
                f"\n=== Medical Image Analysis Summary ({current_time}) ===",
                f"Type: {parsed_sections.get('image_type', 'Medical Image')}",
                f"Region: {parsed_sections.get('anatomical_region', 'Not specified')}",
                f"Primary Diagnosis: {parsed_sections.get('primary_diagnosis', 'Not specified')}",
                f"Reference ID: {analysis_id} (Full report available in Image Analysis section)"
            ]
            new_entry_text = '\n'.join(new_entry)
            
            # Check if medical history exists
            cursor.execute("""
                SELECT notes FROM medical_history 
                WHERE patient_id = ?
            """, (patient_id,))
            
            existing_history = cursor.fetchone()
            
            if existing_history:
                # Update existing history
                cursor.execute("""
                    UPDATE medical_history 
                    SET notes = CASE 
                            WHEN notes IS NULL THEN ? 
                            ELSE notes + ?
                        END,
                        last_updated = GETDATE()
                    WHERE patient_id = ?
                """, (new_entry_text, new_entry_text, patient_id))
            else:
                # Insert new history
                cursor.execute("""
                    INSERT INTO medical_history 
                    (patient_id, notes, last_updated)
                    VALUES (?, ?, GETDATE())
                """, (patient_id, new_entry_text))
            
        except Exception as e:
            raise ValueError(f"Error updating medical history: {str(e)}")
    
    try:
        # Step 1: Validate session
        patient_id = validate_session()
        print(f"✅ Validated session for patient ID: {patient_id}")
        
        # Step 2: Get request data
        request_data = request.get_json()
        print("📦 Request data:", request_data)
        
        # Step 3: Validate and extract request data
        data = validate_request_data(request_data)
        print("✅ Validated request data")
        
        # Step 3: Process image data
        image_binary = process_image_data(data['image_data_base64'])
        print(f"✅ Processed image data ({len(image_binary)} bytes)")
        
        # Step 4: Database operations
        conn = get_db_connection()
        cursor = conn.cursor()
        
        try:
            # In pyodbc, transactions are started automatically
            # Save analysis record
            analysis_id = save_analysis_record(cursor, patient_id, data, image_binary)
            print(f"✅ Saved analysis record (ID: {analysis_id})")
            
            # Update medical history if we have diagnosis info
            if data['parsed_sections'].get('primary_diagnosis'):
                update_medical_history(cursor, patient_id, analysis_id, data['parsed_sections'])
                print("✅ Updated medical history")
                
            # Commit transaction
            conn.commit()
            print("✅ Database transaction committed")
            
            return jsonify({
                "status": "success",
                "message": "Analysis saved successfully!",
                "analysis_id": analysis_id
            }), 200
            
        except Exception as db_error:
            # Rollback on error
            if conn:
                conn.rollback()
            raise ValueError(f"Database operation failed: {str(db_error)}")
            
        finally:
            if cursor:
                cursor.close()
            
    except ValueError as e:
        print(f"❌ Validation error: {str(e)}")
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 400
        
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "status": "error",
            "message": "An unexpected error occurred",
            "details": str(e)
        }), 500
        
    finally:
        if conn:
            conn.close()
            print("Database connection closed")

# Initialize first aid vectorstore
def get_vectorstore():
    """Initialize and load FAISS vectorstore for first aid"""
    try:
        print(f"🔄 Loading first aid vectorstore from: {FIRST_AID_DB_PATH}")
        if not os.path.exists(FIRST_AID_DB_PATH):
            print(f"❌ Vectorstore path does not exist: {FIRST_AID_DB_PATH}")
            return None
            
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        print("✅ Embeddings model initialized")
        
        db = FAISS.load_local(FIRST_AID_DB_PATH, embedding_model, allow_dangerous_deserialization=True)
        print("✅ Vectorstore loaded successfully")
            
        # Verify vectorstore works
        test_docs = db.similarity_search("test", k=1)
        print("✅ Vectorstore is searchable")
            
        return db
            
    except Exception as e:
        print(f"❌ Error loading first aid vectorstore: {str(e)}")
        return None

# Custom prompt template
first_aid_prompt = """
You are a helpful, safety-focused First Aid Assistant chatbot.
Use ONLY the information provided in the context or knowledge base to answer the user's question.
If the context does not contain enough relevant information, respond politely:

"I don't have enough first aid information in my sources to answer that. Please consult a qualified healthcare professional or call your local emergency number if it's urgent."

Response Rules:
- Do NOT invent or assume medical facts
- Provide clear, step-by-step guidance only when available in sources
- Always prioritize safety and recommend emergency help for serious cases
- Use simple, reassuring language
- Stay concise and practical
- Avoid medical jargon unless explained
- Don't discuss diagnosis beyond immediate first aid

Previous discussion (summary):
{chat_summary}

Context:
{context}

Question:
{question}

Answer:
"""

# Initialize Groq LLM and chain components
GROQ_API_KEY = SecretStr("YOUR_GROQ_API_KEY")  # Replace with your actual Groq API key
def get_first_aid_chain():
    """Initialize the first aid conversational chain"""
    try:
        print("\n=== Initializing First Aid Chain ===")
        
        # Initialize LLM
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.4,
            max_tokens=512,
            api_key=GROQ_API_KEY
        )
        print("✅ LLM initialized successfully")
        
        # Initialize vectorstore and retriever
        vectorstore = get_vectorstore()
        if not vectorstore:
            raise ValueError("Failed to initialize vectorstore")
            
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 15, "lambda_mult": 0.7}
        )
        print("✅ Retriever initialized")
            
        # Initialize memory
        buffer_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="question"
        )
        
        summary_memory = ConversationSummaryBufferMemory(
            llm=llm,
            memory_key="chat_summary",
            return_messages=False,
            input_key="question",
            max_token_limit=1000
        )
        
        combined_memory = CombinedMemory(
            memories=[buffer_memory, summary_memory]
        )
        print("✅ Memory systems initialized")
        
        # Initialize prompt template
        prompt = PromptTemplate(
            template=first_aid_prompt,
            input_variables=["context", "question", "chat_summary"]
        )
        
        # Create chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=combined_memory,
            combine_docs_chain_kwargs={"prompt": prompt}
        )
        print("✅ Chain created successfully")
        
        return chain
        
    except Exception as e:
        print(f"❌ Error initializing first aid chain: {str(e)}")
        return None

# Routes
@app.route('/first-aid')
@login_required
def render_first_aid():
    """Render the First Aid chatbot page"""
    return render_template('first_aid.html')

def handle_chat_request(message, get_chain_func):
    """Generic handler for chat requests"""
    try:
        print(f"📝 Processing chat request: {message}")
        
        # Filter inappropriate content
        inappropriate_words = ["stupid", "idiot", "useless"]
        if any(word in message.lower() for word in inappropriate_words):
            return jsonify({
                "response": "I understand you're frustrated. Let's focus on how I can help you with your medical concerns."
            }), 200
            
        # Initialize chain
        chain = get_chain_func()
        if not chain:
            return jsonify({"error": "Failed to initialize chat system"}), 500
            
        # Get response
        response = chain({"question": message})
        if not response or "answer" not in response:
            return jsonify({"error": "Failed to get response"}), 500
            
        return jsonify({"response": response["answer"]}), 200
        
    except Exception as e:
        print(f"❌ Chat error: {str(e)}")
        return jsonify({"error": "An unexpected error occurred"}), 500
        
@app.route('/first_aid_chat', methods=['POST'])
@login_required
def first_aid_chat():
    """Handle First Aid chatbot interactions"""
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({"error": "No message provided"}), 400
        
    return handle_chat_request(data['message'], get_first_aid_chain)

# ===============================
# SYMPTOM CHECKER ROUTES
# ===============================

# Custom prompt template for symptom checker
symptom_checker_prompt = """
You are a friendly and knowledgeable virtual medical assistant chatbot.

Your goal is to help users understand their symptoms, offer basic first aid or home-care advice, and guide them on when to seek professional medical help.

Use the information provided in your context or knowledge base to answer questions. 
If the context does not contain enough relevant information, respond politely:

"I don't have enough verified medical information to answer that accurately. It's best to consult a qualified healthcare professional for proper evaluation."

Response Guidelines:
- Do NOT invent, assume, or speculate about medical facts or diagnoses.
- Provide clear, calm, and easy-to-follow steps or bullet points whenever possible.
- Always include a reminder that you are not a doctor and your guidance is not a substitute for professional medical advice.
- If the symptoms sound severe, urgent, or life-threatening, instruct the user to seek emergency care or call local emergency services immediately.
- Use empathetic and encouraging language — help the user feel supported, not alarmed.
- Explain medical terms in simple language.
- Keep responses concise, practical, and focused on what the user can safely do right now.

Tone & Style:
- Friendly, reassuring, and professional.
- Speak clearly, avoiding jargon unless you explain it.
- Structure responses with short paragraphs or bullet points for easy reading.

Previous discussion (summary):
{chat_summary}

Context:
{context}

Question:
{question}

Answer:
"""

# Initialize symptom checker vectorstore
def get_symptom_vectorstore():
    """Initialize and load FAISS vectorstore for symptom checker"""
    try:
        SYMPTOM_DB_PATH = "C:/Users/DELL/OneDrive/Desktop/nti/NTI-Grad-Project/backend/vectorstore_medibot/db_faiss"
        print(f"\n=== Loading symptom checker vectorstore ===")
        print(f"📂 Vectorstore path: {SYMPTOM_DB_PATH}")
        
        if not os.path.exists(SYMPTOM_DB_PATH):
            print(f"❌ Vectorstore path does not exist: {SYMPTOM_DB_PATH}")
            return None
            
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        print("✅ Embeddings model initialized")
        
        print("🔄 Loading FAISS index...")
        db = FAISS.load_local(SYMPTOM_DB_PATH, embedding_model, allow_dangerous_deserialization=True)
        print("✅ Vectorstore loaded successfully")
            
        # Verify the vectorstore works
        test_docs = db.similarity_search("test", k=1)
        print(f"✅ Vectorstore is searchable")
            
        return db
            
    except Exception as e:
        print(f"❌ Error loading symptom checker vectorstore: {str(e)}")
        return None

def get_symptom_checker_chain():
    """Initialize the symptom checker conversational chain"""
    try:
        print("\n=== Initializing Symptom Checker Chain ===")
        
        # Initialize LLM
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.4,
            max_tokens=512,
            api_key=GROQ_API_KEY
        )
        print("✅ LLM initialized successfully")
        
        # Initialize vectorstore and retriever
        vectorstore = get_symptom_vectorstore()
        if not vectorstore:
            raise ValueError("Failed to initialize vectorstore")
            
        retriever = vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 15, "lambda_mult": 0.7}
        )
        print("✅ Retriever initialized")
            
        # Initialize memory
        buffer_memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="question"
        )
        
        summary_memory = ConversationSummaryBufferMemory(
            llm=llm,
            memory_key="chat_summary",
            return_messages=False,
            input_key="question",
            max_token_limit=1000
        )
        
        combined_memory = CombinedMemory(
            memories=[buffer_memory, summary_memory]
        )
        print("✅ Memory systems initialized")
        
        # Initialize prompt template
        prompt = PromptTemplate(
            template=symptom_checker_prompt,
            input_variables=["context", "question", "chat_summary"]
        )
        
        # Create chain
        chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=combined_memory,
            combine_docs_chain_kwargs={"prompt": prompt}
        )
        print("✅ Chain created successfully")
        
        return chain
        
    except Exception as e:
        print(f"❌ Error initializing symptom checker chain: {str(e)}")
        return None

@app.route('/symptom-checker')
@login_required
def render_symptom_checker():
    """Render the symptom checker page"""
    return render_template('symptom_checker.html')

@app.route('/chat', methods=['POST'])
@login_required
def chat():
    """Handle symptom checker chatbot interactions"""
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({"error": "No message provided"}), 400
        
    return handle_chat_request(data['message'], get_symptom_checker_chain)

# ===============================
# LAB REPORT ANALYZER ROUTES
# ===============================

@app.route('/report-analyzer')
@login_required
def render_report_analyzer():
    """Render the lab report analyzer page"""
    return render_template('report_analyzer.html')

@app.route('/analyze_report', methods=['POST'])
@login_required
def analyze_report():
    """Endpoint to analyze lab reports"""
    try:
        print("\n=== Starting lab report analysis ===")
        
        # Validate request
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
            
        file = request.files['file']
        language = request.form.get('language', 'English')
        report_type = request.form.get('reportType', 'Detailed Medical Report')
        
        if not file or not file.filename:
            return jsonify({"error": "No file selected"}), 400
            
        # Process file
        try:
            file_data = file.read()
            if file.filename.lower().endswith('.pdf'):
                text = ocr_from_pdf(file_data)
            else:
                image = Image.open(io.BytesIO(file_data))
                text = ocr_from_image(image)
                
            if not text.strip():
                return jsonify({"error": "Could not extract text from the file"}), 400
                
            print(f"✅ Successfully extracted text from {file.filename}")
            
        except Exception as e:
            print(f"❌ Error processing file: {str(e)}")
            return jsonify({"error": "Error processing the file"}), 400
            
        # Build prompt based on language and report type
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
        
        # Get lab analysis agent
        lab_agent = get_lab_analysis_agent()
        print("✅ Lab analysis agent initialized")
        
        # Generate analysis
        print("🔄 Generating analysis...")
        try:
            response = lab_agent.run(prompt)
            if not response or not response.content:
                raise ValueError("Empty response from agent")
                
            print("✅ Analysis generated successfully")
            
            # Save analysis to medical history
            try:
                patient_id = session.get('patient_id')
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                
                # Build new history entry
                new_entry = [
                    f"\n=== Lab Report Analysis Summary ({current_time}) ===",
                    f"Report Type: {report_type}",
                    f"Language: {language}",
                    "Analysis Results:",
                    response.content
                ]
                new_entry_text = '\n'.join(new_entry)
                
                conn = get_db_connection()
                cursor = conn.cursor()
                
                # Check if medical history exists
                cursor.execute("""
                    SELECT notes FROM medical_history 
                    WHERE patient_id = ?
                """, (patient_id,))
                
                existing_history = cursor.fetchone()
                
                if existing_history:
                    # Update existing history
                    cursor.execute("""
                        UPDATE medical_history 
                        SET notes = CASE 
                                WHEN notes IS NULL THEN ? 
                                ELSE notes + ?
                            END,
                            last_updated = GETDATE()
                        WHERE patient_id = ?
                    """, (new_entry_text, new_entry_text, patient_id))
                else:
                    # Insert new history
                    cursor.execute("""
                        INSERT INTO medical_history 
                        (patient_id, notes, last_updated)
                        VALUES (?, ?, GETDATE())
                    """, (patient_id, new_entry_text))
                    
                conn.commit()
                conn.close()
                print("✅ Analysis saved to medical history")
                
            except Exception as db_error:
                print(f"⚠️ Warning: Could not save to medical history: {str(db_error)}")
                # Continue without failing - the analysis was still successful
            
            return jsonify({
                "status": "success",
                "analysis": response.content
            }), 200
            
        except Exception as e:
            print(f"❌ Error generating analysis: {str(e)}")
            return jsonify({
                "error": "Failed to analyze the report. Please try again."
            }), 500
            
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": "An unexpected error occurred"
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000,debug=True, use_reloader=False)
    #app.run(host='0.0.0.0', port=5000, debug=True)