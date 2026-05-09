import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from fpdf import FPDF
import cv2
import io

# Load logo image
logo = Image.open("Assets/Areca_logo.png")

# --- 1. SET PAGE CONFIG ---
st.set_page_config(
    page_title="ArecaGuard Pro",
    page_icon=logo,
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 1A. LINK CUSTOM WEB MANIFEST FOR PWA ICON ---
st.markdown(
    """
    <link rel="manifest" href="static/manifest.json">
    """,
    unsafe_allow_html=True
)

# --- 2. THEME-AWARE PROFESSIONAL STYLING ---
st.markdown("""
    <style>
    .main { background-color: var(--background-color); }
    .prediction-card { 
        padding: 30px; 
        border-radius: 20px; 
        background-color: var(--secondary-background-color); 
        border: 1px solid rgba(128, 128, 128, 0.2);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.1); 
        margin-bottom: 25px;
        text-align: center;
        color: var(--text-color);
    }
    .metric-label { 
        font-size: 14px; 
        color: var(--text-color); 
        opacity: 0.8;
        text-transform: uppercase; 
        letter-spacing: 2px; 
        margin-bottom: 10px; 
    }
    .main-diagnosis { 
        font-size: 42px; 
        font-weight: 800; 
        margin: 10px 0; 
        line-height: 1.1; 
    }
    .main-accuracy { 
        font-size: 26px; 
        font-weight: 600; 
        opacity: 0.9;
        margin-top: -5px; 
    }
    .secondary-box { 
        background-color: rgba(33, 150, 243, 0.15); 
        padding: 20px; 
        border-radius: 15px; 
        border-left: 6px solid #2196f3; 
        margin-top: 20px; 
        color: var(--text-color); 
        text-align: left;
    }
    .stFileUploader label { color: var(--text-color); }
    </style>
    """, unsafe_allow_html=True)

# --- 3. CORE PROCESSING FUNCTIONS ---

@st.cache_resource
def load_tflite_model():
    """Loads the TFLite interpreter for the optimized model."""
    interpreter = tf.lite.Interpreter(model_path="areca_model_optimized.tflite")
    interpreter.allocate_tensors()
    return interpreter

def clean_image(image):
    img_cv = np.array(image)
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(img_cv, -1, kernel)
    return Image.fromarray(cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB))

def generate_pdf(result, confidence, steps):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.set_text_color(46, 125, 50)
    pdf.cell(200, 10, txt="ArecaGuard AI - Diagnostic Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, txt=f"Diagnosis: {result}", ln=True)
    pdf.cell(200, 10, txt=f"Confidence Score: {confidence:.1f}%", ln=True)
    pdf.ln(5)

    if "Healthy" not in result:
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(200, 10, txt="Recommended Management Protocol:", ln=True)
        pdf.set_font("Arial", '', 11)
        for step in steps:
            pdf.multi_cell(0, 10, txt=f"- {step}")

    pdf.ln(20)
    pdf.set_font("Arial", 'I', 10)
    pdf.set_text_color(128, 128, 128)
    pdf.multi_cell(
        0,
        8,
        txt="Disclaimer: This report is AI-generated for screening. Consult an agri-professional."
    )
    return pdf.output(dest='S').encode('latin-1')

# --- 4. CONSTANTS ---
CLASS_NAMES = [
    'Healthy Leaf', 'Healthy Nut', 'Healthy Trunk',
    'Mahali Koleroga', 'Stem Bleeding', 'Bud Borer',
    'Healthy Foot', 'Stem Cracking', 'Yellow Leaf Disease'
]

TREATMENTS = {
    'Mahali Koleroga': [
        "Improve drainage.",
        "Spray 1% Bordeaux before monsoon.",
        "Burn fallen rotten nuts."
    ],
    'Stem Bleeding': [
        "Chisel out infected bark.",
        "Apply coal tar or Bordeaux paste.",
        "Check for irrigation stress."
    ],
    'Bud Borer': [
        "Remove and burn infested parts.",
        "Apply Neem cake to the basin.",
        "Set up light traps for moths."
    ],
    'Stem Cracking': [
        "Avoid soil moisture fluctuations.",
        "Apply lime wash to trunk.",
        "Ensure adequate Potash (MOP)."
    ],
    'Yellow Leaf Disease': [
        "Apply Magnesium Sulphate.",
        "Use balanced NPK fertilizers.",
        "Remove severely infected palms."
    ]
}

# --- 5. MODEL LOADING ---
try:
    interpreter = load_tflite_model()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    st.error(
        f"⚠️ Model Load Error: {e}. Ensure 'areca_model_optimized.tflite' is in your GitHub folder."
    )
    st.stop()

# --- 6. MAIN USER INTERFACE ---
st.title("🌴 Arecanut Health Diagnostics")
st.write("AI-powered identification using optimized Edge-AI.")

# Keep the rest of your original code unchanged below this point.