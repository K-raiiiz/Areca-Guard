import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from fpdf import FPDF
import cv2
import io

# --- 1. SET PAGE CONFIG ---
st.set_page_config(
    page_title="ArecaGuard Pro",
    page_icon="static/Areca_logo.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. THEME-AWARE PROFESSIONAL STYLING ---
st.markdown("""
<style>
    .main { background-color: var(--background-color); }
    .prediction-card {
        padding: 30px; border-radius: 20px;
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.1);
        margin-bottom: 25px; text-align: center;
        color: var(--text-color);
    }
    .metric-label {
        font-size: 14px; color: var(--text-color);
        opacity: 0.8; text-transform: uppercase;
        letter-spacing: 2px; margin-bottom: 10px;
    }
    .main-diagnosis {
        font-size: 42px; font-weight: 800;
        margin: 10px 0; line-height: 1.1;
    }
    .main-accuracy {
        font-size: 26px; font-weight: 600;
        opacity: 0.9; margin-top: -5px;
    }
    .secondary-box {
        background-color: rgba(33, 150, 243, 0.15);
        padding: 20px; border-radius: 15px;
        border-left: 6px solid #2196f3;
        margin-top: 20px; color: var(--text-color);
        text-align: left;
    }
    .stFileUploader label { color: var(--text-color); }
</style>
""", unsafe_allow_html=True)

# --- INJECT MANIFEST LINK FOR PWA LOGO SUPPORT ---
# --- INJECT MANIFEST + APPLE TOUCH ICON FOR HOME SCREEN LOGO ---
st.markdown("""
    <link rel="manifest" href="/App/static/manifest.json">
    <link rel="apple-touch-icon" href="/App/static/Areca_logo.png">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="apple-mobile-web-app-status-bar-style" content="default">
    <meta name="apple-mobile-web-app-title" content="ArecaGuard">
    <meta name="theme-color" content="#2e7d32">
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
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
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
    pdf.multi_cell(0, 8, txt="Disclaimer: This report is AI-generated for screening. Consult an agri-professional.")
    return pdf.output(dest='S').encode('latin-1')

# --- 4. CONSTANTS ---
CLASS_NAMES = [
    'Healthy Leaf', 'Healthy Nut', 'Healthy Trunk',
    'Mahali Koleroga', 'Stem Bleeding', 'Bud Borer',
    'Healthy Foot', 'Stem Cracking', 'Yellow Leaf Disease'
]

TREATMENTS = {
    'Mahali Koleroga':    ["Improve drainage.", "Spray 1% Bordeaux before monsoon.", "Burn fallen rotten nuts."],
    'Stem Bleeding':      ["Chisel out infected bark.", "Apply coal tar or Bordeaux paste.", "Check for irrigation stress."],
    'Bud Borer':          ["Remove and burn infested parts.", "Apply Neem cake to the basin.", "Set up light traps for moths."],
    'Stem Cracking':      ["Avoid soil moisture fluctuations.", "Apply lime wash to trunk.", "Ensure adequate Potash (MOP)."],
    'Yellow Leaf Disease':["Apply Magnesium Sulphate.", "Use balanced NPK fertilizers.", "Remove severely infected palms."]
}

# --- 5. MODEL LOADING ---
try:
    interpreter = load_tflite_model()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    st.error(f"⚠️ Model Load Error: {e}. Ensure 'areca_model_optimized.tflite' is in your GitHub folder.")
    st.stop()

# --- 6. MAIN USER INTERFACE ---
st.title("🌴 Arecanut Health Diagnostics")
st.write("AI-powered identification using optimized Edge-AI (TFLite).")

col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
    st.subheader("📸 Upload Sample")
    uploaded_file = st.file_uploader("Upload leaf, nut, or trunk photo...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        raw_image = Image.open(uploaded_file).convert('RGB')
        cleaned_image = clean_image(raw_image)
        st.image(cleaned_image, caption="Optimized View", use_container_width=True)

        # Preprocessing for TFLite (Float32 required)
        img_resized = cleaned_image.resize((224, 224))
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    if uploaded_file:
        with st.spinner('🧬 Analyzing pathogenic patterns...'):

            # TFLite Inference Step
            interpreter.set_tensor(input_details[0]['index'], img_array)
            interpreter.invoke()
            preds = interpreter.get_tensor(output_details[0]['index'])[0]

        sorted_indices = np.argsort(preds)[::-1]
        top_idx = sorted_indices[0]
        top_result = CLASS_NAMES[top_idx]
        top_conf = preds[top_idx] * 100

        if top_conf < 40.0:
            st.error("### ❌ Low Confidence")
            st.write("Image unclear. Please provide a sharper photo.")
            st.stop()

        # --- RESULTS UI ---
        is_healthy = "Healthy" in top_result
        status_color = "#2e7d32" if is_healthy else "#d32f2f"

        st.markdown(f"""
        <div class="prediction-card" style="border-top: 8px solid {status_color};">
            <p class="metric-label">Status Detected</p>
            <h1 class="main-diagnosis" style="color: {status_color};">{top_result.upper()}</h1>
            <p class="main-accuracy">{top_conf:.1f}% Confidence</p>
        </div>
        """, unsafe_allow_html=True)

        if len(sorted_indices) > 1:
            sec_idx = sorted_indices[1]
            sec_result = CLASS_NAMES[sec_idx]
            sec_conf = preds[sec_idx] * 100
            if (top_conf - sec_conf) < 25.0 and not is_healthy:
                st.markdown(f"""
                <div class="secondary-box">
                    <strong>🔍 Note:</strong> Also similar to <b>{sec_result}</b> ({sec_conf:.1f}%).
                </div>
                """, unsafe_allow_html=True)

        # --- PROTOCOLS & PDF EXPORT ---
        current_steps = TREATMENTS.get(top_result, [])

        if not is_healthy:
            st.subheader("🛡️ Management Protocol")
            for step in (current_steps or ["Consult local Agri-department for specific treatment."]):
                st.write(f"✅ {step}")
            pdf_data = generate_pdf(top_result, top_conf, current_steps)
            st.download_button(
                "📥 Download Diagnosis (PDF)",
                data=pdf_data,
                file_name=f"Report_{top_result.replace(' ', '_')}.pdf",
                mime="application/pdf"
            )
        else:
            st.balloons()
            st.success(f"✨ This sample appears to be healthy.")
            pdf_data = generate_pdf(top_result, top_conf, [])
            st.download_button(
                "📥 Download Health Cert (PDF)",
                data=pdf_data,
                file_name="Healthy_Report.pdf",
                mime="application/pdf"
            )
    else:
        st.info("📤 Upload a sample image to begin analysis.")