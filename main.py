import subprocess
import sys
import pkg_resources
import logging

# -------------------------
# Logger setup
# -------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------
# List of required packages
# -------------------------
subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "setuptools", "pip"])

required_packages = {
    "streamlit": "1.51.0",
    "pandas": "2.0.3",
    "openpyxl": "3.0.10",
    "reportlab": "4.4.4",
    "numpy": "1.26.4",
    "torch": "2.2.0+cpu",
    "transformers": "5.1.1",
    "sentence-transformers": "5.1.1",
    "langdetect": "1.0.9"
}

# -------------------------
# Function to install packages
# -------------------------
def install_packages(packages: dict):
    for package, version in packages.items():
        try:
            dist = pkg_resources.get_distribution(package)
            if dist.version == version.replace("+cpu", ""):
                logger.info(f"{package}=={dist.version} already installed.")
                continue
        except pkg_resources.DistributionNotFound:
            logger.info(f"{package} not found. Installing {package}=={version} ...")
        try:
            if package == "torch" and "+cpu" in version:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install",
                    f"torch=={version}", "--index-url", "https://download.pytorch.org/whl/cpu"
                ])
            else:
                subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package}=={version}"])
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install {package}: {e}")

# -------------------------
# Install missing packages
# -------------------------
install_packages(required_packages)


# -------------------------


"""
QLingo - AI-Powered Translation QA Platform
Main Streamlit Application - FIXED VERSION
"""

import streamlit as st
import pandas as pd
from pathlib import Path
import sys
from datetime import datetime
import io

# Add utils to path
sys.path.append(str(Path(__file__).parent))
from auth import check_authentication, show_login_page
from utils.qa_checks import TranslationQAEngine
from utils.file_loader import FileLoader

# Page configuration
st.set_page_config(
    page_title="QLingo - Translation QA Platform",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #2E86AB 0%, #A23B72 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #2E86AB;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    .error-card {
        background: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #F77F00;
        margin: 0.5rem 0;
    }
    .success-card {
        background: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #06D6A0;
        margin: 0.5rem 0;
    }
    .info-box {
        background: #e7f3ff;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'qa_engine' not in st.session_state:
        st.session_state.qa_engine = None
    if 'results_df' not in st.session_state:
        st.session_state.results_df = None
    if 'summary_stats' not in st.session_state:
        st.session_state.summary_stats = None

def show_header():
    st.markdown("""
    <div class="main-header">
        <h1>🌐 QLingo</h1>
        <p>AI-Powered Translation Quality Assurance Platform</p>
    </div>
    """, unsafe_allow_html=True)

def get_quality_badge(score):
    if score >= 90:
        badge_class = "quality-excellent"
        label = "Excellent"
    elif score >= 75:
        badge_class = "quality-good"
        label = "Good"
    elif score >= 60:
        badge_class = "quality-fair"
        label = "Fair"
    else:
        badge_class = "quality-poor"
        label = "Poor"
    return f'<span style="padding:0.5rem 1rem;border-radius:20px;font-weight:600;background:#d4edda;color:#155724">{score:.1f}% - {label}</span>'

def format_error_display(errors):
    if not errors:
        return '<div class="success-card">✅ <strong>No errors found!</strong> Translation quality is excellent.</div>'
    
    severity_icons = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
    html = '<div class="error-card">'
    html += f'<strong>⚠️ {len(errors)} Error(s) Found:</strong><br><br>'
    for error in errors:
        icon = severity_icons.get(error['severity'], '⚪')
        html += f"{icon} <strong>{error['type']}</strong> ({error['severity']})<br>"
        html += f"&nbsp;&nbsp;&nbsp;{error['description']}<br><br>"
    html += '</div>'
    return html

def text_analysis_mode():
    st.markdown("### 📝 Single Segment Analysis")
    st.markdown('<div class="info-box">Analyze individual translation segments with AI-powered quality checks.</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        source_text = st.text_area("Source Text", placeholder="Enter the source text here...", height=150)
        translation_text = st.text_area("Translation", placeholder="Enter the translation here...", height=150)
    
    with col2:
        glossary_text = st.text_area("Glossary (Optional)", placeholder="Format: source_term:target_term (one per line)", height=100)
        similarity_threshold = st.slider("Semantic Similarity Threshold", 0.3, 1.0, 0.6, 0.05,key="similarity_threshold_text")
       
    
    if st.button("🔍 Analyze Translation"):
        if not source_text or not translation_text:
            st.error("⚠️ Please provide both source text and translation!")
            return
        
        with st.spinner("🔄 Analyzing translation quality..."):
            if st.session_state.qa_engine is None:
                st.session_state.qa_engine = TranslationQAEngine()
            
            glossary = {}
            if glossary_text:
                for line in glossary_text.strip().split('\n'):
                    if ':' in line:
                        src, tgt = line.split(':', 1)
                        glossary[src.strip()] = tgt.strip()
            
            st.session_state.qa_engine.load_glossary(glossary)
            result = st.session_state.qa_engine.analyze_segment(source_text, translation_text, similarity_threshold)
        
        st.markdown("---")
        st.markdown("### 📊 Analysis Results")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown("**Quality Score**")
            st.markdown(get_quality_badge(result['quality_score']), unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Semantic Similarity", f"{result['semantic_similarity']:.1%}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Errors Found", result['error_count'])
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("### 🔍 Error Details")
        st.markdown(format_error_display(result['errors']), unsafe_allow_html=True)

def file_analysis_mode():
    st.markdown("### 📁 Batch File Analysis")
    st.markdown('<div class="info-box">Upload translation files for comprehensive batch quality analysis.</div>', unsafe_allow_html=True)
    
    
    
    uploaded_file = st.file_uploader("Upload Translation File", type=['csv', 'xlsx', 'xls', 'json', 'xliff'] )
    
    if uploaded_file:
        col1, col2 = st.columns(2)
        
        with col1:
            source_col = st.text_input("Source Column Name", value="source")
            translation_col = st.text_input("Translation Column Name", value="translation")
        
        with col2:
            glossary_text = st.text_area("Glossary (Optional)", placeholder="source_term:target_term (one per line)", height=100)
            similarity_threshold = st.slider("Semantic Similarity Threshold", 0.3, 1.0, 0.6, 0.05)
        
        if st.button("🔍 Process File"):
            process_file(uploaded_file, source_col, translation_col, glossary_text, similarity_threshold)

def process_file(uploaded_file, source_col, translation_col, glossary_text, similarity_threshold):
    with st.spinner("🔄 Processing file..."):
        try:
            if st.session_state.qa_engine is None:
                st.session_state.qa_engine = TranslationQAEngine()
            
            glossary = {}
            if glossary_text:
                for line in glossary_text.strip().split('\n'):
                    if ':' in line:
                        src, tgt = line.split(':', 1)
                        glossary[src.strip()] = tgt.strip()
            
            st.session_state.qa_engine.load_glossary(glossary)
            
            file_loader = FileLoader()
            df = file_loader.load_file(uploaded_file)
            source_col = 'Source Text (English)'
            translation_col = 'Translated Text (Hindi)'
            
            if df is None:
                st.error(" Failed to load file. Please check the file format.")
                return
            
            if source_col not in df.columns or translation_col not in df.columns:
                st.error(f" Columns '{source_col}' or '{translation_col}' not found!")
                st.info(f"Available columns: {', '.join(df.columns)}")
                return
            
            results = []
            progress_bar = st.progress(0)
            
            for idx, row in df.iterrows():
                progress_bar.progress((idx + 1) / len(df))
                source = str(row[source_col])
                translation = str(row[translation_col])
                analysis = st.session_state.qa_engine.analyze_segment(source, translation, similarity_threshold)
                
                results.append({
                    'Segment ID': idx + 1,
                    'Source': source,
                    'Translation': translation,
                    'Quality Score': round(analysis['quality_score'], 1),
                    'Semantic Similarity': f"{analysis['semantic_similarity']:.1%}",
                    'Error Count': analysis['error_count'],
                    'Errors': '; '.join([f"{e['type']}" for e in analysis['errors']])
                })
            
            progress_bar.empty()
            st.session_state.results_df = pd.DataFrame(results)
            
            avg_quality = st.session_state.results_df['Quality Score'].mean()
            total_errors = st.session_state.results_df['Error Count'].sum()
            segments_with_errors = (st.session_state.results_df['Error Count'] > 0).sum()
            
            st.session_state.summary_stats = {
                'total_segments': len(st.session_state.results_df),
                'avg_quality': avg_quality,
                'total_errors': total_errors,
                'segments_with_errors': segments_with_errors
            }
            
            st.success(" File processed successfully!")
            
        except Exception as e:
            st.error(f" Error processing file: {str(e)}")
            return
    
    display_batch_results()

def display_batch_results():
    if st.session_state.results_df is None:
        return

    st.markdown("---")
    st.markdown("### 📊 Analysis Results")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Segments", st.session_state.summary_stats['total_segments'])
    col2.metric("Avg Quality Score", f"{st.session_state.summary_stats['avg_quality']:.1f}%")
    col3.metric("Segments with Errors", st.session_state.summary_stats['segments_with_errors'])
    col4.metric("Total Errors", st.session_state.summary_stats['total_errors'])

    st.markdown("### 📋 Detailed Results")
    st.dataframe(st.session_state.results_df, use_container_width=True, height=400)

    st.markdown("### 📥 Export Results")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # --- Excel ---
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        summary_df = pd.DataFrame([
            ['Report Generated', timestamp],
            ['Total Segments', st.session_state.summary_stats['total_segments']],
            ['Average Quality Score', f"{st.session_state.summary_stats['avg_quality']:.1f}%"],
            ['Segments with Errors', st.session_state.summary_stats['segments_with_errors']],
            ['Total Errors', st.session_state.summary_stats['total_errors']]
        ], columns=['Metric', 'Value'])
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        st.session_state.results_df.to_excel(writer, sheet_name='QA Results', index=False)
    st.download_button(
        label="📥 Download Excel Report",
        data=excel_buffer.getvalue(),
        file_name=f"qlingo_report_{timestamp}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_excel"
    )

    # --- CSV ---
    csv_data = st.session_state.results_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download CSV Report",
        data=csv_data,
        file_name=f"qlingo_report_{timestamp}.csv",
        mime="text/csv",
        key="download_csv"
    )

    # --- PDF ---
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
        from reportlab.lib import colors

        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(pdf_buffer, pagesize=A4)
        elements = []

        data = [["Metric", "Value"],
                ["Total Segments", str(st.session_state.summary_stats['total_segments'])],
                ["Avg Quality", f"{st.session_state.summary_stats['avg_quality']:.1f}%"],
                ["Segments with Errors", str(st.session_state.summary_stats['segments_with_errors'])],
                ["Total Errors", str(st.session_state.summary_stats['total_errors'])]]

        table = Table(data)
        table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 1, colors.black),
                                   ('BACKGROUND',(0,0),(1,0),colors.lightgrey)]))
        elements.append(table)
        doc.build(elements)

        st.download_button(
            label="📥 Download PDF Report",
            data=pdf_buffer.getvalue(),
            file_name=f"qlingo_report_{timestamp}.pdf",
            mime="application/pdf",
            key="download_pdf"
        )
    except ImportError:
        st.error("Install reportlab: pip install reportlab")


def show_sidebar():
    with st.sidebar:
        st.markdown("### 👤 User Profile")
        st.markdown(f"**Logged in as:** {st.session_state.username}")
        
        if st.button("🚪 Logout"):
            st.session_state.authenticated = False
            st.session_state.username = None
            st.rerun()
        
        st.markdown("---")
        st.markdown("### ℹ️ About QLingo")
        st.markdown("""
        ✓ **Semantic Analysis** - AI-powered meaning verification
        ✓ **Rule-Based Checks** - Numbers, dates, symbols validation
        ✓ **Glossary Management** - Terminology compliance
        ✓ **Batch Processing** - Handle large translation files
        ✓ **Detailed Reports** - Export comprehensive QA reports
        """)

def main():
    initialize_session_state()
    
    if not st.session_state.authenticated:
        show_login_page()
        return
    
    show_header()
    show_sidebar()
    
    tab1, tab2 = st.tabs(["📝 Text Analysis", "📁 File Analysis"])
    
    with tab1:
        text_analysis_mode()
    
    with tab2:
        file_analysis_mode()

if __name__ == "__main__":
    main()
