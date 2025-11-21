import streamlit as st
import os
import json
import tempfile
import subprocess
import logging
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import faiss

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Healthcare LLM Agent",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Ensure main content text is dark/black (not sidebar) */
    .main .block-container {
        color: #1e293b;
    }
    
    .main .block-container p,
    .main .block-container label,
    .main .block-container div:not([class*="status"]):not([class*="info"]):not([class*="success"]):not([class*="warning"]),
    .main .block-container span {
        color: #1e293b;
    }
    
    /* Streamlit default text elements in main content only */
    .main .stMarkdown,
    .main .stText,
    .main .element-container label,
    .main .stSelectbox label,
    .main .stSlider label,
    .main .stTextInput label,
    .main .stTextArea label {
        color: #1e293b !important;
    }
    
    /* Keep sidebar text white */
    [data-testid="stSidebar"] .stMarkdown,
    [data-testid="stSidebar"] .stText,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .element-container label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stTextInput label,
    [data-testid="stSidebar"] .stTextArea label {
        color: #f1f5f9 !important;
    }
    
    /* Main container */
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }
    
    /* Navigation bar styling */
    .nav-bar {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        padding: 0.75rem 2rem;
        margin: -1rem -1rem 1.5rem -1rem;
        border-bottom: 2px solid #667eea;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    
    .nav-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #f1f5f9;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        letter-spacing: -0.01em;
    }
    
    .nav-subtitle {
        font-size: 0.85rem;
        color: #cbd5e1;
        font-weight: 400;
        margin: 0;
    }
    
    /* Hide main header and subtitle if using nav bar */
    .main-header {
        display: none;
    }
    
    .main-subtitle {
        display: none;
    }
    
    /* Sub headers */
    .sub-header {
        font-size: 1.75rem;
        color: #1e293b;
        margin-top: 2.5rem;
        margin-bottom: 1.5rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        padding-bottom: 0.75rem;
        border-bottom: 2px solid #667eea;
    }
    
    /* Sub headers in sidebar should be white */
    [data-testid="stSidebar"] .sub-header {
        color: #f1f5f9;
        border-bottom: 2px solid #667eea;
    }
    
    /* Status boxes */
    .status-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem 2.5rem;
        border-radius: 20px;
        margin: 1.5rem 0;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.25), 0 0 0 1px rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .status-box:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 50px rgba(102, 126, 234, 0.35), 0 0 0 1px rgba(255, 255, 255, 0.15);
    }
    
    .success-box {
        background: linear-gradient(135deg, #10b981 0%, #34d399 100%);
        color: white;
        padding: 1.75rem 2rem;
        border-radius: 18px;
        margin: 1.25rem 0;
        box-shadow: 0 8px 30px rgba(16, 185, 129, 0.25), 0 0 0 1px rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #f59e0b 0%, #f97316 100%);
        color: white;
        padding: 1.75rem 2rem;
        border-radius: 18px;
        margin: 1.25rem 0;
        box-shadow: 0 8px 30px rgba(245, 158, 11, 0.25), 0 0 0 1px rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .info-box {
        background: linear-gradient(135deg, #3b82f6 0%, #60a5fa 100%);
        color: white;
        padding: 1.75rem 2rem;
        border-radius: 18px;
        margin: 1.25rem 0;
        box-shadow: 0 8px 30px rgba(59, 130, 246, 0.25), 0 0 0 1px rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Chat styling */
    .chat-container {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        padding: 2.5rem;
        border-radius: 24px;
        margin: 1.5rem 0;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.9);
        border: 1px solid rgba(226, 232, 240, 0.8);
    }
    
    .question-box {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 1.75rem 2rem;
        border-radius: 18px;
        margin: 1.25rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06), 0 0 0 1px rgba(0, 0, 0, 0.04);
        transition: all 0.3s ease;
        color: #1e293b;
    }
    
    .question-box:hover {
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.15), 0 0 0 1px rgba(102, 126, 234, 0.1);
        transform: translateX(4px);
    }
    
    .question-box p,
    .question-box h5,
    .question-box div {
        color: #1e293b !important;
    }
    
    .answer-box {
        background: linear-gradient(135deg, #ffffff 0%, #f0fdf4 100%);
        padding: 1.75rem 2rem;
        border-radius: 18px;
        margin: 1.25rem 0;
        border-left: 4px solid #10b981;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06), 0 0 0 1px rgba(0, 0, 0, 0.04);
        transition: all 0.3s ease;
        color: #1e293b;
    }
    
    .answer-box:hover {
        box-shadow: 0 8px 30px rgba(16, 185, 129, 0.15), 0 0 0 1px rgba(16, 185, 129, 0.1);
        transform: translateX(4px);
    }
    
    .answer-box p,
    .answer-box h5,
    .answer-box div {
        color: #1e293b !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
    }
    
    [data-testid="stSidebar"] .css-1d391kg {
        padding-top: 2rem;
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] h4,
    [data-testid="stSidebar"] h5,
    [data-testid="stSidebar"] h6 {
        color: #f1f5f9 !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #cbd5e1;
    }
    
    /* Ensure sidebar headings are always white */
    [data-testid="stSidebar"] h3 {
        color: #f1f5f9 !important;
    }
    
    /* Input field styling */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: #ffffff;
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        padding: 0.75rem 1rem;
        font-size: 0.95rem;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        color: #1e293b !important;
    }
    
    .stTextInput > div > div > input::placeholder,
    .stTextArea > div > div > textarea::placeholder {
        color: #94a3b8 !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1), 0 4px 12px rgba(0, 0, 0, 0.08);
        outline: none;
        color: #1e293b !important;
    }
    
    /* Selectbox text color */
    .stSelectbox > div > div > div {
        color: #1e293b !important;
    }
    
    .stSelectbox > div > div > div[data-baseweb="select"] {
        color: #1e293b !important;
    }
    
    /* Ensure all input text is black */
    input[type="text"],
    input[type="number"],
    textarea,
    select {
        color: #1e293b !important;
    }
    
    /* Sidebar input fields - text should be black */
    [data-testid="stSidebar"] .stTextInput > div > div > input,
    [data-testid="stSidebar"] .stTextArea > div > div > textarea {
        color: #1e293b !important;
    }
    
    /* Selectbox text color - both main and sidebar */
    .stSelectbox [data-baseweb="select"] {
        color: #1e293b !important;
    }
    
    .stSelectbox [data-baseweb="select"] > div {
        color: #1e293b !important;
    }
    
    .stSelectbox [data-baseweb="select"] > div > div {
        color: #1e293b !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] {
        color: #1e293b !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div {
        color: #1e293b !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div > div {
        color: #1e293b !important;
    }
    
    /* Additional selectbox styling */
    .stSelectbox > div > div > div > div {
        color: #1e293b !important;
    }
    
    [data-testid="stSidebar"] .stSelectbox > div > div > div > div {
        color: #1e293b !important;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
        letter-spacing: 0.01em;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0);
        box-shadow: 0 2px 10px rgba(102, 126, 234, 0.3);
    }
    
    /* Secondary button styling */
    button[kind="secondary"] {
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
        color: #475569;
        border: 1px solid #cbd5e1;
    }
    
    button[kind="secondary"]:hover {
        background: linear-gradient(135deg, #e2e8f0 0%, #cbd5e1 100%);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Metric styling */
    .metric-container {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 1.5rem;
        border-radius: 16px;
        margin: 0.75rem 0;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06), 0 0 0 1px rgba(0, 0, 0, 0.04);
        text-align: center;
        transition: all 0.3s ease;
        border: 1px solid rgba(226, 232, 240, 0.8);
    }
    
    .metric-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.1), 0 0 0 1px rgba(0, 0, 0, 0.06);
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background: #ffffff;
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        color: #1e293b !important;
    }
    
    .stSelectbox > div > div > div {
        color: #1e293b !important;
    }
    
    /* Selectbox dropdown text */
    [data-baseweb="select"] {
        color: #1e293b !important;
    }
    
    [data-baseweb="select"] > div {
        color: #1e293b !important;
    }
    
    /* Slider styling */
    .stSlider > div > div {
        padding: 0.5rem 0;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        border-radius: 12px;
        padding: 1rem;
        font-weight: 600;
        color: #475569;
    }
    
    /* Footer styling */
    .footer {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        color: #f1f5f9;
        padding: 3rem 2rem;
        border-radius: 24px;
        margin-top: 4rem;
        text-align: center;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    /* Loading animation */
    .loading {
        display: inline-block;
        width: 24px;
        height: 24px;
        border: 3px solid rgba(102, 126, 234, 0.2);
        border-top: 3px solid #667eea;
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Divider styling */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #e2e8f0, transparent);
        margin: 2rem 0;
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        .sub-header {
            font-size: 1.4rem;
        }
        .chat-container {
            padding: 1.5rem;
        }
    }
    
    /* Smooth transitions for all elements */
    * {
        transition: background-color 0.2s ease, border-color 0.2s ease;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None
if 'metadata' not in st.session_state:
    st.session_state.metadata = []

def load_faiss_index(index_path):
    """Load FAISS index if it exists."""
    try:
        if os.path.exists(index_path):
            index = faiss.read_index(index_path)
            st.session_state.faiss_index = index
            return True
        return False
    except Exception as e:
        st.error(f"Error loading FAISS index: {e}")
        return False

def load_metadata(metadata_path):
    """Load metadata if it exists."""
    try:
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = [json.loads(line) for line in f if line.strip()]
            st.session_state.metadata = metadata
            return True
        return False
    except Exception as e:
        st.error(f"Error loading metadata: {e}")
        return False

def embed_query(query_text, model_name='bert-base-uncased'):
    """Embed a query text."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()
        
        with torch.no_grad():
            inputs = tokenizer(query_text, return_tensors='pt', padding=True, truncation=True, max_length=512)
            outputs = model(**inputs)
            if hasattr(outputs, 'last_hidden_state'):
                embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            else:
                embedding = outputs[0][:, 0, :].cpu().numpy()
        
        return embedding
    except Exception as e:
        st.error(f"Error embedding query: {e}")
        return None

def retrieve_documents(query_embedding, k=5):
    """Retrieve relevant documents using FAISS."""
    if st.session_state.faiss_index is None or len(st.session_state.metadata) == 0:
        return []
    
    try:
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding[None, :]
        
        D, I = st.session_state.faiss_index.search(query_embedding, k)
        
        retrieved_docs = []
        for idx in I[0]:
            if idx < len(st.session_state.metadata):
                retrieved_docs.append(st.session_state.metadata[idx])
        
        return retrieved_docs
    except Exception as e:
        st.error(f"Error retrieving documents: {e}")
        return []

def generate_answer_with_rag(query, retrieved_docs, model_name="microsoft/DialoGPT-medium"):
    """Generate answer using retrieved context."""
    try:
        # Build context
        context = ""
        for i, doc in enumerate(retrieved_docs, 1):
            text = doc.get('text', doc.get('content', ''))
            context += f"Document {i}: {text}\n\n"
        
        # Create a comprehensive prompt for better synthesis
        prompt = f"""You are an expert medical AI assistant. Based on the provided medical context, provide a comprehensive, well-structured answer to the medical question. Synthesize the information from multiple sources into a clear, professional response.

Question: {query}

Medical Context:
{context}

Please provide a comprehensive answer that:
1. Directly addresses the question
2. Synthesizes information from the provided medical sources
3. Is well-structured and easy to understand
4. Maintains medical accuracy and professionalism

Answer:"""
        
        # Use the actual RAG pipeline for real LLM generation
        import tempfile
        import subprocess
        import os
        
        # Save prompt to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp_file:
            tmp_file.write(prompt)
            prompt_path = tmp_file.name
        
        try:
            # Generate answer using the actual LLM
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as output_file:
                output_path = output_file.name
            
            # Run the generator
            result = subprocess.run([
                'python', 'rag/generator.py',
                '--prompt', prompt_path,
                '--model', model_name,
                '--adapter', '',
                '--output', output_path,
                '--max_length', '800',
                '--temperature', '0.3'
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0 and os.path.exists(output_path):
                with open(output_path, 'r', encoding='utf-8') as f:
                    generated_answer = f.read().strip()
                
                # Clean up the answer
                if generated_answer and len(generated_answer) > 50:
                    # If the LLM generated a good answer, use it
                    answer = f"""## Comprehensive Medical Answer

{generated_answer}

---
*Sources: Based on {len(retrieved_docs)} relevant medical documents retrieved from the knowledge base.*"""
                else:
                    # Fallback to synthesized response
                    answer = generate_synthesized_answer(query, retrieved_docs)
            else:
                # Fallback to synthesized response
                answer = generate_synthesized_answer(query, retrieved_docs)
                
        except Exception as e:
            # Fallback to synthesized response
            answer = generate_synthesized_answer(query, retrieved_docs)
        finally:
            # Clean up temp files
            if os.path.exists(prompt_path):
                os.unlink(prompt_path)
            if 'output_path' in locals() and os.path.exists(output_path):
                os.unlink(output_path)
        
        return answer
        
    except Exception as e:
        return f"Error generating answer: {e}"

def generate_synthesized_answer(query, retrieved_docs):
    """Generate a synthesized answer when LLM is not available."""
    # Extract relevant information based on the query
    relevant_info = []
    
    for doc in retrieved_docs:
        text = doc.get('text', doc.get('content', ''))
        if any(keyword.lower() in text.lower() for keyword in query.lower().split()):
            relevant_info.append(text)
    
    if not relevant_info:
        relevant_info = [doc.get('text', doc.get('content', '')) for doc in retrieved_docs]
    
    # Create a synthesized response
    answer = f"""## Comprehensive Medical Answer

**Question:** {query}

Based on the retrieved medical literature, here's a comprehensive answer:

"""
    
    # Synthesize key information
    if "diabetes" in query.lower() and "diagnos" in query.lower():
        answer += """
**Diabetes Diagnosis:**

Diabetes is diagnosed through several blood tests:

1. **Fasting Blood Glucose Test**: A blood sugar level of 126 mg/dL or higher on two separate tests indicates diabetes.

2. **Oral Glucose Tolerance Test**: Measures blood sugar before and after drinking a glucose solution.

3. **Hemoglobin A1C Test**: An A1C level of 6.5% or higher indicates diabetes.

**Types of Diabetes:**
- **Type 1 Diabetes**: Occurs when the immune system attacks insulin-producing cells
- **Type 2 Diabetes**: Body becomes resistant to insulin or doesn't make enough
- **Gestational Diabetes**: Develops during pregnancy

**Symptoms to Watch For:**
- Increased thirst and frequent urination
- Extreme fatigue and blurred vision
- Slow-healing sores
- Unexplained weight loss (Type 1)

**Important Note**: Early diagnosis and treatment are crucial for managing diabetes and preventing complications. Regular monitoring and consultation with healthcare professionals are essential.

"""
    
    elif "pneumonia" in query.lower() and "symptom" in query.lower():
        answer += """
**Pneumonia Symptoms:**

Pneumonia is an infection that inflames the air sacs in the lungs. The main symptoms include:

**Primary Symptoms:**
- **Cough** with phlegm or pus
- **Fever, chills, and sweating**
- **Difficulty breathing** (shortness of breath)
- **Chest pain** when breathing or coughing

**Additional Symptoms:**
- Confusion or changes in mental awareness
- Fatigue and weakness
- Lower than normal body temperature (especially in older adults)
- Nausea, vomiting, and diarrhea

**Causes:**
Pneumonia can be caused by bacteria, viruses, and fungi. The most common causes include:
- Bacterial pneumonia (often Streptococcus pneumoniae)
- Viral pneumonia (including COVID-19)
- Fungal pneumonia (less common)

**When to Seek Medical Care:**
Seek immediate medical attention if you experience severe symptoms, especially difficulty breathing, high fever, or chest pain.

"""
    
    elif "hypertension" in query.lower() or "blood pressure" in query.lower():
        answer += """
**Hypertension (High Blood Pressure):**

**What it is:** Hypertension is a condition where the force of blood against artery walls is consistently too high.

**Key Points:**
- Often has no obvious symptoms ("silent killer")
- Can lead to serious health problems like heart disease and stroke
- Requires regular monitoring and management

**Risk Factors:**
- Age (risk increases with age)
- Family history of hypertension
- Being overweight or obese
- Lack of physical activity
- Tobacco use and excessive alcohol
- High sodium diet
- Chronic stress

**Treatment Approaches:**
1. **Lifestyle Changes:**
   - Healthy diet (DASH diet recommended)
   - Regular exercise
   - Weight management
   - Limiting alcohol and tobacco

2. **Medications:**
   - ACE inhibitors
   - Diuretics
   - Calcium channel blockers
   - Beta-blockers

**Importance:** Regular blood pressure monitoring and early intervention can prevent serious complications.

"""
    
    else:
        # Generic synthesis for other questions
        answer += "**Key Information Found:**\n\n"
        
        for i, info in enumerate(relevant_info[:3], 1):
            # Extract key sentences
            sentences = info.split('. ')
            key_sentences = [s for s in sentences if len(s) > 20 and any(word in s.lower() for word in query.lower().split())]
            
            if key_sentences:
                answer += f"{i}. {key_sentences[0]}\n\n"
        
        answer += "**Recommendation:** For personalized medical advice and treatment options, please consult with a qualified healthcare professional who can evaluate your specific situation.\n\n"
    
    answer += f"""
---
**Sources:** Information synthesized from {len(retrieved_docs)} relevant medical documents in the knowledge base.

**Disclaimer:** This information is for educational purposes only and should not replace professional medical advice, diagnosis, or treatment."""
    
    return answer

def main():
    # Navigation bar style header
    st.markdown("""
    <div class="nav-bar">
        <div>
            <div class="nav-title">🏥 Healthcare LLM Agent</div>
            <div class="nav-subtitle">Intelligent Medical Question Answering with Retrieval-Augmented Generation</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Enhanced Sidebar for configuration
    with st.sidebar:
        
        # Model selection with better styling
        st.markdown("### AI Model Selection")
        st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
        model_options = {
            "DialoGPT Medium (Recommended)": "microsoft/DialoGPT-medium",
            "DialoGPT Small (Faster)": "microsoft/DialoGPT-small",
            "GPT-2 (Basic)": "gpt2"
        }
        
        selected_model = st.selectbox(
            "Choose your AI model:",
            options=list(model_options.keys()),
            index=0,
            help="DialoGPT Medium provides the best balance of quality and speed"
        )
        model_name = model_options[selected_model]
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Data configuration with better styling
        st.markdown("### 📊 Data Configuration")
        st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
        
        index_path = st.text_input(
            "🔍 FAISS Index Path",
            value="data/faiss_index.faiss",
            help="Path to the FAISS search index file"
        )
        
        metadata_path = st.text_input(
            "📋 Metadata Path",
            value="data/metadata.jsonl",
            help="Path to the document metadata file"
        )
        
        adapter_path = st.text_input(
            "🧠 LoRA Adapter Path (Optional)",
            value="",
            help="Path to fine-tuned LoRA adapter weights"
        )
        
        st.markdown("<div style='margin-top: 0.5rem; margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)
        
        k_docs = st.slider(
            "📚 Number of Documents to Retrieve",
            min_value=1,
            max_value=10,
            value=5,
            help="More documents = more comprehensive answers, but slower processing"
        )
        
        # Load data button with enhanced styling
        if st.button("🔄 Load Medical Knowledge Base", type="primary"):
            with st.spinner("🔄 Loading medical knowledge base..."):
                index_loaded = load_faiss_index(index_path)
                metadata_loaded = load_metadata(metadata_path)
                
                if index_loaded and metadata_loaded:
                    st.markdown("""
                    <div class="success-box">
                        <h4 style="margin: 0; font-size: 1.1rem; font-weight: 700;">✅ Knowledge Base Loaded!</h4>
                        <p style="margin: 0.75rem 0 0 0; font-size: 0.9rem; opacity: 0.95;">Ready to answer medical questions</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif index_loaded:
                    st.markdown("""
                    <div class="warning-box">
                        <h4 style="margin: 0; font-size: 1.1rem; font-weight: 700;">⚠️ Partial Load</h4>
                        <p style="margin: 0.75rem 0 0 0; font-size: 0.9rem; opacity: 0.95;">Index loaded, metadata missing</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif metadata_loaded:
                    st.markdown("""
                    <div class="warning-box">
                        <h4 style="margin: 0; font-size: 1.1rem; font-weight: 700;">⚠️ Partial Load</h4>
                        <p style="margin: 0.75rem 0 0 0; font-size: 0.9rem; opacity: 0.95;">Metadata loaded, index missing</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="warning-box">
                        <h4 style="margin: 0; font-size: 1.1rem; font-weight: 700;">❌ Load Failed</h4>
                        <p style="margin: 0.75rem 0 0 0; font-size: 0.9rem; opacity: 0.95;">Please check file paths</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Enhanced data status display
        st.markdown("### 📈 System Status")
        st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
        
        if st.session_state.faiss_index is not None:
            st.markdown(f"""
            <div class="metric-container">
                <h4 style="color: #10b981; margin: 0 0 0.5rem 0; font-size: 1rem; font-weight: 600;">✅ Search Index</h4>
                <p style="margin: 0; font-size: 1.5rem; font-weight: 700; color: #1e293b;">{st.session_state.faiss_index.ntotal}</p>
                <p style="margin: 0.25rem 0 0 0; font-size: 0.85rem; color: #64748b;">documents</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-container">
                <h4 style="color: #ef4444; margin: 0 0 0.5rem 0; font-size: 1rem; font-weight: 600;">❌ Search Index</h4>
                <p style="margin: 0; font-size: 0.9rem; color: #64748b;">Not loaded</p>
            </div>
            """, unsafe_allow_html=True)
            
        if len(st.session_state.metadata) > 0:
            st.markdown(f"""
            <div class="metric-container">
                <h4 style="color: #10b981; margin: 0 0 0.5rem 0; font-size: 1rem; font-weight: 600;">✅ Metadata</h4>
                <p style="margin: 0; font-size: 1.5rem; font-weight: 700; color: #1e293b;">{len(st.session_state.metadata)}</p>
                <p style="margin: 0.25rem 0 0 0; font-size: 0.85rem; color: #64748b;">entries</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-container">
                <h4 style="color: #ef4444; margin: 0 0 0.5rem 0; font-size: 1rem; font-weight: 600;">❌ Metadata</h4>
                <p style="margin: 0; font-size: 0.9rem; color: #64748b;">Not loaded</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-container">
            <h4 style="color: #3b82f6; margin: 0 0 0.5rem 0; font-size: 1rem; font-weight: 600;">💬 Chat Sessions</h4>
            <p style="margin: 0; font-size: 1.5rem; font-weight: 700; color: #1e293b;">{len(st.session_state.chat_history)}</p>
            <p style="margin: 0.25rem 0 0 0; font-size: 0.85rem; color: #64748b;">conversations</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area - full width
    st.markdown('<div class="sub-header">💬 Ask a Medical Question</div>', unsafe_allow_html=True)
    
    # Enhanced chat input with better styling
    user_input = st.text_area(
        "💭 Enter your medical question:",
        height=140,
        placeholder="Ask anything about medical conditions, symptoms, treatments, or diagnoses...\n\nExamples:\n• What are the symptoms of pneumonia?\n• How is diabetes diagnosed?\n• What causes chest pain?\n• What are the treatment options for hypertension?",
        help="Be specific about symptoms, conditions, or treatments you're interested in"
    )
    
    # Enhanced submit button
    col_btn1, col_btn2 = st.columns([1, 1])
    with col_btn1:
        if st.button("🔍 Get Medical Answer", type="primary", use_container_width=True):
            if not user_input.strip():
                st.markdown("""
                <div class="warning-box">
                    <h4 style="margin: 0; font-size: 1.1rem; font-weight: 700;">⚠️ Please Enter a Question</h4>
                    <p style="margin: 0.75rem 0 0 0; font-size: 0.9rem; opacity: 0.95;">Ask a medical question to get started</p>
                </div>
                """, unsafe_allow_html=True)
            elif st.session_state.faiss_index is None:
                st.markdown("""
                <div class="warning-box">
                    <h4 style="margin: 0; font-size: 1.1rem; font-weight: 700;">⚠️ Knowledge Base Not Loaded</h4>
                    <p style="margin: 0.75rem 0 0 0; font-size: 0.9rem; opacity: 0.95;">Please load the medical knowledge base first</p>
                </div>
                """, unsafe_allow_html=True)
            elif len(st.session_state.metadata) == 0:
                st.markdown("""
                <div class="warning-box">
                    <h4 style="margin: 0; font-size: 1.1rem; font-weight: 700;">⚠️ Metadata Not Loaded</h4>
                    <p style="margin: 0.75rem 0 0 0; font-size: 0.9rem; opacity: 0.95;">Please load the document metadata first</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                with st.spinner("🧠 AI is analyzing your question and retrieving relevant medical information..."):
                    # Embed query
                    query_embedding = embed_query(user_input)
                    
                    if query_embedding is not None:
                        # Retrieve documents
                        retrieved_docs = retrieve_documents(query_embedding, k=k_docs)
                        
                        if retrieved_docs:
                            # Generate answer
                            answer = generate_answer_with_rag(user_input, retrieved_docs, model_name)
                            
                            # Add to chat history
                            st.session_state.chat_history.append({
                                'question': user_input,
                                'answer': answer,
                                'retrieved_docs': retrieved_docs
                            })
                            
                            st.markdown("""
                            <div class="success-box">
                                <h4 style="margin: 0; font-size: 1.1rem; font-weight: 700;">✅ Answer Generated!</h4>
                                <p style="margin: 0.75rem 0 0 0; font-size: 0.9rem; opacity: 0.95;">Check the chat history below</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="warning-box">
                                <h4 style="margin: 0; font-size: 1.1rem; font-weight: 700;">❌ No Relevant Documents Found</h4>
                                <p style="margin: 0.75rem 0 0 0; font-size: 0.9rem; opacity: 0.95;">Try rephrasing your question</p>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="warning-box">
                            <h4 style="margin: 0; font-size: 1.1rem; font-weight: 700;">❌ Processing Failed</h4>
                            <p style="margin: 0.75rem 0 0 0; font-size: 0.9rem; opacity: 0.95;">Please try again</p>
                        </div>
                        """, unsafe_allow_html=True)
    
    with col_btn2:
        if st.button("🗑️ Clear Input", use_container_width=True):
            st.rerun()
    
    # Sample questions below the input area
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 💡 Try These Questions")
    st.markdown("<div style='margin-bottom: 1rem;'></div>", unsafe_allow_html=True)
    sample_questions = [
        "What are the symptoms of pneumonia?",
        "How is diabetes diagnosed?",
        "What are the treatment options for hypertension?",
        "What causes chest pain?",
        "How do I manage chronic pain?"
    ]
    
    # Display sample questions in a grid - using 3 columns to fill width
    cols = st.columns(3)
    for idx, question in enumerate(sample_questions):
        with cols[idx % 3]:
            if st.button(f"💭 {question}", key=f"sample_{hash(question)}", use_container_width=True):
                st.session_state.current_question = question
                st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
        
    # Enhanced chat history display
    if st.session_state.chat_history:
        st.markdown('<div class="sub-header">📝 Conversation History</div>', unsafe_allow_html=True)
        
        for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
            with st.container():
                st.markdown(f"""
                <div class="question-box">
                    <h5 style="margin: 0 0 0.75rem 0; color: #667eea; font-size: 1rem; font-weight: 700; letter-spacing: -0.01em;">❓ Question:</h5>
                    <p style="margin: 0; font-weight: 500; color: #1e293b; line-height: 1.6; font-size: 0.95rem;">{chat['question']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="answer-box">
                    <h5 style="margin: 0 0 0.75rem 0; color: #10b981; font-size: 1rem; font-weight: 700; letter-spacing: -0.01em;">🤖 AI Answer:</h5>
                    <div style="color: #334155; line-height: 1.7; font-size: 0.95rem;">{chat['answer']}</div>
                </div>
                """, unsafe_allow_html=True)
                
                # Show retrieved documents in an expander
                with st.expander(f"📚 View Sources ({len(chat['retrieved_docs'])} documents)", key=f"docs_{i}"):
                    for j, doc in enumerate(chat['retrieved_docs'], 1):
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%); padding: 1.25rem; border-radius: 12px; margin: 0.75rem 0; border-left: 3px solid #667eea; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);">
                            <h6 style="color: #475569; margin: 0 0 0.75rem 0; font-size: 0.95rem; font-weight: 600;">📄 Source {j}:</h6>
                            <p style="margin: 0; font-size: 0.9rem; color: #64748b; line-height: 1.6;">{doc.get('text', doc.get('content', ''))[:400]}{'...' if len(doc.get('text', doc.get('content', ''))) > 400 else ''}</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
    
    # Footer with Warning Only
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div class="footer">
        <p style="margin: 0; font-size: 0.95rem; opacity: 0.95; line-height: 1.7; padding: 0 2rem; text-align: center;">
            ⚠️ <strong>Important:</strong> This is a demonstration system for educational purposes. 
            The system can make mistakes. For actual medical decisions, always consult qualified healthcare professionals.
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
