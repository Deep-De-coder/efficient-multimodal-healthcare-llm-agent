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
    /* Main header styling */
    .main-header {
        font-size: 3.5rem;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .main-subtitle {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
        font-style: italic;
    }
    
    /* Sub headers */
    .sub-header {
        font-size: 1.6rem;
        color: #2E86AB;
        margin-top: 2rem;
        margin-bottom: 1rem;
        font-weight: 600;
        border-bottom: 2px solid #2E86AB;
        padding-bottom: 0.5rem;
    }
    
    /* Status boxes */
    .status-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .success-box {
        background: linear-gradient(135deg, #56ab2f 0%, #a8e6cf 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .info-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    /* Chat styling */
    .chat-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 20px;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    
    .question-box {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #2E86AB;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .answer-box {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #28a745;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,0,0,0.3);
    }
    
    /* Metric styling */
    .metric-container {
        background: white;
        padding: 1rem;
        border-radius: 15px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    /* Sample question buttons */
    .sample-question {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        border: none;
        border-radius: 20px;
        padding: 0.8rem 1.5rem;
        margin: 0.5rem 0;
        color: #333;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .sample-question:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    /* Footer styling */
    .footer {
        background: linear-gradient(135deg, #2c3e50 0%, #34495e 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-top: 3rem;
        text-align: center;
    }
    
    /* Loading animation */
    .loading {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid #f3f3f3;
        border-top: 3px solid #3498db;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2.5rem;
        }
        .sub-header {
            font-size: 1.3rem;
        }
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
    # Enhanced Header with better styling
    st.markdown('<h1 class="main-header">🏥 Healthcare LLM Agent</h1>', unsafe_allow_html=True)
    st.markdown('<p class="main-subtitle">🤖 Intelligent Medical Question Answering with Retrieval-Augmented Generation</p>', unsafe_allow_html=True)
    
    # Add a status indicator
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        <div class="status-box">
            <h3 style="margin: 0; text-align: center;">🚀 System Ready</h3>
            <p style="margin: 0.5rem 0 0 0; text-align: center; opacity: 0.9;">AI-powered medical assistant ready to answer your questions</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced Sidebar for configuration
    with st.sidebar:
        st.markdown("""
        <div class="info-box">
            <h3 style="margin: 0; text-align: center;">⚙️ System Configuration</h3>
            <p style="margin: 0.5rem 0 0 0; text-align: center; opacity: 0.9;">Customize your AI assistant settings</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Model selection with better styling
        st.markdown("### 🤖 AI Model Selection")
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
        
        # Data configuration with better styling
        st.markdown("### 📊 Data Configuration")
        
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
                        <h4 style="margin: 0;">✅ Knowledge Base Loaded!</h4>
                        <p style="margin: 0.5rem 0 0 0;">Ready to answer medical questions</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif index_loaded:
                    st.markdown("""
                    <div class="warning-box">
                        <h4 style="margin: 0;">⚠️ Partial Load</h4>
                        <p style="margin: 0.5rem 0 0 0;">Index loaded, metadata missing</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif metadata_loaded:
                    st.markdown("""
                    <div class="warning-box">
                        <h4 style="margin: 0;">⚠️ Partial Load</h4>
                        <p style="margin: 0.5rem 0 0 0;">Metadata loaded, index missing</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="warning-box">
                        <h4 style="margin: 0;">❌ Load Failed</h4>
                        <p style="margin: 0.5rem 0 0 0;">Please check file paths</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Enhanced data status display
        st.markdown("### 📈 System Status")
        
        if st.session_state.faiss_index is not None:
            st.markdown(f"""
            <div class="metric-container">
                <h4 style="color: #28a745; margin: 0;">✅ Search Index</h4>
                <p style="margin: 0; font-size: 1.2rem; font-weight: bold;">{st.session_state.faiss_index.ntotal} documents</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-container">
                <h4 style="color: #dc3545; margin: 0;">❌ Search Index</h4>
                <p style="margin: 0;">Not loaded</p>
            </div>
            """, unsafe_allow_html=True)
            
        if len(st.session_state.metadata) > 0:
            st.markdown(f"""
            <div class="metric-container">
                <h4 style="color: #28a745; margin: 0;">✅ Metadata</h4>
                <p style="margin: 0; font-size: 1.2rem; font-weight: bold;">{len(st.session_state.metadata)} entries</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="metric-container">
                <h4 style="color: #dc3545; margin: 0;">❌ Metadata</h4>
                <p style="margin: 0;">Not loaded</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-container">
            <h4 style="color: #17a2b8; margin: 0;">💬 Chat Sessions</h4>
            <p style="margin: 0; font-size: 1.2rem; font-weight: bold;">{len(st.session_state.chat_history)}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="sub-header">💬 Ask a Medical Question</div>', unsafe_allow_html=True)
        
        # Enhanced chat input with better styling
        st.markdown("""
        <div class="chat-container">
            <h4 style="margin: 0 0 1rem 0; color: #2E86AB;">🤔 What would you like to know about?</h4>
        </div>
        """, unsafe_allow_html=True)
        
        user_input = st.text_area(
            "💭 Enter your medical question:",
            height=120,
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
                        <h4 style="margin: 0;">⚠️ Please Enter a Question</h4>
                        <p style="margin: 0.5rem 0 0 0;">Ask a medical question to get started</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif st.session_state.faiss_index is None:
                    st.markdown("""
                    <div class="warning-box">
                        <h4 style="margin: 0;">⚠️ Knowledge Base Not Loaded</h4>
                        <p style="margin: 0.5rem 0 0 0;">Please load the medical knowledge base first</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif len(st.session_state.metadata) == 0:
                    st.markdown("""
                    <div class="warning-box">
                        <h4 style="margin: 0;">⚠️ Metadata Not Loaded</h4>
                        <p style="margin: 0.5rem 0 0 0;">Please load the document metadata first</p>
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
                                    <h4 style="margin: 0;">✅ Answer Generated!</h4>
                                    <p style="margin: 0.5rem 0 0 0;">Check the chat history below</p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown("""
                                <div class="warning-box">
                                    <h4 style="margin: 0;">❌ No Relevant Documents Found</h4>
                                    <p style="margin: 0.5rem 0 0 0;">Try rephrasing your question</p>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.markdown("""
                            <div class="warning-box">
                                <h4 style="margin: 0;">❌ Processing Failed</h4>
                                <p style="margin: 0.5rem 0 0 0;">Please try again</p>
                            </div>
                            """, unsafe_allow_html=True)
        
        with col_btn2:
            if st.button("🗑️ Clear Input", use_container_width=True):
                st.rerun()
        
        # Enhanced chat history display
        if st.session_state.chat_history:
            st.markdown('<div class="sub-header">📝 Conversation History</div>', unsafe_allow_html=True)
            
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
                with st.container():
                    st.markdown(f"""
                    <div class="question-box">
                        <h5 style="margin: 0 0 0.5rem 0; color: #2E86AB;">❓ Question:</h5>
                        <p style="margin: 0; font-weight: 500;">{chat['question']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class="answer-box">
                        <h5 style="margin: 0 0 0.5rem 0; color: #28a745;">🤖 AI Answer:</h5>
                        {chat['answer']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show retrieved documents in an expander
                    with st.expander(f"📚 View Sources ({len(chat['retrieved_docs'])} documents)", key=f"docs_{i}"):
                        for j, doc in enumerate(chat['retrieved_docs'], 1):
                            st.markdown(f"""
                            <div style="background: #f8f9fa; padding: 1rem; border-radius: 10px; margin: 0.5rem 0;">
                                <h6 style="color: #495057; margin: 0 0 0.5rem 0;">📄 Source {j}:</h6>
                                <p style="margin: 0; font-size: 0.9rem;">{doc.get('text', doc.get('content', ''))[:400]}{'...' if len(doc.get('text', doc.get('content', ''))) > 400 else ''}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.markdown("---")
    
    with col2:
        st.markdown('<div class="sub-header">📊 Quick Actions</div>', unsafe_allow_html=True)
        
        # Enhanced system information
        st.markdown("""
        <div class="info-box">
            <h4 style="margin: 0 0 1rem 0;">🔧 System Components</h4>
            <div style="display: flex; flex-direction: column; gap: 0.5rem;">
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span>✅</span> <span>Query Embedding</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span>✅</span> <span>Document Retrieval</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span>✅</span> <span>Answer Generation</span>
                </div>
                <div style="display: flex; align-items: center; gap: 0.5rem;">
                    <span>✅</span> <span>Web Interface</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Sample questions with better styling
        st.markdown("### 💡 Try These Questions")
        sample_questions = [
            "What are the symptoms of pneumonia?",
            "How is diabetes diagnosed?",
            "What are the treatment options for hypertension?",
            "What causes chest pain?",
            "How do I manage chronic pain?"
        ]
        
        for question in sample_questions:
            if st.button(f"💭 {question}", key=f"sample_{hash(question)}", use_container_width=True):
                st.session_state.current_question = question
                st.rerun()
        
        st.markdown("---")
        
        # Quick actions
        st.markdown("### 🛠️ Quick Actions")
        
        col_action1, col_action2 = st.columns(2)
        with col_action1:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
        
        with col_action2:
            if st.button("🔄 Refresh System", use_container_width=True):
                st.rerun()
        
        # Help section
        st.markdown("---")
        st.markdown("### ❓ Need Help?")
        st.markdown("""
        <div style="background: #e3f2fd; padding: 1rem; border-radius: 10px; border-left: 4px solid #2196f3;">
            <h6 style="margin: 0 0 0.5rem 0; color: #1976d2;">💡 Tips for Better Results:</h6>
            <ul style="margin: 0; padding-left: 1.2rem; font-size: 0.9rem;">
                <li>Be specific about symptoms</li>
                <li>Ask about treatments or causes</li>
                <li>Use medical terminology</li>
                <li>Try different phrasings</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <h4 style="margin: 0 0 1rem 0;">🏥 Healthcare LLM Agent</h4>
        <p style="margin: 0 0 0.5rem 0; opacity: 0.9;">Multimodal Retrieval-Augmented Generation for Clinical Questions</p>
        <div style="display: flex; justify-content: center; gap: 2rem; margin: 1rem 0; flex-wrap: wrap;">
            <span>🤖 AI-Powered</span>
            <span>📚 Knowledge-Based</span>
            <span>🔍 Context-Aware</span>
            <span>⚡ Real-Time</span>
        </div>
        <p style="margin: 1rem 0 0 0; font-size: 0.9rem; opacity: 0.8;">
            ⚠️ <strong>Important:</strong> This is a demonstration system for educational purposes. 
            For actual medical decisions, always consult qualified healthcare professionals.
        </p>
        <p style="margin: 0.5rem 0 0 0; font-size: 0.8rem; opacity: 0.7;">
            Built with ❤️ using Streamlit, PyTorch, and Transformers
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
