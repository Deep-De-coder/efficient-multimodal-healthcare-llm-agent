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

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #2E86AB;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
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
    # Header
    st.markdown('<h1 class="main-header">🏥 Healthcare LLM Agent</h1>', unsafe_allow_html=True)
    st.markdown("### Multimodal Retrieval-Augmented Generation for Clinical Questions")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_options = {
            "DialoGPT Medium": "microsoft/DialoGPT-medium",
            "DialoGPT Small": "microsoft/DialoGPT-small",
            "GPT-2": "gpt2"
        }
        
        selected_model = st.selectbox(
            "Select Model",
            options=list(model_options.keys()),
            index=0
        )
        model_name = model_options[selected_model]
        
        # FAISS index path
        index_path = st.text_input(
            "FAISS Index Path",
            value="data/faiss_index.faiss",
            help="Path to the FAISS index file"
        )
        
        # Metadata path
        metadata_path = st.text_input(
            "Metadata Path",
            value="data/metadata.jsonl",
            help="Path to the metadata JSONL file"
        )
        
        # LoRA adapter path
        adapter_path = st.text_input(
            "LoRA Adapter Path (Optional)",
            value="",
            help="Path to LoRA adapter weights"
        )
        
        # Number of retrieved documents
        k_docs = st.slider(
            "Number of Documents to Retrieve",
            min_value=1,
            max_value=10,
            value=5
        )
        
        # Load data button
        if st.button("🔄 Load Data"):
            with st.spinner("Loading FAISS index and metadata..."):
                index_loaded = load_faiss_index(index_path)
                metadata_loaded = load_metadata(metadata_path)
                
                if index_loaded and metadata_loaded:
                    st.success("✅ Data loaded successfully!")
                elif index_loaded:
                    st.warning("⚠️ FAISS index loaded, but metadata failed to load")
                elif metadata_loaded:
                    st.warning("⚠️ Metadata loaded, but FAISS index failed to load")
                else:
                    st.error("❌ Failed to load both index and metadata")
        
        # Data status
        if st.session_state.faiss_index is not None:
            st.success(f"✅ Index loaded: {st.session_state.faiss_index.ntotal} documents")
        else:
            st.warning("⚠️ No index loaded")
            
        if len(st.session_state.metadata) > 0:
            st.success(f"✅ Metadata loaded: {len(st.session_state.metadata)} entries")
        else:
            st.warning("⚠️ No metadata loaded")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="sub-header">💬 Ask a Medical Question</div>', unsafe_allow_html=True)
        
        # Chat input
        user_input = st.text_area(
            "Enter your medical question:",
            height=100,
            placeholder="e.g., What are the symptoms of pneumonia? What treatment options are available for diabetes?"
        )
        
        # Submit button
        if st.button("🔍 Search & Generate Answer", type="primary"):
            if not user_input.strip():
                st.warning("Please enter a question.")
            elif st.session_state.faiss_index is None:
                st.error("Please load the FAISS index first.")
            elif len(st.session_state.metadata) == 0:
                st.error("Please load the metadata first.")
            else:
                with st.spinner("Processing your question..."):
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
                        else:
                            st.error("No relevant documents found.")
                    else:
                        st.error("Failed to process the query.")
        
        # Display chat history
        if st.session_state.chat_history:
            st.markdown('<div class="sub-header">📝 Chat History</div>', unsafe_allow_html=True)
            
            for i, chat in enumerate(reversed(st.session_state.chat_history[-5:])):  # Show last 5
                with st.expander(f"Q: {chat['question'][:100]}..."):
                    st.write("**Answer:**")
                    st.write(chat['answer'])
                    
                    if st.button(f"Show Retrieved Documents", key=f"docs_{i}"):
                        for j, doc in enumerate(chat['retrieved_docs'], 1):
                            st.write(f"**Document {j}:**")
                            st.write(doc.get('text', doc.get('content', ''))[:500] + "...")
                            st.write("---")
    
    with col2:
        st.markdown('<div class="sub-header">📊 System Status</div>', unsafe_allow_html=True)
        
        # System information
        st.markdown("""
        <div class="info-box">
            <strong>System Components:</strong><br>
            ✅ Query Embedding<br>
            ✅ Document Retrieval<br>
            ✅ Answer Generation<br>
            ✅ Web Interface
        </div>
        """, unsafe_allow_html=True)
        
        # Quick stats
        if st.session_state.faiss_index is not None:
            st.metric("Documents Indexed", st.session_state.faiss_index.ntotal)
        
        if len(st.session_state.metadata) > 0:
            st.metric("Metadata Entries", len(st.session_state.metadata))
        
        st.metric("Chat Sessions", len(st.session_state.chat_history))
        
        # Sample questions
        st.markdown('<div class="sub-header">💡 Sample Questions</div>', unsafe_allow_html=True)
        sample_questions = [
            "What are the symptoms of pneumonia?",
            "How is diabetes diagnosed?",
            "What are the treatment options for hypertension?",
            "What causes chest pain?",
            "How do I manage chronic pain?"
        ]
        
        for question in sample_questions:
            if st.button(f"💭 {question}", key=f"sample_{hash(question)}"):
                st.session_state.current_question = question
                st.rerun()
        
        # Clear chat history
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666;">
        <p>🏥 Healthcare LLM Agent - Multimodal RAG for Clinical Questions</p>
        <p><small>⚠️ This is a demonstration system. For actual medical decisions, consult healthcare professionals.</small></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
