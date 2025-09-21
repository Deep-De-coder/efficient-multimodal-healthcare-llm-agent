# 🚀 Quick Start Guide

## Healthcare LLM Agent - Complete Setup & Usage

This guide will get you up and running with the Healthcare LLM Agent in minutes!

## 📋 Prerequisites

- Python 3.10 or higher
- 8GB+ RAM (16GB+ recommended)
- GPU with CUDA support (optional but recommended)

## ⚡ Quick Setup (5 minutes)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Setup Script
```bash
python setup_pipeline.py
```

This will:
- ✅ Create sample medical data
- ✅ Generate text embeddings
- ✅ Build FAISS search index
- ✅ Prepare training data
- ✅ Test the RAG pipeline

### 3. Launch the Web Interface
```bash
streamlit run streamlit_app.py
```

Open your browser to `http://localhost:8501` and start asking medical questions!

## 🔧 Manual Setup (Step by Step)

If you prefer to run each component manually:

### Step 1: Data Preparation
```bash
# The sample data is already created, but you can add your own:
python data_ingestion/extract_mimic_cxr.py --input your_data.csv --output data/mimic_cxr.jsonl
python preprocessing/preprocess_cxr.py --input data/mimic_cxr.jsonl --output data/cxr_chunks.jsonl
```

### Step 2: Create Embeddings
```bash
python embedding/embed_text.py \
  --input data/sample_medical_data.jsonl \
  --output_emb data/embeddings.npy \
  --output_meta data/metadata.jsonl \
  --model bert-base-uncased
```

### Step 3: Build Search Index
```bash
python embedding/faiss_index.py \
  --emb data/embeddings.npy \
  --output data/faiss_index.faiss
```

### Step 4: Test RAG Pipeline
```bash
python rag/pipeline.py \
  --query "What are the symptoms of pneumonia?" \
  --index data/faiss_index.faiss \
  --metadata data/metadata.jsonl \
  --model microsoft/DialoGPT-medium \
  --output outputs/answer.txt
```

## 🎯 Usage Examples

### Web Interface
1. Open `http://localhost:8501`
2. Load your data (FAISS index and metadata)
3. Ask medical questions like:
   - "What are the symptoms of pneumonia?"
   - "How is diabetes diagnosed?"
   - "What causes chest pain?"

### Command Line
```bash
# Direct RAG query
python rag/pipeline.py \
  --query "What are the treatment options for hypertension?" \
  --index data/faiss_index.faiss \
  --metadata data/metadata.jsonl \
  --model microsoft/DialoGPT-medium \
  --output outputs/hypertension_answer.txt

# Generate embeddings for new documents
python embedding/embed_text.py \
  --input new_medical_docs.jsonl \
  --output_emb new_embeddings.npy \
  --output_meta new_metadata.jsonl
```

### Training a Custom Model
```bash
# Prepare QA pairs
python finetune/prepare_qa_pairs.py \
  --input data/raw_qa_data.json \
  --output data/training_data.jsonl

# Fine-tune with LoRA
python finetune/lora_finetune.py \
  --data data/training_data.jsonl \
  --model microsoft/DialoGPT-medium \
  --adapter_output models/healthcare_lora \
  --epochs 3 \
  --batch_size 2

# Use fine-tuned model
python rag/generator.py \
  --prompt prompts/medical_question.txt \
  --model microsoft/DialoGPT-medium \
  --adapter models/healthcare_lora \
  --output outputs/custom_answer.txt
```

## 📁 Project Structure

```
efficient-multimodal-healthcare-llm-agent/
├── data/                          # Data files
│   ├── sample_medical_data.jsonl  # Sample medical documents
│   ├── sample_qa_pairs.jsonl      # Sample Q&A pairs
│   ├── faiss_index.faiss         # Search index
│   ├── metadata.jsonl            # Document metadata
│   └── embeddings.npy            # Text embeddings
├── data_ingestion/               # Data extraction scripts
├── preprocessing/                # Data preprocessing
├── embedding/                    # Text/image embedding
├── rag/                         # RAG pipeline components
├── finetune/                    # Model training scripts
├── inference/                   # Model inference
├── streamlit_app.py             # Web interface
├── setup_pipeline.py            # Automated setup
└── config.json                  # System configuration
```

## 🔍 Testing the System

### Test Individual Components
```bash
# Test embedding
python embedding/embed_text.py --help

# Test retrieval
python rag/retriever.py --index data/faiss_index.faiss --query_emb query.npy --k 5

# Test generation
python rag/generator.py --prompt test_prompt.txt --model microsoft/DialoGPT-medium --output test_answer.txt
```

### Test Complete Pipeline
```bash
python setup_pipeline.py  # This includes a full pipeline test
```

## 🚨 Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Reduce batch size
python embedding/embed_text.py --batch_size 4
python finetune/lora_finetune.py --batch_size 1
```

**2. Model Download Issues**
```bash
# Use smaller models
python rag/generator.py --model microsoft/DialoGPT-small
```

**3. FAISS Index Not Found**
```bash
# Rebuild the index
python embedding/faiss_index.py --emb data/embeddings.npy --output data/faiss_index.faiss
```

**4. Dependencies Issues**
```bash
# Install specific versions
pip install torch==2.0.0 transformers==4.30.0
```

### Getting Help

1. Check the logs in the console output
2. Verify all required files exist in the `data/` directory
3. Ensure you have sufficient disk space and RAM
4. Try running individual components to isolate issues

## 🎉 Success Indicators

You'll know the system is working when:

1. ✅ Setup script completes without errors
2. ✅ Web interface loads at `http://localhost:8501`
3. ✅ You can ask questions and get relevant medical information
4. ✅ Retrieved documents are relevant to your queries
5. ✅ Generated answers reference the retrieved context

## 📈 Next Steps

Once you have the basic system running:

1. **Add Your Own Data**: Replace sample data with real medical documents
2. **Fine-tune Models**: Train on your specific medical domain
3. **Scale Up**: Use larger models and more powerful hardware
4. **Deploy**: Set up for production use with proper security

## ⚠️ Important Notes

- This is a demonstration system for educational purposes
- **NOT for actual medical diagnosis or treatment decisions**
- Always consult healthcare professionals for medical advice
- Ensure compliance with healthcare data regulations (HIPAA, etc.)

---

🎯 **Ready to start?** Run `python setup_pipeline.py` and then `streamlit run streamlit_app.py`!
