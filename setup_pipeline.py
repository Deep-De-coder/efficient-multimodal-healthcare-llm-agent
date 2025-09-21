#!/usr/bin/env python3
"""
Setup script to build the complete healthcare LLM pipeline.
This script creates embeddings, builds FAISS index, and prepares the system for use.
"""

import os
import json
import logging
import subprocess
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(cmd, description):
    """Run a command and log the result."""
    logger.info(f"Running: {description}")
    logger.info(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        logger.info(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error in {description}: {e}")
        return False

def setup_directories():
    """Create necessary directories."""
    directories = ['data', 'outputs', 'models', 'logs']
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"📁 Created directory: {directory}")

def embed_sample_data():
    """Create embeddings for sample medical data."""
    logger.info("🔄 Creating embeddings for sample data...")
    
    # Embed the sample medical data
    cmd = [
        'python', 'embedding/embed_text.py',
        '--input', 'data/sample_medical_data.jsonl',
        '--output_emb', 'data/medical_embeddings.npy',
        '--output_meta', 'data/metadata.jsonl',
        '--model', 'bert-base-uncased',
        '--batch_size', '8'
    ]
    
    return run_command(cmd, "Text embedding creation")

def build_faiss_index():
    """Build FAISS index from embeddings."""
    logger.info("🔄 Building FAISS index...")
    
    cmd = [
        'python', 'embedding/faiss_index.py',
        '--emb', 'data/medical_embeddings.npy',
        '--output', 'data/faiss_index.faiss',
        '--index_type', 'FlatL2'
    ]
    
    return run_command(cmd, "FAISS index building")

def create_sample_qa_pairs():
    """Prepare QA pairs for training."""
    logger.info("🔄 Preparing QA pairs for training...")
    
    cmd = [
        'python', 'finetune/prepare_qa_pairs.py',
        '--input', 'data/sample_qa_pairs.jsonl',
        '--output', 'data/formatted_qa_pairs.jsonl'
    ]
    
    return run_command(cmd, "QA pairs preparation")

def test_rag_pipeline():
    """Test the RAG pipeline with a sample query."""
    logger.info("🔄 Testing RAG pipeline...")
    
    test_query = "What are the symptoms of pneumonia?"
    
    cmd = [
        'python', 'rag/pipeline.py',
        '--query', test_query,
        '--index', 'data/faiss_index.faiss',
        '--metadata', 'data/metadata.jsonl',
        '--model', 'microsoft/DialoGPT-medium',
        '--output', 'outputs/test_answer.txt',
        '--k', '3'
    ]
    
    return run_command(cmd, "RAG pipeline test")

def main():
    """Main setup function."""
    logger.info("🚀 Starting Healthcare LLM Pipeline Setup")
    logger.info("=" * 50)
    
    # Step 1: Setup directories
    setup_directories()
    
    # Step 2: Check if sample data exists
    if not os.path.exists('data/sample_medical_data.jsonl'):
        logger.error("❌ Sample medical data not found. Please ensure data/sample_medical_data.jsonl exists.")
        return False
    
    if not os.path.exists('data/sample_qa_pairs.jsonl'):
        logger.error("❌ Sample QA pairs not found. Please ensure data/sample_qa_pairs.jsonl exists.")
        return False
    
    # Step 3: Create embeddings
    if not embed_sample_data():
        logger.error("❌ Failed to create embeddings. Setup aborted.")
        return False
    
    # Step 4: Build FAISS index
    if not build_faiss_index():
        logger.error("❌ Failed to build FAISS index. Setup aborted.")
        return False
    
    # Step 5: Prepare QA pairs
    if not create_sample_qa_pairs():
        logger.error("❌ Failed to prepare QA pairs. Setup aborted.")
        return False
    
    # Step 6: Test RAG pipeline
    if not test_rag_pipeline():
        logger.warning("⚠️ RAG pipeline test failed, but setup will continue.")
    
    # Step 7: Create configuration file
    config = {
        "faiss_index_path": "data/faiss_index.faiss",
        "metadata_path": "data/metadata.jsonl",
        "embeddings_path": "data/medical_embeddings.npy",
        "qa_pairs_path": "data/formatted_qa_pairs.jsonl",
        "default_model": "microsoft/DialoGPT-medium",
        "default_k": 5
    }
    
    with open('config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info("✅ Configuration file created: config.json")
    
    # Final summary
    logger.info("=" * 50)
    logger.info("🎉 Setup completed successfully!")
    logger.info("")
    logger.info("📁 Created files:")
    logger.info("  - data/medical_embeddings.npy (text embeddings)")
    logger.info("  - data/metadata.jsonl (document metadata)")
    logger.info("  - data/faiss_index.faiss (FAISS search index)")
    logger.info("  - data/formatted_qa_pairs.jsonl (training data)")
    logger.info("  - config.json (system configuration)")
    logger.info("")
    logger.info("🚀 Next steps:")
    logger.info("  1. Run: streamlit run streamlit_app.py")
    logger.info("  2. Or test individual components with the scripts in each module")
    logger.info("  3. For training: python finetune/lora_finetune.py --help")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
