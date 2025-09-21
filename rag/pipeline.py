import argparse
import logging
import subprocess
import os
import json
import tempfile
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

def embed_query(query_text, model_name='bert-base-uncased', output_path=None):
    """Embed a query text using the specified model."""
    logging.info(f"Embedding query: {query_text[:100]}...")
    
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
    
    if output_path:
        np.save(output_path, embedding)
        logging.info(f"Query embedding saved to {output_path}")
    
    return embedding

def retrieve_context(index_path, query_emb_path, metadata_path, k=5):
    """Retrieve relevant context using FAISS index."""
    logging.info(f"Retrieving top-{k} relevant documents...")
    
    try:
        import faiss
        # Load FAISS index
        index = faiss.read_index(index_path)
        query_emb = np.load(query_emb_path)
        
        if len(query_emb.shape) == 1:
            query_emb = query_emb[None, :]
        
        # Search for similar documents
        D, I = index.search(query_emb, k)
        
        # Load metadata to get actual text
        retrieved_docs = []
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata_lines = f.readlines()
        
        for idx in I[0]:  # I[0] contains the indices for the first query
            if idx < len(metadata_lines):
                try:
                    doc_meta = json.loads(metadata_lines[idx])
                    retrieved_docs.append(doc_meta)
                except json.JSONDecodeError:
                    continue
        
        logging.info(f"Retrieved {len(retrieved_docs)} documents")
        return retrieved_docs
        
    except Exception as e:
        logging.error(f"Error during retrieval: {e}")
        return []

def build_prompt_from_context(query, retrieved_docs):
    """Build a prompt with the query and retrieved context."""
    context_text = ""
    for i, doc in enumerate(retrieved_docs, 1):
        text = doc.get('text', doc.get('content', ''))
        context_text += f"Document {i}:\n{text}\n\n"
    
    prompt = f"""Question: {query}

Relevant Medical Information:
{context_text}

Please provide a comprehensive answer based on the above medical information. If the information is insufficient, please indicate what additional information would be needed."""
    
    return prompt

def run_rag_pipeline(query, index_path, metadata_path, model_name, adapter_path, output_path, k=5):
    """Run the complete RAG pipeline."""
    logging.info("Starting RAG pipeline...")
    
    # Step 1: Embed the query
    with tempfile.NamedTemporaryFile(suffix='.npy', delete=False) as tmp_file:
        query_emb_path = tmp_file.name
    
    try:
        query_embedding = embed_query(query, output_path=query_emb_path)
        
        # Step 2: Retrieve relevant context
        retrieved_docs = retrieve_context(index_path, query_emb_path, metadata_path, k)
        
        # Step 3: Build prompt with context
        prompt = build_prompt_from_context(query, retrieved_docs)
        
        # Save prompt to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as tmp_file:
            tmp_file.write(prompt)
            prompt_path = tmp_file.name
        
        try:
            # Step 4: Generate answer
            logging.info("Generating answer with LLM...")
            subprocess.run([
                'python', 'rag/generator.py',
                '--prompt', prompt_path,
                '--model', model_name,
                '--adapter', adapter_path if adapter_path and os.path.exists(adapter_path) else '',
                '--output', output_path
            ], check=True)
            
            logging.info("RAG pipeline completed successfully!")
            
        finally:
            # Clean up temporary prompt file
            if os.path.exists(prompt_path):
                os.unlink(prompt_path)
                
    finally:
        # Clean up temporary embedding file
        if os.path.exists(query_emb_path):
            os.unlink(query_emb_path)

def main():
    parser = argparse.ArgumentParser(description='Run the full RAG pipeline.')
    parser.add_argument('--query', required=True, help='User query text')
    parser.add_argument('--index', required=True, help='FAISS index path')
    parser.add_argument('--metadata', required=True, help='Metadata JSONL file')
    parser.add_argument('--model', required=True, help='Model name')
    parser.add_argument('--adapter', default=None, help='LoRA adapter path (optional)')
    parser.add_argument('--output', required=True, help='Output answer file')
    parser.add_argument('--k', type=int, default=5, help='Top-k retrieval')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Validate required files exist
    if not os.path.exists(args.index):
        logging.error(f"FAISS index not found: {args.index}")
        return
    
    if not os.path.exists(args.metadata):
        logging.error(f"Metadata file not found: {args.metadata}")
        return
    
    # Run the pipeline
    run_rag_pipeline(
        query=args.query,
        index_path=args.index,
        metadata_path=args.metadata,
        model_name=args.model,
        adapter_path=args.adapter,
        output_path=args.output,
        k=args.k
    )

if __name__ == '__main__':
    main() 