import argparse
import logging
import subprocess

def main():
    parser = argparse.ArgumentParser(description='Run the full RAG pipeline.')
    parser.add_argument('--index', required=True, help='FAISS index path')
    parser.add_argument('--query_emb', required=True, help='Query embedding .npy')
    parser.add_argument('--context', required=True, help='Context JSONL file')
    parser.add_argument('--prompt', required=True, help='Prompt file')
    parser.add_argument('--model', required=True, help='Model name')
    parser.add_argument('--adapter', required=True, help='LoRA adapter path')
    parser.add_argument('--output', required=True, help='Output answer file')
    parser.add_argument('--k', type=int, default=5, help='Top-k retrieval')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    # Stub: Replace with actual subprocess calls
    logging.info('Running retriever...')
    # subprocess.run([...])
    logging.info('Running prompt builder...')
    # subprocess.run([...])
    logging.info('Running generator...')
    # subprocess.run([...])
    logging.info('Pipeline complete.')

if __name__ == '__main__':
    main() 