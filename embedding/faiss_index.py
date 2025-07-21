import os
import argparse
import logging
import numpy as np
import faiss

def build_faiss_index(emb_path, index_path, index_type='FlatL2'):
    embs = np.load(emb_path)
    dim = embs.shape[1]
    if index_type == 'FlatL2':
        index = faiss.IndexFlatL2(dim)
    elif index_type == 'IVF':
        nlist = 100
        quantizer = faiss.IndexFlatL2(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist)
        index.train(embs)
    else:
        raise ValueError(f'Unknown index type: {index_type}')
    index.add(embs)
    faiss.write_index(index, index_path)
    logging.info(f"Saved FAISS index to {index_path}")

def main():
    parser = argparse.ArgumentParser(description='Build a FAISS index from embeddings.')
    parser.add_argument('--emb', required=True, help='Input .npy embeddings file')
    parser.add_argument('--output', required=True, help='Output FAISS index file')
    parser.add_argument('--index_type', default='FlatL2', choices=['FlatL2', 'IVF'], help='Type of FAISS index')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    build_faiss_index(args.emb, args.output, args.index_type)

if __name__ == '__main__':
    main() 