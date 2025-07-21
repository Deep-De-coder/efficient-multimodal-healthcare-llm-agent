import argparse
import numpy as np
import faiss
import logging

def retrieve(index_path, query_emb_path, k=5):
    index = faiss.read_index(index_path)
    query_emb = np.load(query_emb_path)
    if len(query_emb.shape) == 1:
        query_emb = query_emb[None, :]
    D, I = index.search(query_emb, k)
    return D, I

def main():
    parser = argparse.ArgumentParser(description='Retrieve top-k results from FAISS index.')
    parser.add_argument('--index', required=True, help='Path to FAISS index')
    parser.add_argument('--query_emb', required=True, help='Path to query embedding (.npy)')
    parser.add_argument('--k', type=int, default=5, help='Number of results to retrieve')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    D, I = retrieve(args.index, args.query_emb, args.k)
    print('Distances:', D)
    print('Indices:', I)

if __name__ == '__main__':
    main() 