import os
import json
import argparse
import logging
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import torch

def embed_texts(input_path, output_emb_path, output_meta_path, model_name, batch_size=32, device='cpu'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    texts = []
    metas = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            texts.append(entry['text'])
            metas.append({k: v for k, v in entry.items() if k != 'text'})
    all_embs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i+batch_size]
            inputs = tokenizer(batch, padding=True, truncation=True, return_tensors='pt').to(device)
            outputs = model(**inputs)
            if hasattr(outputs, 'last_hidden_state'):
                emb = outputs.last_hidden_state[:,0,:].cpu().numpy()
            else:
                emb = outputs[0][:,0,:].cpu().numpy()
            all_embs.append(emb)
    all_embs = np.concatenate(all_embs, axis=0)
    np.save(output_emb_path, all_embs)
    with open(output_meta_path, 'w', encoding='utf-8') as f:
        for meta in metas:
            f.write(json.dumps(meta) + '\n')
    logging.info(f"Saved embeddings to {output_emb_path} and metadata to {output_meta_path}")

def main():
    parser = argparse.ArgumentParser(description='Embed text chunks using a HuggingFace model.')
    parser.add_argument('--input', required=True, help='Input chunked JSONL')
    parser.add_argument('--output_emb', required=True, help='Output .npy file for embeddings')
    parser.add_argument('--output_meta', required=True, help='Output JSONL for metadata')
    parser.add_argument('--model', default='bert-base-uncased', help='Model name')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', default='cpu', help='Device (cpu or cuda)')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    embed_texts(args.input, args.output_emb, args.output_meta, args.model, args.batch_size, args.device)

if __name__ == '__main__':
    main() 