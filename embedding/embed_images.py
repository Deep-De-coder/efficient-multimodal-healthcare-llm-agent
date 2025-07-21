import os
import argparse
import logging
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch
import json
from transformers import CLIPProcessor, CLIPModel

def embed_images(input_dir, output_emb_path, output_meta_path, model_name='openai/clip-vit-base-patch16', batch_size=32, device='cpu'):
    processor = CLIPProcessor.from_pretrained(model_name)
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.jpg')]
    all_embs = []
    metas = []
    with torch.no_grad():
        for i in tqdm(range(0, len(image_files), batch_size)):
            batch_files = image_files[i:i+batch_size]
            images = [Image.open(os.path.join(input_dir, f)).convert('RGB') for f in batch_files]
            inputs = processor(images=images, return_tensors='pt', padding=True).to(device)
            outputs = model.get_image_features(**inputs)
            emb = outputs.cpu().numpy()
            all_embs.append(emb)
            for fname in batch_files:
                metas.append({'filename': fname})
    all_embs = np.concatenate(all_embs, axis=0)
    np.save(output_emb_path, all_embs)
    with open(output_meta_path, 'w', encoding='utf-8') as f:
        for meta in metas:
            f.write(json.dumps(meta) + '\n')
    logging.info(f"Saved image embeddings to {output_emb_path} and metadata to {output_meta_path}")

def main():
    parser = argparse.ArgumentParser(description='Embed images using CLIP.')
    parser.add_argument('--input_dir', required=True, help='Input directory with JPEGs')
    parser.add_argument('--output_emb', required=True, help='Output .npy file for embeddings')
    parser.add_argument('--output_meta', required=True, help='Output JSONL for metadata')
    parser.add_argument('--model', default='openai/clip-vit-base-patch16', help='CLIP model name')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', default='cpu', help='Device (cpu or cuda)')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    embed_images(args.input_dir, args.output_emb, args.output_meta, args.model, args.batch_size, args.device)

if __name__ == '__main__':
    main() 