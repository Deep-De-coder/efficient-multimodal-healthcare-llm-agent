import os
import json
import argparse
import logging
from preprocessing.utils import get_tokenizer, chunk_text

def chunk_dataset(input_path, output_path, tokenizer_model, max_tokens=512):
    tokenizer = get_tokenizer(tokenizer_model)
    with open(input_path, 'r', encoding='utf-8') as in_f, open(output_path, 'w', encoding='utf-8') as out_f:
        for i, line in enumerate(in_f):
            entry = json.loads(line)
            text = entry['text'] if 'text' in entry else entry.get('abstract', '')
            chunks = chunk_text(text, tokenizer, max_tokens=max_tokens)
            for idx, chunk in enumerate(chunks):
                out = dict(entry)
                out['chunk_id'] = f"{entry.get('id', i)}_{idx}"
                out['text'] = chunk
                out_f.write(json.dumps(out) + '\n')
    logging.info(f"Chunked dataset saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Chunk a text dataset into token segments.')
    parser.add_argument('--input', required=True, help='Input JSONL file')
    parser.add_argument('--output', required=True, help='Output JSONL file')
    parser.add_argument('--tokenizer', default='bert-base-uncased', help='Tokenizer model name')
    parser.add_argument('--max_tokens', type=int, default=512, help='Max tokens per chunk')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    chunk_dataset(args.input, args.output, args.tokenizer, args.max_tokens)

if __name__ == '__main__':
    main() 