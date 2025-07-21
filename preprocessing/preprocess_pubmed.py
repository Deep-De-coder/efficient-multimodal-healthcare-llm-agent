import os
import argparse
import logging
import subprocess

def main():
    parser = argparse.ArgumentParser(description='Preprocess PubMed abstracts into chunks.')
    parser.add_argument('--input', required=True, help='Input cleaned PubMed JSONL')
    parser.add_argument('--output', required=True, help='Output chunked JSONL')
    parser.add_argument('--tokenizer', default='bert-base-uncased', help='Tokenizer model name')
    parser.add_argument('--max_tokens', type=int, default=512, help='Max tokens per chunk')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    cmd = [
        'python', os.path.join(os.path.dirname(__file__), 'chunk_text.py'),
        '--input', args.input,
        '--output', args.output,
        '--tokenizer', args.tokenizer,
        '--max_tokens', str(args.max_tokens)
    ]
    logging.info(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    main() 