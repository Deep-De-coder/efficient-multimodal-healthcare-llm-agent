import argparse
import json
import logging

def build_prompt(query, context_path, output_path):
    with open(context_path, 'r', encoding='utf-8') as f:
        context_chunks = [json.loads(line)['text'] for line in f]
    prompt = query + '\n\n' + '\n'.join(context_chunks)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(prompt)
    logging.info(f"Prompt saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Build prompt from query and context.')
    parser.add_argument('--query', required=True, help='User query')
    parser.add_argument('--context', required=True, help='Context JSONL file')
    parser.add_argument('--output', required=True, help='Output prompt file')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    build_prompt(args.query, args.context, args.output)

if __name__ == '__main__':
    main() 