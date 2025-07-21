import argparse
import json
import logging

def prepare_qa_pairs(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Assume data is a list of {question, answer}
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, qa in enumerate(data):
            out = {
                'id': i,
                'instruction': qa['question'],
                'output': qa['answer']
            }
            f.write(json.dumps(out) + '\n')
    logging.info(f"Saved formatted QA pairs to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Prepare QA pairs for LoRA fine-tuning.')
    parser.add_argument('--input', required=True, help='Input raw QA JSON')
    parser.add_argument('--output', required=True, help='Output formatted JSONL')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    prepare_qa_pairs(args.input, args.output)

if __name__ == '__main__':
    main() 