import argparse
import json
import logging

def prepare_qa_pairs(input_path, output_path):
    # Handle both JSON and JSONL formats
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        
    # Try to parse as JSON first (single object or array)
    try:
        data = json.loads(content)
        if not isinstance(data, list):
            data = [data]
    except json.JSONDecodeError:
        # If JSON parsing fails, treat as JSONL
        data = []
        for line in content.split('\n'):
            if line.strip():
                data.append(json.loads(line))
    
    # Format the data
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, qa in enumerate(data):
            # Handle different possible field names
            instruction = qa.get('instruction', qa.get('question', ''))
            output = qa.get('output', qa.get('answer', ''))
            
            out = {
                'id': i,
                'instruction': instruction,
                'output': output
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