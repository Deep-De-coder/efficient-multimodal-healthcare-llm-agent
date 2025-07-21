import os
import json
import argparse
import logging

REQUIRED_FIELDS = ['pmid', 'title', 'abstract']

def validate_json(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for entry in data:
            for field in REQUIRED_FIELDS:
                if field not in entry:
                    raise ValueError(f"Missing required field: {field}")
        return data

def extract_and_save(input_path, output_path):
    data = validate_json(input_path)
    with open(output_path, 'w', encoding='utf-8') as out_f:
        for entry in data:
            clean_entry = {field: entry[field] for field in REQUIRED_FIELDS}
            out_f.write(json.dumps(clean_entry) + '\n')
    logging.info(f"Saved cleaned data to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Extract and validate PubMed abstracts.')
    parser.add_argument('--input', required=True, help='Path to input JSON')
    parser.add_argument('--output', required=True, help='Path to output JSONL')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    extract_and_save(args.input, args.output)

if __name__ == '__main__':
    main() 