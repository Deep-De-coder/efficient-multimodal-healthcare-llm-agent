import os
import csv
import json
import argparse
import logging

REQUIRED_COLUMNS = ['subject_id', 'study_id', 'report']

def validate_csv(input_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for col in REQUIRED_COLUMNS:
            if col not in reader.fieldnames:
                raise ValueError(f"Missing required column: {col}")
        return reader

def extract_and_save(input_path, output_path):
    reader = validate_csv(input_path)
    with open(output_path, 'w', encoding='utf-8') as out_f:
        for row in reader:
            clean_row = {col: row[col] for col in REQUIRED_COLUMNS}
            out_f.write(json.dumps(clean_row) + '\n')
    logging.info(f"Saved cleaned data to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Extract and validate MIMIC-CXR reports.')
    parser.add_argument('--input', required=True, help='Path to input CSV')
    parser.add_argument('--output', required=True, help='Path to output JSONL')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    extract_and_save(args.input, args.output)

if __name__ == '__main__':
    main() 