import os
import csv
import json
from preprocessing.utils import get_tokenizer, chunk_text

INPUT_CSV = os.path.join('data', 'mimic_cxr_reports.csv')
OUTPUT_JSONL = os.path.join('data', 'cxr_chunks.jsonl')
TOKENIZER_MODEL = 'bert-base-uncased'  # Change to LLaMA if available


def process_reports():
    tokenizer = get_tokenizer(TOKENIZER_MODEL)
    with open(INPUT_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        with open(OUTPUT_JSONL, 'w', encoding='utf-8') as out_f:
            for row in reader:
                report = row['report']
                subject_id = row['subject_id']
                study_id = row['study_id']
                chunks = chunk_text(report, tokenizer, max_tokens=512)
                for idx, chunk in enumerate(chunks):
                    out = {
                        'source': 'mimic_cxr',
                        'subject_id': subject_id,
                        'study_id': study_id,
                        'chunk_id': f"{subject_id}_{study_id}_{idx}",
                        'text': chunk
                    }
                    out_f.write(json.dumps(out) + '\n')

if __name__ == '__main__':
    process_reports() 