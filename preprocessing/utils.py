import os
from typing import List
from transformers import AutoTokenizer

# Load a HuggingFace tokenizer (default: bert-base-uncased)
def get_tokenizer(model_name: str = 'bert-base-uncased'):
    return AutoTokenizer.from_pretrained(model_name)

# Chunk text into segments of max_tokens length
def chunk_text(text: str, tokenizer, max_tokens: int = 512) -> List[str]:
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = [tokens[i:i+max_tokens] for i in range(0, len(tokens), max_tokens)]
    return [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in chunks]

# Detokenize a list of token ids back to text
def detokenize(tokens: List[int], tokenizer) -> str:
    return tokenizer.decode(tokens, skip_special_tokens=True) 