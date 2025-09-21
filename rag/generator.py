import argparse
import logging
import torch
import json
import os
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    pipeline
)
from peft import PeftModel
import warnings
warnings.filterwarnings("ignore")

def load_model_and_tokenizer(model_name, adapter_path=None, device='auto'):
    """Load base model and tokenizer, optionally with LoRA adapter."""
    logging.info(f"Loading model: {model_name}")
    
    # Configure quantization for memory efficiency (optional)
    bnb_config = None
    try:
        if torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
    except Exception as e:
        logging.warning(f"Could not configure quantization: {e}")
        bnb_config = None
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Load base model
        if device == 'auto':
            device_map = "auto"
        else:
            device_map = None
            
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map=device_map,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
        
        # Load LoRA adapter if provided
        if adapter_path and os.path.exists(adapter_path):
            logging.info(f"Loading LoRA adapter from: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
            
        model.eval()
        return model, tokenizer
        
    except Exception as e:
        logging.warning(f"Failed to load {model_name}: {e}")
        # Fallback to a smaller model
        fallback_model = "microsoft/DialoGPT-medium"
        logging.info(f"Using fallback model: {fallback_model}")
        tokenizer = AutoTokenizer.from_pretrained(fallback_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(fallback_model)
        model.eval()
        return model, tokenizer

def generate_answer(prompt_path, model_name, adapter_path, output_path, max_length=512, temperature=0.7):
    """Generate answer using the loaded model."""
    try:
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(model_name, adapter_path)
        
        # Read prompt
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt = f.read().strip()
            
        logging.info(f"Generating answer for prompt: {prompt[:100]}...")
        
        # Create a healthcare-focused system prompt
        system_prompt = """You are a helpful medical AI assistant. Based on the provided context, answer the medical question accurately and professionally. If you cannot provide a definitive answer, explain what information would be needed for a proper diagnosis.

Context:
"""
        
        full_prompt = system_prompt + prompt + "\n\nAnswer:"
        
        # Tokenize input
        inputs = tokenizer.encode(full_prompt, return_tensors='pt', truncation=True, max_length=1024)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=len(inputs[0]) + max_length,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                top_k=50,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=3
            )
        
        # Decode response
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the answer part (after "Answer:")
        if "Answer:" in full_response:
            answer = full_response.split("Answer:")[-1].strip()
        else:
            answer = full_response[len(tokenizer.decode(inputs[0], skip_special_tokens=True)):].strip()
            
        # Clean up the answer
        if answer.startswith(full_prompt):
            answer = answer[len(full_prompt):].strip()
            
        # Ensure we have a meaningful answer
        if not answer or len(answer) < 10:
            answer = "I need more specific information to provide a comprehensive medical answer. Please consult with a healthcare professional for personalized medical advice."
            
        # Save answer
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(answer)
            
        logging.info(f"Answer saved to {output_path}")
        logging.info(f"Generated answer: {answer[:200]}...")
        
    except Exception as e:
        logging.error(f"Error generating answer: {e}")
        # Fallback response
        fallback_answer = f"Error generating response: {str(e)}. Please check the model configuration and try again."
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(fallback_answer)
        logging.info(f"Fallback answer saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate answer using LoRA-adapted LLaMA.')
    parser.add_argument('--prompt', required=True, help='Prompt file')
    parser.add_argument('--model', required=True, help='Model name (e.g., microsoft/DialoGPT-medium)')
    parser.add_argument('--adapter', default=None, help='LoRA adapter path (optional)')
    parser.add_argument('--output', required=True, help='Output answer file')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum response length')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    generate_answer(
        args.prompt, 
        args.model, 
        args.adapter, 
        args.output,
        args.max_length,
        args.temperature
    )

if __name__ == '__main__':
    main() 