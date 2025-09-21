import argparse
import logging
import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments, 
    Trainer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
    PeftModel
)
from datasets import Dataset as HFDataset
import warnings
warnings.filterwarnings("ignore")

# Optionally import DeepSpeed and PyTorch Lightning
try:
    import deepspeed
except ImportError:
    deepspeed = None
try:
    import pytorch_lightning as pl
    from pytorch_lightning import Trainer as PLTrainer
except ImportError:
    pl = None

class HealthcareQADataset(Dataset):
    """Dataset for healthcare QA pairs."""
    
    def __init__(self, data_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        # Load data
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        self.data.append(item)
                    except json.JSONDecodeError:
                        continue
        
        logging.info(f"Loaded {len(self.data)} QA pairs")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Format the instruction and response
        instruction = item.get('instruction', item.get('question', ''))
        output = item.get('output', item.get('answer', ''))
        
        # Create prompt
        prompt = f"### Instruction:\n{instruction}\n\n### Response:\n{output}"
        
        # Tokenize
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }

def create_lora_config():
    """Create LoRA configuration."""
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,  # Rank
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )

def load_model_for_training(model_name, use_quantization=True):
    """Load model for training with optional quantization."""
    logging.info(f"Loading model: {model_name}")
    
    # Configure quantization for memory efficiency
    bnb_config = None
    if use_quantization and torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
        
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
        return model, tokenizer

def lora_finetune(data_path, model_name, adapter_output, epochs, batch_size, 
                  deepspeed_config=None, use_lightning=False, learning_rate=5e-4):
    """Perform LoRA fine-tuning."""
    
    # Load model and tokenizer
    model, tokenizer = load_model_for_training(model_name)
    
    # Create LoRA config
    lora_config = create_lora_config()
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Create dataset
    dataset = HealthcareQADataset(data_path, tokenizer)
    
    # Create training arguments
    training_args = TrainingArguments(
        output_dir="./lora_output",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,
        warmup_steps=100,
        learning_rate=learning_rate,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        report_to=None,  # Disable wandb/tensorboard
        dataloader_pin_memory=False,
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    
    # Start training
    logging.info("Starting LoRA fine-tuning...")
    trainer.train()
    
    # Save the adapter
    model.save_pretrained(adapter_output)
    tokenizer.save_pretrained(adapter_output)
    logging.info(f"LoRA adapter saved to {adapter_output}")

def main():
    parser = argparse.ArgumentParser(description='LoRA fine-tuning for healthcare LLM.')
    parser.add_argument('--data', required=True, help='QA pairs JSONL')
    parser.add_argument('--model', required=True, help='Base model name')
    parser.add_argument('--adapter_output', required=True, help='Output adapter directory')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size (reduced for memory)')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--deepspeed_config', default=None, help='Path to DeepSpeed config JSON (optional)')
    parser.add_argument('--use_lightning', action='store_true', help='Use PyTorch Lightning for training')
    parser.add_argument('--no_quantization', action='store_true', help='Disable quantization')
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Validate data file exists
    if not os.path.exists(args.data):
        logging.error(f"Data file not found: {args.data}")
        return
    
    # Create output directory
    os.makedirs(args.adapter_output, exist_ok=True)
    
    lora_finetune(
        args.data, 
        args.model, 
        args.adapter_output, 
        args.epochs, 
        args.batch_size,
        deepspeed_config=args.deepspeed_config, 
        use_lightning=args.use_lightning,
        learning_rate=args.learning_rate
    )

if __name__ == '__main__':
    main() 