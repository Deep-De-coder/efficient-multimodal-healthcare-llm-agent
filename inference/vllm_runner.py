import argparse
import logging
from vllm import LLM, SamplingParams

# If using LoRA, vllm supports loading adapters via the 'lora' argument

def run_vllm_inference(model_name, adapter_path, prompt_file, batch_size, output_file, max_tokens=256, dtype="auto"):
    # Read prompts (one per line)
    with open(prompt_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    # vllm LLM setup
    llm = LLM(
        model=model_name,
        lora=adapter_path,
        dtype=dtype,  # "auto" lets vllm pick best for GPU
        max_num_batched_tokens=4096,  # adjust for GPU memory
        tensor_parallel_size=1,  # set >1 for multi-GPU
    )
    sampling_params = SamplingParams(max_tokens=max_tokens)
    # Batched inference
    outputs = llm.generate(prompts, sampling_params)
    with open(output_file, 'w', encoding='utf-8') as f:
        for output in outputs:
            f.write(output.outputs[0].text.strip() + '\n')
    logging.info(f"Saved vllm outputs to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='vLLM inference with FlashAttention and paged attention.')
    parser.add_argument('--model', required=True, help='Base model name or path (e.g., LLaMA-2)')
    parser.add_argument('--adapter', required=True, help='LoRA adapter path')
    parser.add_argument('--prompt_file', required=True, help='File with prompts (one per line)')
    parser.add_argument('--output_file', required=True, help='File to save outputs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for inference')
    parser.add_argument('--max_tokens', type=int, default=256, help='Max tokens to generate')
    parser.add_argument('--dtype', default='auto', help='Data type (auto, float16, bfloat16, float32)')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run_vllm_inference(args.model, args.adapter, args.prompt_file, args.batch_size, args.output_file, args.max_tokens, args.dtype)

if __name__ == '__main__':
    main() 