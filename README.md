# Efficient Multimodal Healthcare LLM Agent

[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/release/python-3100/)

---

## 📑 Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Data Flow](#data-flow)
- [RAG Pipeline](#rag-pipeline)
- [Illustrative Images](#illustrative-images)
- [Module Breakdown](#module-breakdown)
- [Setup & Installation](#setup--installation)
- [Training & Fine-tuning](#training--fine-tuning)
- [Inference & Deployment](#inference--deployment)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)
- [References](#references)

---

## 🚀 Project Overview

**Efficient Multimodal Healthcare LLM Agent** is a research-grade, production-ready clinical assistant that answers healthcare questions using both radiology text (MIMIC-CXR reports, PubMed abstracts) and X-ray images. It leverages a Retrieval-Augmented Generation (RAG) pipeline, LoRA fine-tuning, and state-of-the-art inference optimizations (FlashAttention, paged attention, DeepSpeed, PyTorch Lightning, vLLM) for scalable, high-throughput, and accurate clinical QA.

---

## ✨ Key Features
- **Multimodal RAG**: Combines text and image retrieval for context-rich answers
- **LoRA Fine-tuning**: Efficiently adapts LLaMA-2 7B to clinical QA
- **FlashAttention & Paged Attention**: Ultra-fast inference with vLLM
- **Distributed Training**: DeepSpeed & PyTorch Lightning support
- **Colab/Cloud Ready**: Optimized for T4/A100 GPUs
- **Streamlit UI**: Interactive demo interface
- **Extensible & Modular**: Clean, research-friendly codebase

---

## 🏛️ Architecture

```mermaid
flowchart TD
    subgraph DataIngestion
        A1["MIMIC-CXR CSV"] -->|extract_mimic_cxr.py| B1["Cleaned CXR JSONL"]
        A2["PubMed JSON"] -->|download_pubmed.py| B2["Cleaned PubMed JSONL"]
        A3["DICOM Images"] -->|dicom_to_jpeg.py| B3["JPEG Images"]
    end
    subgraph Preprocessing
        B1 -->|preprocess_cxr.py| C1["CXR Chunks"]
        B2 -->|preprocess_pubmed.py| C2["PubMed Chunks"]
        B3 -->|preprocess_images.py| C3["Processed JPEGs"]
    end
    subgraph Embedding
        C1 -->|embed_text.py| D1["Text Embeddings"]
        C2 -->|embed_text.py| D2["Text Embeddings"]
        C3 -->|embed_images.py| D3["Image Embeddings"]
    end
    subgraph Indexing
        D1 -->|faiss_index.py| E1["FAISS Index"]
        D2 -->|faiss_index.py| E1
        D3 -->|faiss_index.py| E1
    end
    subgraph RAG
        E1 -->|retriever.py| F1["Top-k Chunks"]
        F1 -->|prompt_builder.py| G1["Prompt"]
        G1 -->|generator.py| H1["LLM Answer"]
    end
    subgraph Finetune
        QA["QA Pairs"] -->|lora_finetune.py| LORA["LoRA Adapter"]
        LORA -->|generator.py| H1
    end
    subgraph Evaluation
        E1 -->|retrieval_metrics.py| M1["Recall@5"]
        H1 -->|qa_metrics.py| M2["Exact Match"]
        H1 -->|latency_metrics.py| M3["Latency/QPS"]
    end
    subgraph Deployment
        H1 -->|streamlit_app.py| UI["Streamlit UI"]
        E1 -->|drive_utils.py| GD["Google Drive"]
        LORA -->|drive_utils.py| GD
    end
```

---

## 🔄 Data Flow

```mermaid
flowchart LR
    subgraph DataFlow
        A["Raw Data"] --> B["Preprocessing"]
        B --> C["Chunked/Processed Data"]
        C --> D["Embedding"]
        D --> E["FAISS Index"]
        E --> F["RAG Pipeline"]
        F --> G["LLM Answer"]
    end
```

---

## 🧩 RAG Pipeline

```mermaid
flowchart TD
    Q["User Query"] --> E["Embed Query"]
    E --> R["Retrieve Top-k (FAISS)"]
    R --> C["Collect Context"]
    C --> P["Build Prompt"]
    P --> G["LLM (LoRA) Generate"]
    G --> A["Answer"]
```

---

## 🖼️ Illustrative Images

For a visually rich repo, add the following images to the `assets/` directory and reference them in this README:

| Filename                        | Description                                                      |
|----------------------------------|------------------------------------------------------------------|
| `assets/architecture.png`        | Full system architecture (exported from Mermaid or draw.io)      |
| `assets/data_flow.png`           | Data flow from ingestion to answer                               |
| `assets/rag_pipeline.png`        | RAG pipeline step-by-step illustration                           |
| `assets/streamlit_ui.png`        | Screenshot of the Streamlit demo UI                              |
| `assets/example_query.png`       | Example: user query, retrieved context, and generated answer     |
| `assets/colab_runtime.png`       | Colab runtime setup screenshot                                  |

> **Tip:** Use [draw.io](https://draw.io), [Excalidraw](https://excalidraw.com/), or export Mermaid diagrams as PNGs for maximum clarity.

---

## 🗂️ Module Breakdown

- **data_ingestion/**: Scripts for extracting, validating, and converting raw data (MIMIC-CXR, PubMed, DICOM)
- **preprocessing/**: Chunking, tokenization, and image preprocessing utilities
- **embedding/**: Text/image embedding and FAISS indexing
- **rag/**: Retrieval, prompt building, and LLM answer generation
- **finetune/**: LoRA fine-tuning (DeepSpeed, PyTorch Lightning), QA prep, configs
- **inference/**: vLLM runner for FlashAttention & paged attention inference
- **deployment/**: Colab setup, Streamlit UI, Google Drive integration
- **assets/**: Diagrams, screenshots, and illustrative images

---

## ⚙️ Setup & Installation

```bash
# Clone the repo
https://github.com/Deep-De-coder/efficient-multimodal-healthcare-llm-agent.git
cd efficient-multimodal-healthcare-llm-agent

# Install dependencies
pip install -r requirements.txt
```

---

## 🏋️ Training & Fine-tuning

### LoRA Fine-tuning (Standard, DeepSpeed, or Lightning)

**Standard:**
```bash
python finetune/lora_finetune.py \
  --data data/qa_pairs.jsonl \
  --model <llama-2-model-path-or-hf-name> \
  --adapter_output health_lora.pt \
  --epochs 3 \
  --batch_size 8
```

**DeepSpeed:**
```bash
python finetune/lora_finetune.py \
  --data data/qa_pairs.jsonl \
  --model <llama-2-model-path-or-hf-name> \
  --adapter_output health_lora.pt \
  --epochs 3 \
  --batch_size 8 \
  --deepspeed_config finetune/deepspeed_config.json
```

**PyTorch Lightning:**
```bash
python finetune/lora_finetune.py \
  --data data/qa_pairs.jsonl \
  --model <llama-2-model-path-or-hf-name> \
  --adapter_output health_lora.pt \
  --epochs 3 \
  --batch_size 8 \
  --use_lightning
```

---

## ⚡ Inference & Deployment

### Fast Inference with vLLM (FlashAttention & Paged Attention)
```bash
python inference/vllm_runner.py \
  --model <llama-2-model-path-or-hf-name> \
  --adapter health_lora.pt \
  --prompt_file prompts.txt \
  --output_file outputs.txt \
  --batch_size 8
```

### Streamlit Demo
```bash
streamlit run deployment/streamlit_app.py
```

### Colab/Cloud Setup
- See `deployment/colab_setup.ipynb` for a ready-to-run notebook

---

## 🤝 Contributing

Contributions are welcome! Please open issues or pull requests for improvements, bug fixes, or new features.

---

## 📖 Citation

If you use this project in your research, please cite as:

```bibtex
@software{deep_decoder_multimodal_llm_2024,
  author = {Deep-De-coder},
  title = {Efficient Multimodal Healthcare LLM Agent},
  year = {2024},
  url = {https://github.com/Deep-De-coder/efficient-multimodal-healthcare-llm-agent}
}
```

---

## 📝 License

This project is licensed under the [MIT License](LICENSE).

---

## 🔗 References
- [MIMIC-CXR Dataset](https://physionet.org/content/mimic-cxr/2.0.0/)
- [PubMed](https://pubmed.ncbi.nlm.nih.gov/)
- [LLaMA-2](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)
- [LoRA: Low-Rank Adaptation](https://arxiv.org/abs/2106.09685)
- [vLLM](https://github.com/vllm-project/vllm)
- [DeepSpeed](https://www.deepspeed.ai/)
- [PyTorch Lightning](https://www.pytorchlightning.ai/)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Streamlit](https://streamlit.io/)

---

> **Author:** Deep-De-coder 