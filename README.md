# Retrieval-Augmented Fine-Tuning (RAFT)

This repository contains code for a simple training pipeline for **Retrieval-Augmented Fine-Tuning (RAFT)**, based on the approach described in the paper [RAFT: Adapting Language Models to Domain-Specific Retrieval-Augmented Generation (RAG)](https://arxiv.org/pdf/2403.10131).

## Getting Started

### Preparing Documents
- Place your documents in the `./documents` directory.  
  - Note: Only PDF files are accepted as input for training.

### Running the Pipeline
- `create_dataset.py`: Generates a Hugging Face Dataset object (`*.hf` format) from the provided documents.
  - The Dataset object will be saved to disk.
- `finetune_hf.py`: Fine-tunes a pre-trained model using the created dataset.

### Tested Configuration
- Model: **Phi-3.5-mini-instruct**
- Hardware: **NVIDIA A100 GPU**
