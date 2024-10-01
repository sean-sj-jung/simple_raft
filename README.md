Simple code for retrieval augmented fine tuning (RAFT)

This is my training pipeline for [RAFT: Adapting Language Model to Domain Specific RAG](https://arxiv.org/pdf/2403.10131)
It only takes pdf files as training input data.

create_dataset.py creates a Dataset object (*.hf)
finetune_hf.py fine-tunes a model

The code was tested for Phi-3.5-mini-instruct using one A100
