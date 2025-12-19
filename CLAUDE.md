# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MiniMind is a lightweight language model implementation designed for educational purposes and research. The project implements a transformer-based architecture with support for various training techniques including full fine-tuning, LoRA, DPO, PPO, and other RLHF methods.

## Architecture Structure

### Core Components
- `model/model_minimind.py`: Main model implementation with configuration, attention mechanisms, feed-forward networks, and MoE support
- `trainer/`: Contains various training scripts for different methodologies (SFT, DPO, PPO, etc.)
- `dataset/lm_dataset.py`: Dataset implementations for pretraining, SFT, DPO, and RLHF training
- `scripts/`: Utility scripts for evaluation, deployment, and demos

### Key Features
- Transformer architecture with rotary positional embeddings (RoPE)
- Mixture of Experts (MoE) support
- Flash attention optimization
- Multiple training paradigms (pretraining, SFT, DPO, PPO, GRPO, SPO)
- LoRA fine-tuning support
- Streamlit web demo
- OpenAI-compatible API server

## Common Development Commands

### Environment Setup
```bash
pip install -r requirements.txt
```

### Training Models
```bash
# Pretraining
python trainer/train_pretrain.py --epochs 1 --batch_size 32 --learning_rate 5e-4

# Full SFT training
python trainer/train_full_sft.py --epochs 3 --batch_size 16 --learning_rate 2e-5

# LoRA fine-tuning
python trainer/train_lora.py --epochs 3 --batch_size 16 --learning_rate 2e-4

# DPO training
python trainer/train_dpo.py --epochs 2 --batch_size 8 --learning_rate 5e-6
```

### Evaluating Models
```bash
# Interactive evaluation
python eval_llm.py --weight full_sft --hidden_size 512

# Batch evaluation with specific prompts
python eval_llm.py --weight full_sft --hidden_size 512 --prompts_file test_prompts.jsonl
```

### Running Demos
```bash
# Web demo
streamlit run scripts/web_demo.py

# API server
python scripts/serve_openai_api.py --weight full_sft --hidden_size 512
```

### Testing
```bash
# Run unit tests (if available)
python -m pytest tests/

# Test model loading
python -c "from model.model_minimind import MiniMindForCausalLM; model = MiniMindForCausalLM()"
```

## Model Variants

1. **Standard Model**: Basic transformer with 8 layers, 512 hidden dimensions (~26M parameters)
2. **MoE Model**: Mixture of Experts variant with routing mechanism (~145M parameters)
3. **Base Model**: Larger variant with 16 layers, 768 hidden dimensions (~104M parameters)

## Key Implementation Details

### Model Configuration
- Hidden sizes: 512 (small), 640 (MoE), 768 (base)
- Number of layers: 8 (small/MoE), 16 (base)
- Attention heads: Configurable per model size
- Context length: Up to 32,768 tokens with RoPE scaling

### Training Infrastructure
- Mixed precision training support (bfloat16/float16)
- Gradient accumulation and clipping
- Learning rate scheduling with cosine decay
- Distributed training support (DDP)
- Checkpointing with resume capability
- Wandb/SwanLab logging support

### Data Processing
- JSONL format for training data
- Dynamic loss masking for instruction tuning
- Tokenizer-aware preprocessing
- Support for chat templates and system prompts

## Repository Layout
```
minimind/
├── model/              # Model definitions and configurations
├── trainer/            # Training scripts for various methods
├── dataset/            # Dataset processing and loading
├── scripts/            # Utility scripts and demos
├── out/                # Default output directory for checkpoints
└── checkpoints/        # Training checkpoint storage
```