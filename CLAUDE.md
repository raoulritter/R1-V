# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

R1-V is a framework for Reinforcing Vision Language Models (VLMs) using Generalized Reinforcement Learning Policy Optimization (GRPO). The project focuses on improving visual reasoning capabilities in VLMs through reinforcement learning.

## Environment Setup

```bash
# Create and activate a conda environment
conda create -n r1-v python=3.11
conda activate r1-v

# Run the setup script
bash setup.sh
```

The setup script installs necessary packages including:
- Required PyTorch libraries
- Flash attention
- vLLM support
- wandb and tensorboardx for tracking

## Key Commands

### Training VLMs with GRPO

```bash
# Run GRPO training on multiple GPUs
export DEBUG_MODE="true"  # Enable debug logs
export LOG_PATH="./debug_log_2b.txt"  # Log file path

torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/grpo.py \
    --output_dir <OUTPUT_DIR> \
    --model_name_or_path <PATH-TO-MODEL> \
    --dataset_name <DATASET_NAME> \
    --max_prompt_length 512 \
    --max_completion_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 2 \
    --run_name <RUN_NAME> \
    --save_steps 100 \
    --save_only_model true \
    --num_generations 8  # Number of outputs in GRPO
```

### Training with vLLM (Faster)

```bash
export DEBUG_MODE="true"
export LOG_PATH="./vllm_run.txt"

# Use X+1 GPUs (X for training, 1 for vLLM)
CUDA_VISIBLE_DEVICES="0,1,2,3,4" torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/grpo.py --use_vllm True \
    --output_dir <OUTPUT_DIR> \
    --model_name_or_path <MODEL_PATH> \
    --dataset_name <DATASET_NAME> \
    --max_prompt_length 512 \
    --max_completion_length 1024 \
    --temperature 1.0 \
    --num_generations 4 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 400000 \
    --max_steps 13125 \
    --run_name <RUN_NAME> \
    --save_steps 1000 \
    --save_only_model true
```

### Supervised Fine-Tuning (SFT)

```bash
accelerate launch --config_file src/r1-v/configs/zero2.yaml \
    src/r1-v/src/open_r1/sft.py \
    --config src/r1-v/configs/qwen2vl_sft_config.yaml
```

### Evaluation

#### SuperCLEVR Counting Task Evaluation
```bash
cd ./src/eval
python test_qwen2vl_counting_superclevr.py
```

#### GeoQA Evaluation
```bash
cd ./src/eval
python test_qwen2vl_geoqa.py

# For multi-GPU evaluation
bash src/scripts/test_grpo_geoqa_multigpu.sh
```

## Architecture Overview

1. **GRPO Trainer (`grpo_trainer.py`)**: 
   - Core implementation of the GRPO algorithm for vision-language models
   - Supports models like Qwen2-VL, Qwen2.5-VL, Aria, and Janus
   - Handles pixel processing, reward computation, and KL divergence constraints

2. **Main GRPO Script (`grpo.py`)**:
   - Entry point for training with reinforcement learning
   - Implements reward functions (accuracy and format rewards)
   - Formats prompts and manages conversation templates

3. **Evaluation Scripts**:
   - Test scripts for counting tasks (`test_qwen2vl_counting_superclevr.py`)
   - Test scripts for geometry reasoning (`test_qwen2vl_geoqa.py`)

## Supported Models

- Qwen2-VL
- Qwen2.5-VL
- Janus (recently added)

## Datasets

1. CLEVR-70k-Counting: Item counting problems
2. CLEVR-70k-Complex: Number-related reasoning
3. GEOQA-8k: Geometry reasoning
4. Test sets: SuperCLEVR-200 and GeoQA-Test-Direct-Answer-735

## Important Notes

1. Batch size should be kept at 1 for stable training (due to known issues)
2. Adjust `--num_generations` to manage GPU memory consumption
3. vLLM support requires `vllm==0.7.2`
4. For reproducible results, use the exact configurations provided in scripts
5. Format reward encourages models to follow structured output: `<think>reasoning</think><answer>answer</answer>`