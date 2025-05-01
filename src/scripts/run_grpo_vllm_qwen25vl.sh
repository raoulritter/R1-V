#!/bin/bash

# Removed vllm dependencies comments

# The latest transformers is required too, install by: pip install git+https://github.com/huggingface/transformers.git@a40f1ac602fe900281722254c52ce3773f28eb0e

export WANDB_ENTITY="mae-testing" # Added from test-grpo-qwen.job
export WANDB_PROJECT="grpo-testing" # Added from test-grpo-qwen.job

export DEBUG_MODE="true"
export LOG_PATH="./debug_log_2b.txt" # Updated log path

MODEL_PATH="Qwen/Qwen2-VL-2B-Instruct" # Updated model path
HF_DATASET="leonardPKU/clevr_cogen_a_train" # Updated dataset
OUTPUT_DIR="./share_models/Qwen2-VL-2B-Instruct_GRPO_CLEVR-70k" # Updated output dir
if [ ! -d "$OUTPUT_DIR" ]; then
 mkdir -p "$OUTPUT_DIR"
fi
RUN_NAME="Qwen2-VL-2B-GRPO-CLEVR-70k" # Updated run name
DS_CONFIG="src/r1-v/local_scripts/zero3.json" # Updated deepspeed config path

# NOTE: Adjusted for 4 GPUs as in test-grpo-qwen.job
# Assuming 4 GPUs are available (0,1,2,3)

CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun \
    --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/r1-v/src/open_r1/grpo.py \
    --output_dir ${OUTPUT_DIR} \
    --model_name_or_path ${MODEL_PATH} \
    --dataset_name ${HF_DATASET} \
    --max_prompt_length 512 \
    --max_completion_length 512 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 true \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 2 \
    --run_name ${RUN_NAME} \
    --save_steps 100 \
    --save_only_model true \
    --report_to wandb \
    --num_generations 8 \
    --dataloader_num_workers 8 \
    --deepspeed ${DS_CONFIG} \
    2>&1 | tee "${OUTPUT_DIR}/training_log.txt"
