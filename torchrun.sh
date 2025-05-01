cd src/r1-v
export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b.txt"

# os.environ["HF_ENTITY"]="raoul"


torchrun --nproc_per_node="1" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    /home/rritter/test/R1-V/src/r1-v/src/open_r1/grpo.py \
    --output_dir="./share_models/Qwen2-VL-2B-Instruct_GRPO_CLEVR-70k" \
    --model_name_or_path="Qwen/Qwen2-VL-2B-Instruct" \
    --dataset_name="leonardPKU/clevr_cogen_a_train" \
    --deepspeed="/home/rritter/test/R1-V/src/r1-v/local_scripts/zero3.json" \
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
    --run_name Qwen2-VL-2B-GRPO-CLEVR-70k \
    --save_steps 100 \
    --save_only_model true \
    --num_generations 8   # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance  
