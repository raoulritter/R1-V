# Install the packages in r1-v .
module purge
module load 2024 
module load CUDA/12.6.0

uv venv --python 3.12

source .venv/bin/activate
cd src/r1-v 
uv pip install -e ".[dev]"

# Addtional modules
uv pip install wandb
uv pip install tensorboardx
uv pip install qwen_vl_utils torchvision
uv pip install flash-attn --no-build-isolation

# vLLM support 
uv pip install vllm

# fix transformers version
uv pip install git+https://github.com/huggingface/transformers.git@336dc69d63d56f232a183a3e7f52790429b871ef