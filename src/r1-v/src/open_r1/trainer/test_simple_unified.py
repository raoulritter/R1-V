#!/usr/bin/env python3
"""
Simple unified GRPO pipeline without staged baseline generation.
All image generation and evaluation happens during GRPO training.
"""

import argparse
import os

import torch
from datasets import load_dataset
from transformers import JanusForConditionalGeneration, JanusProcessor
from trl import GRPOConfig

# Import the fixed GRPO trainer
from open_r1.trainer.grpo_trainer_fixed import Qwen2VLGRPOTrainerFixed


def format_dataset_for_grpo(dataset, text_field="text"):
    """Format dataset for GRPO training with proper message format for Janus"""
    def format_example(example):
        prompt_text = example.get(text_field, "")
        # Create the message format expected by Janus
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text}
                    ]
                }
            ]
        }
    
    return dataset.map(format_example)


def main():
    parser = argparse.ArgumentParser(description="Simple unified GRPO pipeline")
    parser.add_argument("--dataset", type=str, default="BLIP3o/BLIP3o-60k", 
                        help="Dataset name")
    parser.add_argument("--num_prompts", type=int, default=2, 
                        help="Number of prompts to use")
    parser.add_argument("--num_generations", type=int, default=3, 
                        help="Number of generations per prompt")
    parser.add_argument("--max_steps", type=int, default=5, 
                        help="Maximum training steps")
    parser.add_argument("--output_dir", type=str, default="./unified_grpo_results", 
                        help="Output directory")
    parser.add_argument("--learning_rate", type=float, default=5e-6,
                        help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="KL penalty coefficient")
    parser.add_argument("--base_seed", type=int, default=42,
                        help="Base seed for reproducibility")
    
    args = parser.parse_args()
    
    print("=== Simple Unified GRPO Pipeline ===")
    print(f"Dataset: {args.dataset}")
    print(f"Number of prompts: {args.num_prompts}")
    print(f"Generations per prompt: {args.num_generations}")
    print(f"Max training steps: {args.max_steps}")
    print(f"Output directory: {args.output_dir}")
    
    # Load model and processor
    model_id = "deepseek-community/Janus-Pro-1B"
    print(f"\nLoading model: {model_id}")
    processor = JanusProcessor.from_pretrained(model_id)
    
    # Determine dtype and device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        dtype = torch.bfloat16
    else:
        device = torch.device("cpu")
        dtype = torch.float16
    
    print(f"Device: {device}, Dtype: {dtype}")
    
    model = JanusForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load and format dataset
    print(f"\nLoading dataset: {args.dataset}")
    dataset = load_dataset(args.dataset, split="train")
    
    # Limit to specified number of prompts
    if args.num_prompts and args.num_prompts < len(dataset):
        dataset = dataset.select(range(args.num_prompts))
    
    print(f"Selected {len(dataset)} prompts")
    
    # Format dataset for GRPO
    formatted_dataset = format_dataset_for_grpo(dataset, "text")
    
    # Print sample prompts
    print("\nSample prompts:")
    for i in range(min(3, len(dataset))):
        print(f"  {i+1}: {dataset[i]['text'][:100]}...")
    
    # Create GRPO config - handle MPS properly
    use_bf16 = False
    use_fp16 = False
    if device.type == "cuda":
        use_bf16 = dtype == torch.bfloat16
        use_fp16 = dtype == torch.float16
    # For MPS, we don't set bf16/fp16 flags in the config
    
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        run_name="unified-grpo",
        beta=args.beta,
        num_generations=args.num_generations,
        max_prompt_length=512,
        max_completion_length=1024,  # Increased for image generation
        temperature=1.0,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        logging_steps=1,
        save_steps=100,
        bf16=use_bf16,
        fp16=use_fp16,
        report_to="wandb",  # Disable wandb for testing
    )
    
    # Dummy reward function (not used by fixed trainer)
    def dummy_reward_func(prompts, completions, **kwargs):
        return [0.0] * len(prompts)
    
    # Initialize trainer
    print("\nInitializing GRPO trainer...")
    trainer = Qwen2VLGRPOTrainerFixed(
        model=model,
        reward_funcs=[dummy_reward_func],
        args=grpo_config,
        train_dataset=formatted_dataset,
        base_seed=args.base_seed,
        # Don't pass processing_class - let it be auto-detected
    )
    
    print("\n=== Starting GRPO Training ===")
    print("For each training step, the trainer will:")
    print(f"1. Generate {args.num_generations} images per prompt using different seeds")
    print("2. Evaluate each image with VQA to get a score (0-10)")
    print("3. Use VQA scores as rewards for GRPO")
    print("4. Update the model to prefer higher-scoring generations")
    print(f"\nImages will be saved to: {os.path.join(args.output_dir, 'generated_images')}")
    
    # Train
    trainer.train()
    
    print("\n=== Training Complete ===")
    
    # Save model
    final_model_path = os.path.join(args.output_dir, "final_model")
    trainer.save_model(final_model_path)
    print(f"Model saved to: {final_model_path}")
    
    # Print summary
    print("\n=== Summary ===")
    print(f"Total prompts processed: {len(dataset)}")
    print(f"Images generated per prompt per step: {args.num_generations}")
    print(f"Total training steps: {args.max_steps}")
    print(f"Total images generated: {len(dataset) * args.num_generations * args.max_steps}")
    print(f"\nCheck the generated images in: {os.path.join(args.output_dir, 'generated_images')}")


if __name__ == "__main__":
    main()