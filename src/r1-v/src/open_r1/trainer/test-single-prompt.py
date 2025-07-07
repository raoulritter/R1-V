#!/usr/bin/env python3
"""
Single prompt GRPO test - runs 100 steps with one prompt to test strict scoring.
"""

import argparse
import os

import torch
from datasets import Dataset
from transformers import JanusForConditionalGeneration, JanusProcessor
from trl import GRPOConfig

# Import the fixed GRPO trainer
from open_r1.trainer.grpo_trainer_optimized import Qwen2VLGRPOTrainerOptimized


def create_single_prompt_dataset(prompt_text, num_steps=100):
    """Create a dataset with a single prompt repeated for the specified number of steps"""
    data = [{"text": prompt_text} for _ in range(num_steps)]
    return Dataset.from_list(data)


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
    parser = argparse.ArgumentParser(description="Single prompt GRPO test with strict scoring")
    parser.add_argument("--prompt", type=str, default="a photo of a computer mouse and a chair", 
                        help="Single prompt to test")
    parser.add_argument("--num_steps", type=int, default=100, 
                        help="Number of training steps")
    parser.add_argument("--num_generations", type=int, default=8, 
                        help="Number of generations per step")
    parser.add_argument("--output_dir", type=str, default="./single_prompt_test", 
                        help="Output directory")
    parser.add_argument("--learning_rate", type=float, default=4e-5,
                        help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="KL penalty coefficient")
    parser.add_argument("--base_seed", type=int, default=42,
                        help="Base seed for reproducibility")
    
    args = parser.parse_args()
    
    print("=== Single Prompt GRPO Test with Strict Scoring ===")
    print(f"Prompt: {args.prompt}")
    print(f"Number of steps: {args.num_steps}")
    print(f"Generations per step: {args.num_generations}")
    print(f"Output directory: {args.output_dir}")
    print(f"Total images to generate: {args.num_steps * args.num_generations}")
    
    # Load model and processor
    model_id = "deepseek-community/Janus-Pro-7B"
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
    
    # Create single prompt dataset
    print(f"\nCreating dataset with single prompt repeated {args.num_steps} times")
    dataset = create_single_prompt_dataset(args.prompt, args.num_steps)
    
    # Format dataset for GRPO
    formatted_dataset = format_dataset_for_grpo(dataset, "text")
    
    # Create GRPO config - handle MPS properly
    use_bf16 = False
    use_fp16 = False
    if device.type == "cuda":
        use_bf16 = dtype == torch.bfloat16
        use_fp16 = dtype == torch.float16
    # For MPS, we don't set bf16/fp16 flags in the config
    
    grpo_config = GRPOConfig(
        output_dir=args.output_dir,
        run_name=f"single-prompt-strict-{args.num_steps}steps-{args.num_generations}gens",
        beta=args.beta,
        num_generations=args.num_generations,
        max_prompt_length=512,
        max_completion_length=1024,
        temperature=1.0,
        max_steps=args.num_steps,  # Direct step count
        per_device_train_batch_size=1,  # Process one prompt at a time
        gradient_accumulation_steps=1,
        learning_rate=args.learning_rate,
        logging_steps=1,
        save_steps=100,  # Save every 25 steps
        bf16=use_bf16,
        fp16=use_fp16,
        report_to="wandb",
    )
    
    # Dummy reward function (not used by fixed trainer)
    def dummy_reward_func(prompts, completions, **kwargs):
        return [0.0] * len(prompts)
    
    # Initialize trainer
    print("\nInitializing GRPO trainer...")
    trainer = Qwen2VLGRPOTrainerOptimized(
        model=model,
        reward_funcs=[dummy_reward_func],
        args=grpo_config,
        train_dataset=formatted_dataset,
        base_seed=args.base_seed,
    )
    
    print("\n=== Starting Single Prompt GRPO Training ===")
    print("This test will:")
    print(f"1. Use the same prompt '{args.prompt}' for all {args.num_steps} steps")
    print(f"2. Generate {args.num_generations} images per step (total: {args.num_steps * args.num_generations} images)")
    print("3. Use STRICT scoring - only perfect matches get score 10")
    print("4. Track how the model improves at generating this specific prompt")
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
    print(f"Prompt tested: {args.prompt}")
    print(f"Training steps: {args.num_steps}")
    print(f"Images generated per step: {args.num_generations}")
    print(f"Total images generated: {args.num_steps * args.num_generations}")
    print(f"\nCheck the generated images in: {os.path.join(args.output_dir, 'generated_images')}")
    # print("Look for improvement in image quality and adherence to the prompt over time!")


if __name__ == "__main__":
    main() 