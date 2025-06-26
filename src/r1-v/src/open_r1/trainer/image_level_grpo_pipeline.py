#!/usr/bin/env python3
"""
Image-Level GRPO pipeline that properly trains vision components.
Uses continuous latent space for gradient flow to vision encoder/decoder.
"""

import argparse
import os

import torch
from datasets import load_dataset
from transformers import JanusForConditionalGeneration, JanusProcessor
from trl import GRPOConfig

# Import the image-level GRPO trainer
from open_r1.trainer.grpo_trainer_image_level import Qwen2VLGRPOTrainerImageLevel


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
    parser = argparse.ArgumentParser(description="Image-Level GRPO pipeline")
    parser.add_argument("--dataset", type=str, default="BLIP3o/BLIP3o-60k", 
                        help="Dataset name")
    parser.add_argument("--num_prompts", type=int, default=2, 
                        help="Number of prompts to use")
    parser.add_argument("--num_generations", type=int, default=3, 
                        help="Number of generations per prompt")
    parser.add_argument("--max_steps", type=int, default=5, 
                        help="Maximum training steps")
    parser.add_argument("--output_dir", type=str, default="./image_level_grpo_results", 
                        help="Output directory")
    parser.add_argument("--learning_rate", type=float, default=5e-6,
                        help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="KL penalty coefficient")
    parser.add_argument("--base_seed", type=int, default=42,
                        help="Base seed for reproducibility")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing to save memory")
    
    args = parser.parse_args()
    
    print("=== Image-Level GRPO Pipeline ===")
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
    
    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    
    # Important: Set model to training mode
    model.train()
    
    # Ensure vision components are not frozen
    print("\nChecking if vision components are trainable...")
    vision_params_trainable = 0
    vision_params_frozen = 0
    
    for name, param in model.named_parameters():
        if any(component in name.lower() for component in ['vision', 'vq', 'decoder', 'encoder']):
            if param.requires_grad:
                vision_params_trainable += 1
            else:
                vision_params_frozen += 1
                # Unfreeze it
                param.requires_grad = True
                vision_params_trainable += 1
                vision_params_frozen -= 1
    
    print(f"Vision parameters - Trainable: {vision_params_trainable}, Frozen: {vision_params_frozen}")
    
    if vision_params_frozen > 0:
        print("Unfroze frozen vision parameters")
    
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
        run_name="image-level-grpo",
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
        report_to="wandb",  # Enable wandb for monitoring
        gradient_checkpointing=args.gradient_checkpointing,
    )
    
    # Dummy reward function (not used by image-level trainer)
    def dummy_reward_func(prompts, completions, **kwargs):
        return [0.0] * len(prompts)
    
    # Initialize trainer
    print("\nInitializing Image-Level GRPO trainer...")
    trainer = Qwen2VLGRPOTrainerImageLevel(
        model=model,
        reward_funcs=[dummy_reward_func],
        args=grpo_config,
        train_dataset=formatted_dataset,
        base_seed=args.base_seed,
        # Don't pass processing_class - let it be auto-detected
    )
    
    print("\n=== Starting Image-Level GRPO Training ===")
    print("Key differences from token-level GRPO:")
    print("✓ Computes log probabilities in continuous latent space")
    print("✓ Maintains differentiable path through vision encoder/decoder")
    print("✓ One reward per complete image (not per token)")
    print("✓ Ensures gradients flow to ALL vision components")
    print("")
    print("For each training step, the trainer will:")
    print(f"1. Generate {args.num_generations} images per prompt with gradient tracking")
    print("2. Evaluate each image with VQA to get a score (0-10)")
    print("3. Compute image-level log probabilities (not token-level)")
    print("4. Apply GRPO loss: -(log_probs * advantages).mean()")
    print("5. Update ALL model components including vision encoder/decoder")
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
    
    # Check if vision components received gradients
    print("\n=== Vision Component Training Verification ===")
    if hasattr(trainer, '_metrics') and 'debug/image_encoder_training' in trainer._metrics:
        training_success = sum(trainer._metrics['debug/image_encoder_training']) / len(trainer._metrics['debug/image_encoder_training'])
        print(f"Vision components received gradients in {training_success*100:.1f}% of steps")
    
    print(f"\nCheck the generated images in: {os.path.join(args.output_dir, 'generated_images')}")
    print("Monitor training progress in Weights & Biases if enabled")


if __name__ == "__main__":
    main()