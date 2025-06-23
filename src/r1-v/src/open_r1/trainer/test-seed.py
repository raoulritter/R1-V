import argparse
import json
import os
import re
from typing import List

import torch
from datasets import load_dataset
from transformers import (
    JanusForConditionalGeneration,
    JanusProcessor,
)
from trl import GRPOConfig

# Import the existing GRPO trainer
from open_r1.trainer.grpo_trainer import Qwen2VLGRPOTrainer


class VQARewardFunction:
    """VQA-based reward function for GRPO training"""
    
    def __init__(self, processor, model):
        self.processor = processor
        self.model = model
    
    def __call__(self, prompts: List[str], completions: List[torch.Tensor], **kwargs) -> List[float]:
        """
        Evaluate generated images using VQA and return reward scores.
        
        Args:
            prompts: List of text prompts
            completions: List of generated image tensors
            
        Returns:
            List of reward scores (0-10 scale)
        """
        rewards = []
        
        for prompt, completion in zip(prompts, completions):
            try:
                # Decode the image tokens to get PIL image
                decoded_image = self.model.decode_image_tokens(completion.unsqueeze(0))
                images = self.processor.postprocess(
                    list(decoded_image.float()), 
                    return_tensors="PIL.Image.Image"
                )
                image = images["pixel_values"][0]
                
                # Create VQA evaluation question
                evaluation_question = f"Does this image match and adhere to: '{prompt}'? Give a score from 0-10, where 0 means not at all and 10 means perfectly. Explain your reasoning."
                
                # Format messages for VQA
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": evaluation_question}
                        ]
                    }
                ]

                # Apply chat template
                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    generation_mode="text",
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt"
                ).to(self.model.device)
                
                # Generate VQA response
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    generation_mode="text",
                    do_sample=False,
                    num_beams=3,
                )
                
                # Decode generated text
                raw_answer = self.processor.decode(outputs[0], skip_special_tokens=True)
                
                # Extract text after "ASSISTANT:" if present
                if "ASSISTANT:" in raw_answer:
                    answer = raw_answer.split("ASSISTANT:", 1)[1].strip()
                else:
                    answer = raw_answer.strip()
                
                # Extract score from answer
                score = self.extract_score_from_answer(answer)
                rewards.append(float(score))
                
            except Exception as e:
                print(f"Error evaluating image for prompt '{prompt}': {str(e)}")
                rewards.append(0.0)
        
        return rewards
    
    def extract_score_from_answer(self, answer):
        """Extract numerical score from evaluation answer"""
        # Look for patterns like "score: X" or "X/10" or just a standalone number
        score_patterns = [
            r'score:?\s*(\d+)',  # "score: X"
            r'(\d+)\s*/\s*10',   # "X/10"
            r'(\d+)\s*out of\s*10',  # "X out of 10"
            r'give\s*(?:it|a|the)?\s*(?:score\s*(?:of)?)?\s*(\d+)',  # "give it a score of X" 
            r'rating\s*(?:of|is)?\s*(\d+)',  # "rating of X"
        ]
        
        # Try each pattern
        for pattern in score_patterns:
            match = re.search(pattern, answer.lower())
            if match:
                score = int(match.group(1))
                # Limit score to 0-10 range
                return max(0, min(10, score))
        
        # If no pattern matched, look for any standalone digit from 0-10
        digit_match = re.search(r'\b([0-9]|10)\b', answer)
        if digit_match:
            return int(digit_match.group(1))
        
        # Default score if no score found
        return 0


class JanusGRPOTrainer:
    """GRPO Trainer for Janus image generation with VQA-based rewards"""
    
    def __init__(self, output_dir="./janus_grpo_results"):
        # Load Janus-Pro model and processor
        model_id = "deepseek-community/Janus-Pro-1B"
        self.processor = JanusProcessor.from_pretrained(model_id)
        
        # Determine proper dtype for the device
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
            self.dtype = torch.bfloat16
        else:
            self.device = torch.device("cpu")
            self.dtype = torch.float16
        
        # Load model with appropriate configuration
        self.model = JanusForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=self.dtype, 
            device_map="auto",
            trust_remote_code=True
        )
        
        # Create output directory
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "checkpoints"), exist_ok=True)
        
        print(f"Janus-Pro GRPO trainer initialized on {self.device}")
        print(f"Using dtype: {self.dtype}")

    def create_reward_function(self):
        """Create VQA-based reward function for GRPO"""
        return VQARewardFunction(self.processor, self.model)

    def format_dataset_for_grpo(self, dataset, text_field="text"):
        """Format dataset for GRPO training"""
        def format_example(example):
            prompt_text = example.get(text_field, "")
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

    def train_with_grpo(
        self,
        dataset_name="BLIP3o/BLIP3o-60k",
        num_prompts=None,
        text_field="text",
        num_generations=8,  # Number of diverse images per prompt
        max_prompt_length=512,
        max_completion_length=512,
        num_train_epochs=1,
        max_steps=500,
        beta=0.1,
        learning_rate=5e-6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        save_steps=100,
        logging_steps=10,
        eval_steps=100
    ):
        """
        Train the model using GRPO with VQA-based rewards.
        
        The training process:
        1. For each prompt, generate `num_generations` diverse images using different seeds
        2. Evaluate each image with VQA to get reward scores
        3. Use GRPO to update the model to prefer higher-scoring images
        4. Repeat for all prompts in the dataset
        """
        
        # Load dataset
        dataset = load_dataset(dataset_name, split="train")
        
        # Limit to specified number of prompts if needed
        if num_prompts and num_prompts > 0:
            if num_prompts < len(dataset):
                dataset = dataset.select(range(num_prompts))
        
        print(f"Training on {len(dataset)} prompts from {dataset_name}")
        print(f"Generating {num_generations} diverse images per prompt for GRPO training")
        
        # Format dataset for GRPO
        formatted_dataset = self.format_dataset_for_grpo(dataset, text_field)
        
        # Create reward function
        reward_func = self.create_reward_function()
        
        # Determine appropriate precision settings based on device
        use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        # MPS doesn't support fp16 in accelerate/transformers, so disable it
        use_fp16 = not use_bf16 and torch.cuda.is_available()
        
        # Create GRPO config with all necessary parameters
        grpo_config = GRPOConfig(
            output_dir=self.output_dir,
            run_name="janus-grpo-image-gen",
            beta=beta,
            num_generations=num_generations,
            max_prompt_length=max_prompt_length,
            max_completion_length=max_completion_length,
            temperature=1.0,  # Fixed temperature for GRPO
            use_vllm=False,  # Disable vLLM for now
            num_train_epochs=num_train_epochs,
            max_steps=max_steps,
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            weight_decay=0.01,
            logging_steps=logging_steps,
            save_steps=save_steps,
            eval_steps=eval_steps,
            bf16=use_bf16,
            fp16=use_fp16,
            report_to="wandb",  # Enable wandb logging
        )
        
        # Initialize GRPO trainer
        trainer = Qwen2VLGRPOTrainer(
            model=self.model,
            reward_funcs=[reward_func],
            args=grpo_config,
            train_dataset=formatted_dataset,
            # processing_class=self.processor,
        )
        
        print("Starting GRPO training...")
        print("GRPO Training Process:")
        print("1. Generate diverse images per prompt using different seeds")
        print("2. Evaluate each image with VQA for reward scores") 
        print("3. Update model to prefer higher-scoring generations")
        print("4. Repeat across dataset to improve image generation quality")
        
        # Train the model
        trainer.train()
        
        # Save the trained model
        final_model_path = os.path.join(self.output_dir, "final_model")
        trainer.save_model(final_model_path)
        
        print(f"GRPO training completed. Model saved to {final_model_path}")
        
        # Save training summary
        training_summary = {
            "dataset": dataset_name,
            "num_prompts": len(dataset),
            "num_generations_per_prompt": num_generations,
            "total_images_generated": len(dataset) * num_generations * max_steps,
            "max_steps": max_steps,
            "beta": beta,
            "learning_rate": learning_rate,
            "final_model_path": final_model_path
        }
        
        summary_path = os.path.join(self.output_dir, "training_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(training_summary, f, indent=2)
        
        print(f"Training summary saved to {summary_path}")
        
        return trainer


def main():
    parser = argparse.ArgumentParser(description="GRPO Training for Janus Image Generation with Diverse Seed-based Generation")
    parser.add_argument("--dataset", type=str, default="BLIP3o/BLIP3o-60k",
                        help="Dataset name or path")
    parser.add_argument("--output_dir", type=str, default="./janus_grpo_results",
                        help="Directory to save training results and generated images")
    parser.add_argument("--num_prompts", type=int, default=None,
                        help="Number of prompts to process (default: process all)")
    parser.add_argument("--text_field", type=str, default="text",
                        help="Field in dataset containing text prompts")
    
    # GRPO training arguments
    parser.add_argument("--num_generations", type=int, default=8,
                        help="Number of diverse images to generate per prompt for GRPO (using different seeds)")
    parser.add_argument("--max_prompt_length", type=int, default=512,
                        help="Maximum prompt length")
    parser.add_argument("--max_completion_length", type=int, default=512,
                        help="Maximum completion length (image tokens)")
    parser.add_argument("--num_train_epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=500,
                        help="Maximum number of training steps")
    parser.add_argument("--beta", type=float, default=0.1,
                        help="KL penalty coefficient for GRPO")
    parser.add_argument("--learning_rate", type=float, default=5e-6,
                        help="Learning rate")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1,
                        help="Batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--save_steps", type=int, default=100,
                        help="Save checkpoint every N steps")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Log every N steps")
    parser.add_argument("--eval_steps", type=int, default=100,
                        help="Evaluate every N steps")
    parser.add_argument("--wandb_project", type=str, default="janus-grpo-training",
                        help="Wandb project name")
    
    args = parser.parse_args()
    
    # Initialize wandb if using it
    if os.environ.get("WANDB_PROJECT") is None:
        os.environ["WANDB_PROJECT"] = args.wandb_project
    
    # Initialize GRPO trainer
    trainer = JanusGRPOTrainer(output_dir=args.output_dir)
    
    # Run GRPO training with diverse seed-based generation
    print("Starting GRPO training with diverse seed-based image generation...")
    print(f"Will generate {args.num_generations} diverse images per prompt using different seeds")
    print("VQA evaluation will provide reward signals for GRPO optimization")
    
    trained_model = trainer.train_with_grpo(
        dataset_name=args.dataset,
        num_prompts=args.num_prompts,
        text_field=args.text_field,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        beta=args.beta,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps
    )
    
    print("GRPO training completed successfully!")
    print("The model has been trained to generate better images based on VQA feedback.")


if __name__ == "__main__":
    main()