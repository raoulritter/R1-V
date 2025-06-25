# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re

import torch
import torch.utils.data
from transformers import (
    is_wandb_available,
)
from transformers.utils import is_peft_available
from trl.models import (
    unwrap_model_for_generation,
)

if is_peft_available():
    pass

if is_wandb_available():
    import wandb

# Import the original trainer to extend from
from open_r1.trainer.grpo_trainer import Qwen2VLGRPOTrainer


class Qwen2VLGRPOTrainerFixed(Qwen2VLGRPOTrainer):
    """
    Fixed version of GRPO trainer that properly handles image generation with seeds
    and VQA evaluation in a single pipeline.
    """
    
    def __init__(self, *args, base_seed=42, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_seed = base_seed
        self.step_counter = 0
        
        # Create output directory for generated images
        self.image_output_dir = os.path.join(self.args.output_dir, "generated_images")
        os.makedirs(self.image_output_dir, exist_ok=True)
    
    def extract_score_from_vqa_answer(self, answer):
        """Extract numerical score from VQA evaluation answer"""
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
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute loss for GRPO training with proper image generation and VQA evaluation.
        
        This method:
        1. Extracts prompts from inputs
        2. Generates multiple images per prompt using different seeds
        3. Evaluates each image with VQA
        4. Calculates rewards based on VQA scores
        5. Computes GRPO loss
        """
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        
        device = self.accelerator.device
        
        # Extract prompts from inputs
        prompts = []
        for item in inputs:
            prompt_data = item.get("prompt", "")
            
            # Check if prompt is already in message format
            if isinstance(prompt_data, list) and len(prompt_data) > 0 and isinstance(prompt_data[0], dict):
                # Already in message format, extract the text content
                content = prompt_data[0].get("content", [])
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "text":
                            prompts.append(c.get("text", ""))
                            break
                else:
                    prompts.append(str(content))
            else:
                # Should not happen with proper formatting, but handle as fallback
                prompts.append(str(prompt_data))
        
        print(f"\n=== GRPO Training Step {self.step_counter} ===")
        print(f"Processing {len(prompts)} prompts")
        print(f"Generating {self.num_generations} images per prompt")
        
        # Lists to store all generated data
        all_generated_images = []
        all_rewards = []
        all_prompt_ids = []
        all_completion_ids = []
        all_attention_masks = []
        
        # Process each prompt
        for prompt_idx, prompt in enumerate(prompts):
            print(f"\nProcessing prompt {prompt_idx + 1}/{len(prompts)}: {prompt[:50]}...")
            
            # Generate multiple images for this prompt using different seeds
            prompt_rewards = []
            prompt_generated_images = []
            
            for gen_idx in range(self.num_generations):
                # Set seed for this generation
                seed = self.base_seed + self.step_counter * 1000 + prompt_idx * 100 + gen_idx
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                
                try:
                    # Format prompt for generation
                    messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt}
                            ]
                        }
                    ]
                    
                    # Apply chat template
                    formatted_prompt = self.processing_class.apply_chat_template(
                        messages, 
                        add_generation_prompt=True
                    )
                    
                    # Prepare inputs for image generation
                    generation_inputs = self.processing_class(
                        text=formatted_prompt,
                        generation_mode="image",
                        return_tensors="pt"
                    ).to(device)
                    
                    # Generate image
                    with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                        outputs = unwrapped_model.generate(
                            **generation_inputs,
                            generation_mode="image",
                            do_sample=True,
                            temperature=1.0,
                            use_cache=True,
                            num_return_sequences=1
                        )
                    
                    # Debug output shape
                    print(f"    Generated output shape: {outputs.shape}")
                    
                    # Store prompt and completion IDs
                    prompt_length = generation_inputs["input_ids"].size(1)
                    prompt_ids = outputs[:, :prompt_length]
                    completion_ids = outputs[:, prompt_length:]
                    
                    print(f"    Prompt length: {prompt_length}, Completion shape: {completion_ids.shape}")
                    
                    # Decode image - need to handle the full output, not just completion
                    try:
                        # The model expects the full generated sequence for decoding
                        decoded_image = model.decode_image_tokens(outputs)
                        images = self.processing_class.postprocess(
                            list(decoded_image.float()), 
                            return_tensors="PIL.Image.Image"
                        )
                        pil_image = images["pixel_values"][0]
                    except Exception as decode_error:
                        # If that fails, try with just the completion part
                        print(f"    Decode error with full output: {decode_error}")
                        print("    Trying with completion only...")
                        decoded_image = model.decode_image_tokens(completion_ids)
                        images = self.processing_class.postprocess(
                            list(decoded_image.float()), 
                            return_tensors="PIL.Image.Image"
                        )
                        pil_image = images["pixel_values"][0]
                    
                    # Save image
                    image_filename = f"step_{self.step_counter}_prompt_{prompt_idx}_gen_{gen_idx}_seed_{seed}.png"
                    image_path = os.path.join(self.image_output_dir, image_filename)
                    pil_image.save(image_path)
                    
                    # Perform VQA evaluation
                    # vqa_question = f"Does this image match and adhere to: '{prompt}'? Give a score from 0-10, where 0 means not at all and 10 means perfectly. Explain your reasoning."
                    vqa_question = f"Does this image match and adhere to: '{prompt}'? You MUST only give a score from 0-10. ONLY GIVE A SCORE, NO EXPLANATION."
                    
                    
                    # Format VQA messages
                    vqa_messages = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": pil_image},
                                {"type": "text", "text": vqa_question}
                            ]
                        }
                    ]
                    
                    # Apply chat template for VQA
                    vqa_inputs = self.processing_class.apply_chat_template(
                        vqa_messages,
                        add_generation_prompt=True,
                        generation_mode="text",
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt"
                    ).to(device)
                    
                    # Generate VQA response
                    with torch.no_grad():
                        vqa_outputs = model.generate(
                            **vqa_inputs,
                            max_new_tokens=256,
                            generation_mode="text",
                            do_sample=False,
                            num_beams=3,
                        )
                    
                    # Decode VQA answer
                    raw_answer = self.processing_class.decode(vqa_outputs[0], skip_special_tokens=False)
                    
                    # Extract assistant's response
                    answer = raw_answer
                    print(f"Raw answer: {raw_answer}")
                    if "<|Assistant|>:" in raw_answer:
                        assistant_part = raw_answer.split("<|Assistant|>:", 1)[1]
                        if "<｜end▁of▁sentence｜>" in assistant_part:
                            answer = assistant_part.split("<｜end▁of▁sentence｜>")[0].strip()
                        else:
                            answer = assistant_part.strip()
                    
                    # Extract score from answer
                    score = self.extract_score_from_vqa_answer(answer)
                    reward = float(score) / 10.0  # Normalize to 0-1 range
                    
                    print(f"  Gen {gen_idx + 1}: Score {score}/10 (seed {seed})")
                    
                    # Store results
                    prompt_rewards.append(reward)
                    prompt_generated_images.append(outputs)
                    all_prompt_ids.append(prompt_ids)
                    all_completion_ids.append(completion_ids)
                    
                    # Log first image per prompt to wandb
                    if gen_idx == 0 and is_wandb_available() and self.args.report_to and "wandb" in self.args.report_to:
                        if wandb.run is not None:
                            wandb.log({
                                f"generated_image/prompt_{prompt_idx}": wandb.Image(
                                    pil_image, 
                                    caption=f"Prompt: {prompt[:100]}... | Score: {score}/10"
                                )
                            }, step=self.state.global_step)
                    
                except Exception as e:
                    print(f"  Error generating image {gen_idx + 1}: {str(e)}")
                    prompt_rewards.append(0.0)
                    # Add dummy tensors to maintain consistency
                    if len(all_prompt_ids) > 0:
                        all_prompt_ids.append(all_prompt_ids[-1])
                        all_completion_ids.append(all_completion_ids[-1])
                    else:
                        # Skip this prompt entirely if first generation fails
                        continue
            
            # Convert rewards to tensor and add to all_rewards
            if prompt_rewards:
                all_rewards.extend(prompt_rewards)
        
        # Check if we have any successful generations
        if not all_rewards:
            print("No successful image generations, returning zero loss")
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Convert lists to tensors
        rewards = torch.tensor(all_rewards, dtype=torch.float32, device=device)
        
        # Stack all prompt and completion IDs
        prompt_ids = torch.cat(all_prompt_ids, dim=0)
        completion_ids = torch.cat(all_completion_ids, dim=0)
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        
        # Create attention masks
        prompt_mask = torch.ones_like(prompt_ids)
        completion_mask = torch.ones_like(completion_ids)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        
        # Get log probabilities
        per_token_logps = self._get_per_token_logps(
            model, 
            prompt_completion_ids, 
            attention_mask, 
            pixel_values=None,  # Not needed for Janus
            image_grid_thw=None
        )
        
        # Remove prompt portion
        prompt_length = prompt_ids.size(1)
        per_token_logps = per_token_logps[:, prompt_length - 1:]
        
        # Get reference model log probabilities
        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, 
                    prompt_completion_ids, 
                    attention_mask, 
                    pixel_values=None,
                    image_grid_thw=None
                )
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        model, 
                        prompt_completion_ids, 
                        attention_mask, 
                        pixel_values=None,
                        image_grid_thw=None
                    )
        ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]
        
        # Compute KL divergence
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        
        # Compute grouped rewards statistics
        rewards_per_prompt = rewards.view(-1, self.num_generations)
        mean_grouped_rewards = rewards_per_prompt.mean(dim=1)
        std_grouped_rewards = rewards_per_prompt.std(dim=1)
        
        # Normalize rewards to compute advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        
        # Compute GRPO loss
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        
        # Log metrics
        self._metrics["reward"].append(rewards.mean().item())
        self._metrics["reward_std"].append(std_grouped_rewards.mean().item())
        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(mean_kl.item())
        
        # Log reward statistics
        if is_wandb_available() and self.args.report_to and "wandb" in self.args.report_to:
            if wandb.run is not None:
                wandb.log({
                    "train/mean_reward": rewards.mean().item(),
                    "train/max_reward": rewards.max().item(),
                    "train/min_reward": rewards.min().item(),
                    "train/reward_std": std_grouped_rewards.mean().item(),
                }, step=self.state.global_step)
        
        print(f"\nStep {self.step_counter} summary:")
        print(f"  Mean reward: {rewards.mean().item():.3f}")
        print(f"  Max reward: {rewards.max().item():.3f}")
        print(f"  Min reward: {rewards.min().item():.3f}")
        
        self.step_counter += 1
        
        return loss