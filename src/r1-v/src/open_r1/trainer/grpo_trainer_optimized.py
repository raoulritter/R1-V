#!/usr/bin/env python3
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
from transformers import is_wandb_available
from transformers.utils import is_peft_available
from trl.models import unwrap_model_for_generation

if is_peft_available():
    pass

if is_wandb_available():
    import wandb

# Import the original trainer to extend from
from open_r1.trainer.grpo_trainer import Qwen2VLGRPOTrainer


class Qwen2VLGRPOTrainerOptimized(Qwen2VLGRPOTrainer):
    """
    Optimized version of GRPO trainer that:
    1. Generates multiple images per prompt in parallel using num_return_sequences
    2. Processes multiple prompts in parallel via batch_size
    3. Handles variable sequence lengths properly
    4. Reduces memory overhead and GPU idle time
    """
    
    def __init__(self, *args, base_seed=42, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_seed = base_seed
        self.step_counter = 0
        
        # Create output directory for generated images
        self.image_output_dir = os.path.join(self.args.output_dir, "generated_images")
        os.makedirs(self.image_output_dir, exist_ok=True)
        
        print(f"Optimized GRPO Trainer initialized with:")
        print(f"  - Batch size: {self.args.per_device_train_batch_size}")
        print(f"  - Generations per prompt: {self.num_generations}")
        print(f"  - Base seed: {self.base_seed}")
    
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
    
    def pad_to_max_length(self, tensor_list, pad_token_id=0):
        """Pad a list of tensors to the same length"""
        if not tensor_list:
            return torch.empty(0, 0, dtype=torch.long)
        
        max_length = max(t.size(1) for t in tensor_list)
        padded_tensors = []
        
        for tensor in tensor_list:
            if tensor.size(1) < max_length:
                padding = torch.full((tensor.size(0), max_length - tensor.size(1)), 
                                   pad_token_id, dtype=tensor.dtype, device=tensor.device)
                padded_tensor = torch.cat([tensor, padding], dim=1)
            else:
                padded_tensor = tensor
            padded_tensors.append(padded_tensor)
        
        return torch.cat(padded_tensors, dim=0)
    
    def process_single_prompt(self, model, prompt, prompt_idx, device):
        """Process a single prompt with parallel image generation"""
        
        # Set unique seeds for each generation
        base_seed = self.base_seed + self.step_counter * 10000 + prompt_idx * 1000
        torch.manual_seed(base_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(base_seed)
        
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
            
            # Generate multiple images in parallel - THIS IS THE KEY OPTIMIZATION!
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                outputs = unwrapped_model.generate(
                    **generation_inputs,
                    generation_mode="image",
                    do_sample=True,
                    temperature=1.0,
                    use_cache=True,
                    num_return_sequences=self.num_generations,  # Generate all images at once!
                )
            
            print(f"      Generated {outputs.shape[0]} images in parallel")
            
            # Decode and save all images
            images = []
            prompt_length = generation_inputs["input_ids"].size(1)
            
            # Store tensor data for GRPO loss computation
            prompt_ids_list = []
            completion_ids_list = []
            
            # Process each generated image
            for gen_idx in range(outputs.shape[0]):
                try:
                    # Extract single generation
                    gen_output = outputs[gen_idx:gen_idx+1]  # Keep batch dimension
                    
                    # Store tensor data
                    prompt_ids = gen_output[:, :prompt_length]
                    completion_ids = gen_output[:, prompt_length:]
                    prompt_ids_list.append(prompt_ids)
                    completion_ids_list.append(completion_ids)
                    
                    # Decode image
                    decoded_image = model.decode_image_tokens(gen_output)
                    processed_images = self.processing_class.postprocess(
                        list(decoded_image.float()), 
                        return_tensors="PIL.Image.Image"
                    )
                    pil_image = processed_images["pixel_values"][0]
                    images.append(pil_image)
                    
                    # Save image
                    seed = base_seed + gen_idx
                    image_filename = f"step_{self.step_counter}_prompt_{prompt_idx}_gen_{gen_idx}_seed_{seed}.png"
                    image_path = os.path.join(self.image_output_dir, image_filename)
                    pil_image.save(image_path)
                    
                except Exception as e:
                    print(f"        Error decoding image {gen_idx}: {e}")
                    continue
            
            if not images:
                print(f"      No images successfully decoded for prompt {prompt_idx}")
                return None
            
            # Evaluate all images for this prompt
            rewards = self.evaluate_images_batch(model, images, prompt, device)
            
            if not rewards:
                print(f"      No rewards generated for prompt {prompt_idx}")
                return None
            
            print(f"      Rewards: {[f'{r:.2f}' for r in rewards]}")
            
            # Log first image to wandb
            if is_wandb_available() and self.args.report_to and "wandb" in self.args.report_to:
                if wandb.run is not None and images:
                    wandb.log({
                        f"generated_image/step_{self.step_counter}_prompt_{prompt_idx}": wandb.Image(
                            images[0], 
                            caption=f"Prompt: {prompt[:100]}... | Score: {rewards[0]*10:.1f}/10"
                        )
                    }, step=self.state.global_step)
            
            # Return results for this prompt
            return {
                'rewards': rewards,
                'prompt_ids': prompt_ids_list,
                'completion_ids': completion_ids_list,
                'images': images
            }
            
        except Exception as e:
            print(f"      Error processing prompt {prompt_idx}: {e}")
            return None
    
    def evaluate_images_batch(self, model, images, prompt, device):
        """Evaluate multiple images with VQA in optimized batches"""
        
        if not images:
            return []
        
        vqa_question = f"Score: How well does this image match '{prompt}' ? Scale 0-10. Answer with number only:"
        rewards = []
        
        # Process VQA evaluations - we can't easily batch these due to different image sizes
        # But we can optimize by reducing inference overhead
        for i, pil_image in enumerate(images):
            try:
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
                        max_new_tokens=50,  # Reduced from 256 since we only need a number
                        generation_mode="text",
                        do_sample=False,
                        # num_beams=1,  # Reduced from 3 for speed
                    )
                
                # Decode VQA answer
                raw_answer = self.processing_class.decode(vqa_outputs[0], skip_special_tokens=False)
                
                # Extract assistant's response
                answer = raw_answer
                if "<|Assistant|>:" in raw_answer:
                    assistant_part = raw_answer.split("<|Assistant|>:", 1)[1]
                    if " " in assistant_part:
                        answer = assistant_part.split(" ")[0].strip()
                    else:
                        answer = assistant_part.strip()
                
                # Extract score from answer
                score = self.extract_score_from_vqa_answer(answer)
                reward = float(score) / 10.0  # Normalize to 0-1 range
                rewards.append(reward)
                
            except Exception as e:
                print(f"        Error evaluating image {i}: {e}")
                rewards.append(0.0)
        
        return rewards
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Optimized compute_loss that handles variable sequence lengths properly
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
        print(f"Processing {len(prompts)} prompts in batch")
        print(f"Generating {self.num_generations} images per prompt in parallel")
        
        # Process each prompt and collect results
        all_rewards = []
        all_prompt_ids = []
        all_completion_ids = []
        all_images = []
        
        for prompt_idx, prompt in enumerate(prompts):
            print(f"    Processing prompt {prompt_idx + 1}/{len(prompts)}: {prompt[:50]}...")
            
            result = self.process_single_prompt(model, prompt, prompt_idx, device)
            
            if result is not None:
                all_rewards.extend(result['rewards'])
                all_prompt_ids.extend(result['prompt_ids'])
                all_completion_ids.extend(result['completion_ids'])
                all_images.extend(result['images'])
        
        # Check if we have any successful generations
        if not all_rewards:
            print("No successful image generations, returning zero loss")
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Convert results to tensors for GRPO loss computation
        rewards = torch.tensor(all_rewards, dtype=torch.float32, device=device)
        
        if not all_prompt_ids:
            print("No prompt/completion IDs available, returning zero loss")
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Pad tensors to the same length before concatenating
        prompt_ids = self.pad_to_max_length(all_prompt_ids, pad_token_id=self.processing_class.tokenizer.pad_token_id or 0)
        completion_ids = self.pad_to_max_length(all_completion_ids, pad_token_id=self.processing_class.tokenizer.pad_token_id or 0)
        
        # Combine prompt and completion IDs
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
                    "train/num_images_generated": len(all_images),
                    "train/batch_size": len(prompts),
                }, step=self.state.global_step)
        
        print(f"\nStep {self.step_counter} summary:")
        print(f"  Processed {len(prompts)} prompts")
        print(f"  Generated {len(all_images)} images total")
        print(f"  Mean reward: {rewards.mean().item():.3f}")
        print(f"  Max reward: {rewards.max().item():.3f}")
        print(f"  Min reward: {rewards.min().item():.3f}")
        print(f"  Reward std: {std_grouped_rewards.mean().item():.3f}")
        
        self.step_counter += 1
        
        return loss