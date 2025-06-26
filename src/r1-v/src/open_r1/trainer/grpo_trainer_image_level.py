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
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from transformers import (
    is_wandb_available,
)
from transformers.utils import is_peft_available

if is_peft_available():
    pass

if is_wandb_available():
    import wandb

# Import the original trainer to extend from
from open_r1.trainer.grpo_trainer import Qwen2VLGRPOTrainer


class Qwen2VLGRPOTrainerImageLevel(Qwen2VLGRPOTrainer):
    """
    Image-Level GRPO trainer that properly trains vision components.
    
    Key differences from token-level GRPO:
    1. Computes log probabilities in continuous latent space (not discrete tokens)
    2. Maintains differentiable path through vision encoder/decoder
    3. One reward per complete image (not per token)
    4. Ensures gradients flow to all vision components
    """
    
    def __init__(self, *args, base_seed=42, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_seed = base_seed
        self.step_counter = 0
        
        # Create output directory for generated images
        self.image_output_dir = os.path.join(self.args.output_dir, "generated_images")
        os.makedirs(self.image_output_dir, exist_ok=True)
        
        # Store reference model's vision components for KL computation
        if self.ref_model is not None:
            self.ref_vision_encoder = self.ref_model.vision_model
            self.ref_vq_model = self.ref_model.vq_model
        else:
            # Will use model with adapter disabled as reference
            self.ref_vision_encoder = None
            self.ref_vq_model = None
    
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
    
    def compute_image_level_log_probs(self, model, text_inputs, generated_latents, is_reference=False):
        """
        Compute log probabilities at the image level using continuous latents.
        
        This maintains differentiability through the vision components.
        """
        # Get text embeddings
        text_outputs = model.language_model(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
            output_hidden_states=True
        )
        text_hidden_states = text_outputs.hidden_states[-1]  # Last layer hidden states
        
        # Get the conditioning vector from text (last token or pooled)
        # This is model-specific - adjust based on Janus architecture
        text_condition = text_hidden_states[:, -1, :]  # Use last token as condition
        
        # Compute log probability of generating these latents given the text condition
        # This is where we need to understand Janus's architecture
        # We need to find the distribution parameters (mean, logvar) for the latents
        
        # Option 1: If model has explicit latent distribution heads
        if hasattr(model, 'latent_mean_head') and hasattr(model, 'latent_logvar_head'):
            latent_mean = model.latent_mean_head(text_condition)
            latent_logvar = model.latent_logvar_head(text_condition)
            
            # Compute log probability under Gaussian distribution
            log_probs = -0.5 * (
                latent_logvar + 
                ((generated_latents - latent_mean) ** 2) / torch.exp(latent_logvar)
            ).sum(dim=-1)
        
        # Option 2: Use VQ model's quantization as probability measure
        elif hasattr(model, 'vq_model'):
            # Get VQ embeddings
            vq_model = model.vq_model if not is_reference else self.ref_vq_model
            
            # Quantize the generated latents
            quantized, indices, commit_loss = vq_model.quantize(generated_latents)
            
            # Compute distances to all codebook entries
            distances = torch.cdist(
                generated_latents.view(-1, generated_latents.size(-1)),
                vq_model.codebook.weight,
                p=2
            )
            
            # Convert distances to log probabilities (negative distances as logits)
            log_probs_per_token = F.log_softmax(-distances, dim=-1)
            
            # Get log prob of selected indices
            batch_size = generated_latents.size(0)
            selected_log_probs = log_probs_per_token[
                torch.arange(indices.numel(), device=indices.device),
                indices.view(-1)
            ].view(batch_size, -1)
            
            # Sum over spatial dimensions to get image-level log prob
            log_probs = selected_log_probs.sum(dim=1)
        
        # Option 3: Simple L2 distance as proxy for log probability
        else:
            # Use negative L2 distance as log probability proxy
            # This assumes a fixed-variance Gaussian centered at text_condition
            # Project text condition to latent dimension if needed
            if hasattr(model, 'text_to_latent_proj'):
                expected_latent = model.text_to_latent_proj(text_condition)
            else:
                # Simple linear projection
                latent_dim = generated_latents.view(generated_latents.size(0), -1).size(1)
                if text_condition.size(-1) != latent_dim:
                    # Create a learnable projection if it doesn't exist
                    if not hasattr(self, '_text_to_latent_proj'):
                        self._text_to_latent_proj = nn.Linear(
                            text_condition.size(-1), 
                            latent_dim
                        ).to(text_condition.device)
                    expected_latent = self._text_to_latent_proj(text_condition)
                else:
                    expected_latent = text_condition
            
            # Flatten latents for comparison
            flat_latents = generated_latents.view(generated_latents.size(0), -1)
            
            # Compute log probability as negative L2 distance
            variance = 1.0  # Could be learned
            log_probs = -0.5 * ((flat_latents - expected_latent) ** 2).sum(dim=-1) / variance
        
        return log_probs
    
    def generate_image_with_gradients(self, model, text_inputs, seed):
        """
        Generate image through forward pass while maintaining gradients.
        
        This is the key difference - we don't use model.generate() which breaks gradients.
        """
        device = text_inputs["input_ids"].device
        
        # Set seed for this generation
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        # Forward pass through language model to get text features
        text_outputs = model.language_model(
            input_ids=text_inputs["input_ids"],
            attention_mask=text_inputs["attention_mask"],
            output_hidden_states=True
        )
        
        # Get text conditioning
        # The exact method depends on Janus architecture
        text_features = text_outputs.hidden_states[-1][:, -1, :]  # Last token
        
        # Generate image latents from text features
        # This part is model-specific and needs to match Janus's architecture
        
        # Option 1: If model has explicit image generation head
        if hasattr(model, 'text_to_image_head'):
            image_latents = model.text_to_image_head(text_features)
        
        # Option 2: Sample from learned distribution
        elif hasattr(model, 'latent_mean_head') and hasattr(model, 'latent_logvar_head'):
            latent_mean = model.latent_mean_head(text_features)
            latent_logvar = model.latent_logvar_head(text_features)
            
            # Reparameterization trick for sampling
            std = torch.exp(0.5 * latent_logvar)
            eps = torch.randn_like(std)
            image_latents = latent_mean + eps * std
        
        # Option 3: Direct projection with noise
        else:
            # Get expected latent dimensions from vision model
            # This is a simplified approach - adjust based on actual model
            if hasattr(model, 'vision_model'):
                # Typical vision encoder output shape
                latent_height = latent_width = 16  # Adjust based on model
                latent_channels = 512  # Adjust based on model
            else:
                # Default values
                latent_height = latent_width = 16
                latent_channels = 512
            
            latent_dim = latent_channels * latent_height * latent_width
            
            # Project text features to latent dimension
            if not hasattr(self, '_text_to_latent_proj_forward'):
                self._text_to_latent_proj_forward = nn.Linear(
                    text_features.size(-1),
                    latent_dim
                ).to(device)
            
            flat_latents = self._text_to_latent_proj_forward(text_features)
            
            # Add noise for diversity
            noise_scale = 0.1  # Adjust as needed
            noise = torch.randn_like(flat_latents) * noise_scale
            flat_latents = flat_latents + noise
            
            # Reshape to spatial dimensions
            image_latents = flat_latents.view(
                flat_latents.size(0), 
                latent_channels, 
                latent_height, 
                latent_width
            )
        
        # Pass through VQ model if present
        if hasattr(model, 'vq_model'):
            # Quantize latents
            quantized_latents, indices, vq_loss = model.vq_model(image_latents)
            
            # Use straight-through estimator to maintain gradients
            image_latents = image_latents + (quantized_latents - image_latents).detach()
        
        # Decode latents to image
        if hasattr(model, 'vision_decoder'):
            decoded_image = model.vision_decoder(image_latents)
        elif hasattr(model, 'decode_image_tokens'):
            # Need to convert continuous latents to format expected by decode_image_tokens
            # This is model-specific
            decoded_image = model.decode_image_tokens(image_latents)
        else:
            # Fallback - simple upsampling
            if not hasattr(self, '_simple_decoder'):
                self._simple_decoder = nn.Sequential(
                    nn.ConvTranspose2d(image_latents.size(1), 256, 4, 2, 1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(256, 128, 4, 2, 1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(128, 64, 4, 2, 1),
                    nn.ReLU(),
                    nn.ConvTranspose2d(64, 3, 4, 2, 1),
                    nn.Tanh()
                ).to(device)
            decoded_image = self._simple_decoder(image_latents)
        
        return decoded_image, image_latents
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute image-level GRPO loss with proper gradient flow to vision components.
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
        
        print(f"\n=== Image-Level GRPO Training Step {self.step_counter} ===")
        print(f"Processing {len(prompts)} prompts")
        print(f"Generating {self.num_generations} images per prompt")
        
        # Lists to store all generated data
        all_rewards = []
        all_log_probs = []
        all_ref_log_probs = []
        
        # Process each prompt
        for prompt_idx, prompt in enumerate(prompts):
            print(f"\nProcessing prompt {prompt_idx + 1}/{len(prompts)}: {prompt[:50]}...")
            
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
            
            # Prepare text inputs
            text_inputs = self.processing_class(
                text=formatted_prompt,
                return_tensors="pt"
            ).to(device)
            
            # Generate multiple images for this prompt
            prompt_rewards = []
            prompt_log_probs = []
            prompt_ref_log_probs = []
            
            for gen_idx in range(self.num_generations):
                seed = self.base_seed + self.step_counter * 1000 + prompt_idx * 100 + gen_idx
                
                try:
                    # Generate image with gradients maintained
                    decoded_image, image_latents = self.generate_image_with_gradients(
                        model, text_inputs, seed
                    )
                    
                    # Convert to PIL for VQA and saving
                    # Normalize from [-1, 1] to [0, 1]
                    image_tensor = (decoded_image + 1) / 2
                    image_tensor = torch.clamp(image_tensor, 0, 1)
                    
                    # Convert to PIL
                    from torchvision.transforms import ToPILImage
                    to_pil = ToPILImage()
                    pil_image = to_pil(image_tensor[0].cpu())
                    
                    # Save image
                    image_filename = f"step_{self.step_counter}_prompt_{prompt_idx}_gen_{gen_idx}_seed_{seed}.png"
                    image_path = os.path.join(self.image_output_dir, image_filename)
                    pil_image.save(image_path)
                    
                    # Perform VQA evaluation
                    vqa_question = f"Score: How well does this image match '{prompt}'? Scale 0-10. Answer with number only: "
                    
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
                    
                    # Generate VQA response (no gradients needed here)
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
                    
                    # Compute log probabilities for this image
                    log_prob = self.compute_image_level_log_probs(
                        model, text_inputs, image_latents, is_reference=False
                    )
                    
                    # Compute reference log probabilities
                    with torch.no_grad():
                        if self.ref_model is not None:
                            # Generate reference latents
                            ref_decoded_image, ref_latents = self.generate_image_with_gradients(
                                self.ref_model, text_inputs, seed
                            )
                            ref_log_prob = self.compute_image_level_log_probs(
                                self.ref_model, text_inputs, ref_latents, is_reference=True
                            )
                        else:
                            # Use model with adapter disabled
                            with self.accelerator.unwrap_model(model).disable_adapter():
                                ref_decoded_image, ref_latents = self.generate_image_with_gradients(
                                    model, text_inputs, seed
                                )
                                ref_log_prob = self.compute_image_level_log_probs(
                                    model, text_inputs, ref_latents, is_reference=True
                                )
                    
                    # Store results
                    prompt_rewards.append(reward)
                    prompt_log_probs.append(log_prob)
                    prompt_ref_log_probs.append(ref_log_prob)
                    
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
                    import traceback
                    traceback.print_exc()
                    # Skip this generation
                    continue
            
            # Add to all lists
            if prompt_rewards:
                all_rewards.extend(prompt_rewards)
                all_log_probs.extend(prompt_log_probs)
                all_ref_log_probs.extend(prompt_ref_log_probs)
        
        # Check if we have any successful generations
        if not all_rewards:
            print("No successful image generations, returning zero loss")
            return torch.tensor(0.0, device=device, requires_grad=True)
        
        # Convert to tensors
        rewards = torch.tensor(all_rewards, dtype=torch.float32, device=device)
        log_probs = torch.stack(all_log_probs)
        ref_log_probs = torch.stack(all_ref_log_probs)
        
        # Compute KL divergence at image level
        kl_divergence = torch.exp(ref_log_probs - log_probs) - (ref_log_probs - log_probs) - 1
        
        # Compute grouped rewards statistics
        rewards_per_prompt = rewards.view(-1, self.num_generations)
        mean_grouped_rewards = rewards_per_prompt.mean(dim=1)
        std_grouped_rewards = rewards_per_prompt.std(dim=1)
        
        # Normalize rewards to compute advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        
        # Compute image-level GRPO loss
        # Key difference: one loss per image, not per token
        policy_loss = -(log_probs * advantages)
        kl_penalty = self.beta * kl_divergence
        
        loss = (policy_loss + kl_penalty).mean()
        
        # Verify gradients BEFORE optimizer step
        print(f"\n🧠 GRADIENT FLOW VERIFICATION (Step {self.step_counter})")
        has_image_grads = self.verify_image_encoder_gradients(model, loss)
        
        # Log metrics
        self._metrics["reward"].append(rewards.mean().item())
        self._metrics["reward_std"].append(std_grouped_rewards.mean().item())
        self._metrics["kl"].append(kl_divergence.mean().item())
        
        # Log to wandb
        if is_wandb_available() and self.args.report_to and "wandb" in self.args.report_to:
            if wandb.run is not None:
                wandb.log({
                    "train/mean_reward": rewards.mean().item(),
                    "train/max_reward": rewards.max().item(),
                    "train/min_reward": rewards.min().item(),
                    "train/reward_std": std_grouped_rewards.mean().item(),
                    "train/kl_divergence": kl_divergence.mean().item(),
                    "train/policy_loss": policy_loss.mean().item(),
                    "debug/image_encoder_training": float(has_image_grads),
                }, step=self.state.global_step)
        
        print(f"\nStep {self.step_counter} summary:")
        print(f"  Mean reward: {rewards.mean().item():.3f}")
        print(f"  Max reward: {rewards.max().item():.3f}")
        print(f"  Min reward: {rewards.min().item():.3f}")
        print(f"  KL divergence: {kl_divergence.mean().item():.3f}")
        print(f"  Image encoder training: {'✅ YES' if has_image_grads else '❌ NO'}")
        
        self.step_counter += 1
        
        return loss
    
    def verify_image_encoder_gradients(self, model, loss):
        """Specifically verify gradients are flowing through image encoder/decoder"""
        print("\n🔍 IMAGE ENCODER GRADIENT VERIFICATION")
        print("="*50)
        
        # Compute gradients
        loss.backward(retain_graph=True)
        
        # Focus on key image generation components
        image_encoder_components = {
            'Vision Encoder': ['vision_model', 'visual', 'encoder'],
            'Image Decoder': ['decoder', 'vq', 'quantize'],  
            'Conv Layers': ['conv', 'upsample', 'downsample'],
            'Embeddings': ['embed', 'position'],
            'Attention': ['self_attn', 'attn', 'q_proj', 'k_proj', 'v_proj'],
            'Custom Projections': ['_text_to_latent_proj', '_simple_decoder']
        }
        
        gradient_summary = {}
        
        # Check main model parameters
        for component_name, patterns in image_encoder_components.items():
            layers_with_grads = 0
            total_grad_norm = 0
            max_grad_norm = 0
            layer_names = []
            
            for name, param in model.named_parameters():
                if param.grad is not None and any(pattern in name.lower() for pattern in patterns):
                    grad_norm = torch.norm(param.grad).item()
                    if grad_norm > 1e-8:
                        layers_with_grads += 1
                        total_grad_norm += grad_norm
                        max_grad_norm = max(max_grad_norm, grad_norm)
                        layer_names.append((name, grad_norm))
            
            gradient_summary[component_name] = {
                'layers_with_grads': layers_with_grads,
                'total_grad_norm': total_grad_norm,
                'max_grad_norm': max_grad_norm,
                'layer_names': layer_names
            }
        
        # Also check our custom projection layers
        for attr_name in ['_text_to_latent_proj', '_text_to_latent_proj_forward', '_simple_decoder']:
            if hasattr(self, attr_name):
                module = getattr(self, attr_name)
                for name, param in module.named_parameters():
                    if param.grad is not None:
                        grad_norm = torch.norm(param.grad).item()
                        if grad_norm > 1e-8:
                            gradient_summary['Custom Projections']['layers_with_grads'] += 1
                            gradient_summary['Custom Projections']['total_grad_norm'] += grad_norm
                            gradient_summary['Custom Projections']['layer_names'].append(
                                (f"{attr_name}.{name}", grad_norm)
                            )
        
        # Print summary
        total_image_layers_with_grads = 0
        for component, data in gradient_summary.items():
            layers = data['layers_with_grads']
            total_norm = data['total_grad_norm']
            max_norm = data['max_grad_norm']
            
            total_image_layers_with_grads += layers
            
            if layers > 0:
                print(f"✅ {component}: {layers} layers, total_norm={total_norm:.2e}, max_norm={max_norm:.2e}")
                # Show top 2 layers for this component
                top_layers = sorted(data['layer_names'], key=lambda x: x[1], reverse=True)[:2]
                for layer_name, grad_norm in top_layers:
                    short_name = layer_name.split('.')[-2:] if '.' in layer_name else [layer_name]
                    print(f"   🔹 {'.'.join(short_name)}: {grad_norm:.2e}")
            else:
                print(f"❌ {component}: No gradients found")
        
        print(f"\n📊 SUMMARY: {total_image_layers_with_grads} image-related layers have gradients")
        
        # Log to wandb
        if is_wandb_available() and self.args.report_to and "wandb" in self.args.report_to:
            if wandb.run is not None:
                log_dict = {
                    "debug/total_image_layers_with_grads": total_image_layers_with_grads,
                }
                for component, data in gradient_summary.items():
                    log_dict[f"debug/{component.lower().replace(' ', '_')}_layers"] = data['layers_with_grads']
                    log_dict[f"debug/{component.lower().replace(' ', '_')}_grad_norm"] = data['total_grad_norm']
                
                wandb.log(log_dict, step=self.state.global_step)
        
        return total_image_layers_with_grads > 0