import argparse
import json
import os

import ray
import torch
from datasets import load_dataset  # Add import for datasets library
from tqdm import tqdm
from transformers import JanusForConditionalGeneration, JanusProcessor


# Define the Actor Class
# @ray.remote(num_gpus=1)
class JanusImageGenVQAActor:
    def __init__(self, output_dir="./janus_results"):
        # Load Janus-Pro model and processor
        model_id = "deepseek-community/Janus-Pro-1B"
        self.processor = JanusProcessor.from_pretrained(model_id)
        
        # Determine if bfloat16 is supported
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.bfloat16 if torch.backends.mps.is_available() else torch.float16
        
        # Load model with appropriate configuration
        self.model = JanusForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=dtype, 
            device_map="auto",
            trust_remote_code=True
        )
        
        # Ensure the actor uses the GPU assigned to it by Ray
        self.gpu_ids = ray.get_gpu_ids() if ray.is_initialized() else [0]
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        # Set default generation config
        self.model.generation_config.num_return_sequences = 1  # Default to 1 image per caption
        
        # Create output directory
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, "images"), exist_ok=True)
        
        print(f"Janus-Pro actor initialized on GPU {self.gpu_ids}")
        print(f"Using dtype: {dtype}")

    def generate_image_from_caption(self, caption, seed=42, num_images=1):
        """Generate images from a caption using Janus-Pro"""
        print(f"Generating {num_images} image(s) from caption: {caption}")
        
        # Handle caption if it's a list
        if isinstance(caption, list):
            # Take first item from the list if it's a list
            caption = caption[0] if caption else ""
        
        # Set seed for reproducibility
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        
        # Prepare input for image generation
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": caption}
                ]
            }
        ]
        
        # Apply chat template
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        
        # Set the number of images to generate
        self.model.generation_config.num_return_sequences = num_images
        
        # Prepare inputs
        inputs = self.processor(
            text=prompt,
            generation_mode="image",
            return_tensors="pt"
        ).to(self.model.device)
        
        # Generate images
        try:
            outputs = self.model.generate(
                **inputs,
                generation_mode="image",
                do_sample=True,
                use_cache=True
            )
            
            # Decode the generated image tokens
            decoded_image = self.model.decode_image_tokens(outputs)
            
            # Convert to PIL images
            images = self.processor.postprocess(
                list(decoded_image.float()), 
                return_tensors="PIL.Image.Image"
            )
            
            # Save generated images
            image_paths = []
            for i, image in enumerate(images["pixel_values"]):
                # Create a safe filename from the caption
                safe_caption = "".join(c if c.isalnum() else "_" for c in caption[:30])
                image_filename = f"{safe_caption}_{seed}_{i}.png"
                image_path = os.path.join(self.output_dir, "images", image_filename)
                
                # Save the image
                image.save(image_path)
                image_paths.append(image_path)
                
            return image_paths, images["pixel_values"]
            
        except Exception as e:
            print(f"Error generating image from caption '{caption}': {str(e)}")
            return [], []

    # def perform_vqa(self, image, questions):
    #     """Perform VQA on an image using a list of questions"""
    #     results = {}
        
    #     for question in questions:
    #         print(f"Processing VQA question: {question}")
            
    #         # Format messages for VQA
    #         messages = [
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {"type": "image", "image": image},
    #                     {"type": "text", "text": question}
    #                 ]
    #             }
    #         ]

    #         # Apply chat template
    #         inputs = self.processor.apply_chat_template(
    #             messages,
    #             add_generation_prompt=True,
    #             generation_mode="text",
    #             tokenize=True,
    #             return_dict=True,
    #             return_tensors="pt"
    #         ).to(self.model.device)
            
    #         # Generate response
    #         try:
    #             outputs = self.model.generate(
    #                 **inputs,
    #                 max_new_tokens=512,
    #                 generation_mode="text",
    #                 do_sample=False,
    #                 num_beams=3,
    #             )
                
    #             # Decode generated text
    #             answer = self.processor.decode(outputs[0], skip_special_tokens=True)
                
    #             # Remove the template parts to get only the model's answer
    #             if "ASSISTANT:" in answer:
    #                 answer = answer.split("ASSISTANT:", 1)[1].strip()
                
    #             answer
    #             # Store result
    #             results[question] = answer
    #             print(f"Question: {question}")
    #             print(f"Answer: {answer}")
                
    #         except Exception as e:
    #             print(f"Error performing VQA with question '{question}': {str(e)}")
    #             results[question] = f"Error: {str(e)}"
            
    #     return results

    def perform_vqa(self, image, questions):
        """Perform VQA on an image using a list of questions"""
        results = {}
        
        for question in questions:
            print(f"Processing VQA question: {question}")
            
            # Format messages for VQA
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": question}
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
            
            # Generate response
            try:
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
                
                # Look for keywords that indicate the real answer starts
                # First, try to find the actual result which often follows a specific pattern
                answer_patterns = [
                    "Answer:", "answer:", 
                    "Score:", "score:", 
                    "The reality score", "the reality score",
                    "The scene score", "the scene score",
                    "Result:", "result:"
                ]
                
                clean_answer = answer  # Default if no patterns match
                
                # Try to find any of the answer patterns
                for pattern in answer_patterns:
                    if pattern in answer:
                        # Extract everything after the pattern
                        clean_answer = answer.split(pattern, 1)[1].strip()
                        break
                
                # If no answer pattern was found, try the colon approach - often the actual answer 
                # is after the last colon in cases like "...:\nAnswer: 2 points"
                if clean_answer == answer and ":" in answer:
                    colon_parts = answer.split(":")
                    # If the last part after a colon looks like an actual succinct answer
                    last_part = colon_parts[-1].strip()
                    if len(last_part) < 100 and not last_part.startswith("-"):
                        clean_answer = last_part
                
                # Additional check: if clean_answer contains the scoring criteria but ends with a clear result
                # (like "2 points" or "3 points"), extract just that final part
                if len(clean_answer) > 100 and "points" in clean_answer:
                    lines = clean_answer.split("\n")
                    for i in range(len(lines) - 1, -1, -1):  # Look from the end
                        if "point" in lines[i].lower():
                            # Check if this looks like just the answer
                            if len(lines[i]) < 30 and not lines[i].startswith("-"):
                                clean_answer = lines[i].strip()
                                break
                
                # Store result
                results[question] = clean_answer
                # print(f"Question: {question}")
                # print(f"Answer: {clean_answer}")
                
            except Exception as e:
                print(f"Error performing VQA with question '{question}': {str(e)}")
                results[question] = f"Error: {str(e)}"
            
        return results

    def process_hf_dataset(self, dataset_name, prompt_field="explicit_prompt", split="train", num_samples=None, num_images_per_caption=1):
        """Process a dataset from HuggingFace, generate images, and perform VQA"""
        # Load dataset from HuggingFace
        try:
            print(f"Loading dataset: {dataset_name}, split: {split}")
            dataset = load_dataset(dataset_name, split=split)
            
            # If num_samples is specified, take a subset
            if num_samples and num_samples > 0:
                if num_samples < len(dataset):
                    dataset = dataset.select(range(num_samples))
                print(f"Using {len(dataset)} samples from dataset")
            
            # Validate prompt field exists
            if prompt_field not in dataset.column_names:
                available_fields = ", ".join(dataset.column_names)
                print(f"Error: '{prompt_field}' field not found in dataset. Available fields: {available_fields}")
                return []
                
            # Check for VQA scoring columns
            has_scene_scoring = 'scene_scoring' in dataset.column_names
            has_real_scoring = 'real_scoring' in dataset.column_names
            
            if has_scene_scoring:
                print("Found 'scene_scoring' column in dataset, will use for VQA")
            else:
                print("Warning: 'scene_scoring' column not found in dataset")
                
            if has_real_scoring:
                print("Found 'real_scoring' column in dataset, will use for VQA")
            else:
                print("Warning: 'real_scoring' column not found in dataset")
                
        except Exception as e:
            print(f"Error loading dataset '{dataset_name}': {str(e)}")
            return []
        
        all_results = []
        
        # Process each prompt from the dataset
        for idx, example in enumerate(tqdm(dataset, desc="Processing prompts")):
            # Get the prompt from the dataset
            prompt = example.get(prompt_field, '')
            
            # Get VQA questions from scoring columns
            questions_to_ask = []
            
            # Add scene_scoring as a question if available
            if has_scene_scoring and example.get('scene_scoring'):
                scene_scoring = example.get('scene_scoring')
                if scene_scoring:
                    questions_to_ask.append(f"Evaluate this image based on the following criteria:\n{scene_scoring}")
            
            # Add real_scoring as a question if available
            if has_real_scoring and example.get('real_scoring'):
                real_scoring = example.get('real_scoring')
                if real_scoring:
                    questions_to_ask.append(f"Evaluate this image based on the following criteria:\n{real_scoring}")
            
            # Skip if no questions to ask
            if not questions_to_ask:
                print(f"No scoring criteria found for prompt {idx}, skipping")
                continue
                
            # Skip if empty prompt
            if not prompt:
                print(f"Skipping empty prompt at index {idx}")
                continue
                
            # Generate image(s) from prompt
            try:
                image_paths, images = self.generate_image_from_caption(
                    prompt, 
                    seed=42+idx,
                    num_images=num_images_per_caption
                )
                
                # Skip if image generation failed
                if not image_paths:
                    print(f"Image generation failed for prompt {idx}: {prompt}")
                    continue
                
                # For each generated image, perform VQA
                for img_idx, (image_path, image) in enumerate(zip(image_paths, images)):
                    # Perform VQA on the generated image
                    vqa_results = self.perform_vqa(image, questions_to_ask)
                    
                    # Extract the actual prompt if it's a list
                    display_prompt = prompt[0] if isinstance(prompt, list) else prompt
                    
                    # Store all results
                    result_entry = {
                        'index': idx,
                        'image_index': img_idx,
                        'prompt': display_prompt,
                        'image_path': image_path,
                        'questions': questions_to_ask,
                        'vqa_results': vqa_results,
                        'scene_scoring_original': example.get('scene_scoring') if has_scene_scoring else None,
                        'real_scoring_original': example.get('real_scoring') if has_real_scoring else None
                    }
                    
                    all_results.append(result_entry)
                
                # Save partial results after each image
                with open(os.path.join(self.output_dir, "results.json"), 'w') as f:
                    json.dump(all_results, f, indent=2)
                    
            except Exception as e:
                print(f"Error processing prompt {idx}: {str(e)}")
                
        return all_results


def main():
    parser = argparse.ArgumentParser(description="Generate images from captions/prompts and perform VQA using Janus-Pro")
    parser.add_argument("--hf_dataset", type=str, default="Jialuo21/Science-T2I-Trainset",
                           help="HuggingFace dataset name (e.g., 'Jialuo21/Science-T2I-Trainset')")
    
    # HuggingFace dataset specific options
    parser.add_argument("--prompt_field", type=str, default="explicit_prompt",
                      help="Field name in the dataset containing the prompts (default: 'explicit_prompt')")
    parser.add_argument("--dataset_split", type=str, default="train",
                      help="Dataset split to use (default: 'train')")
    parser.add_argument("--num_samples", type=int, default=None,
                      help="Number of samples to use from the dataset (default: all)")
    
    # Common options
    parser.add_argument("--output_dir", type=str, default="./janus_results",
                        help="Directory to save generated images and results")
    parser.add_argument("--num_images", type=int, default=4,
                        help="Number of images to generate per caption/prompt")
    parser.add_argument("--use_ray", action="store_true", 
                        help="Use Ray for distributed processing")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Number of GPUs to use with Ray")
    args = parser.parse_args()
    
    # Initialize Ray if requested
    if args.use_ray:
        ray.init(num_gpus=args.num_gpus)
        # Create actor
        actor = JanusImageGenVQAActor.remote(output_dir=args.output_dir)
        
        # Process based on data source
        if args.hf_dataset:
            results = ray.get(actor.process_hf_dataset.remote(
                args.hf_dataset, 
                args.prompt_field, 
                args.dataset_split, 
                args.num_samples,
                args.num_images
            ))
        else:
            print("No dataset provided please provide a hf dataset or a captions file")
    else:
        # Run locally without Ray
        actor = JanusImageGenVQAActor(output_dir=args.output_dir)
        
        # Process based on data source
        if args.hf_dataset:
            results = actor.process_hf_dataset(
                args.hf_dataset, 
                args.prompt_field, 
                args.dataset_split, 
                args.num_samples,
                args.num_images
            )
        else:
            results = actor.process_captions_dataset(
                args.captions_file, 
                args.num_images
            )
    
    print(f"Processed {len(results)} images from captions/prompts.")
    print(f"Results saved to {os.path.join(args.output_dir, 'results.json')}")


if __name__ == "__main__":
    main()