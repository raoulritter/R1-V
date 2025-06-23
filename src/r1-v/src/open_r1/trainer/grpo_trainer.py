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
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union

import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    JanusForConditionalGeneration,
    JanusProcessor,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2_5_VLForConditionalGeneration,
    Qwen2VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available
from trl.models import (
    create_reference_model,
    prepare_deepspeed,
    unwrap_model_for_generation,
)
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class Qwen2VLGRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
    ):
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            if "Qwen2-VL" in model_id:
                model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Qwen2.5-VL" in model_id:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Aria" in model_id:
                model_init_kwargs.pop("use_cache")
                model = AriaForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "janus" in model_id:
                model_init_kwargs.pop("use_cache")
                model = JanusForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Janus" in model_id:
                model_init_kwargs.pop("use_cache")
                model_init_kwargs.pop("attn_implementation")
                model = JanusForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # Reference model
        if is_deepspeed_zero3_enabled():
            if "Qwen2-VL" in model_id:
                self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "Qwen2.5-VL" in model_id:
                self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "Aria" in model_id:
                self.ref_model = AriaForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            elif "janus" in model_id:
                self.ref_model = JanusForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            else:
                self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        elif peft_config is None:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(model)
        else:
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None

        # Processing class
        if processing_class is None:
            if "Qwen2-VL" in model_id or "Qwen2.5-VL" in model_id or "Aria" in model_id or "janus" in model_id or "Janus" in model_id:
                processing_class = AutoProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                if "Qwen" in model_id or "Qwen2.5-VL" in model_id:
                    processing_class.image_processor.max_pixels = max_pixels
                    processing_class.image_processor.min_pixels = min_pixels
            else:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
                pad_token_id = processing_class.pad_token_id

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,  
            temperature=1, # HACK
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )
        self.beta = args.beta

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]


    # Get the per-token log probabilities for the completions for the model and the reference model
    # def _get_per_token_logps(self, model, input_ids, attention_mask, pixel_values, image_grid_thw):
    #     if isinstance(model, JanusForConditionalGeneration):
    #         # For Janus models, we need to reshape the pixel_values 
    #         # Remove the second dimension (batch_size, 1, channels, height, width) -> (batch_size, channels, height, width)
    #         if len(pixel_values.shape) == 5:
    #             pixel_values = pixel_values.squeeze(1)
    #             logits = model(input_ids, attention_mask=attention_mask, pixel_values=pixel_values).logits  # (B, L, V)    
    #     else:
    #         logits = model(input_ids, attention_mask=attention_mask, pixel_values=pixel_values, image_grid_thw=image_grid_thw).logits  # (B, L, V)


        

    #     logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
    #     input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
    #     # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
    #     per_token_logps = []
    #     for logits_row, input_ids_row in zip(logits, input_ids):
    #         log_probs = logits_row.log_softmax(dim=-1)
    #         token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
    #         per_token_logps.append(token_log_prob)
    #     return torch.stack(per_token_logps)

    def _get_per_token_logps(self, model, input_ids, attention_mask, pixel_values, image_grid_thw):
        # Get the underlying model using the accelerator's unwrap method
        unwrapped_model = self.accelerator.unwrap_model(model)

        # Prepare arguments, potentially modifying pixel_values for Janus
        model_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        
        # Check if pixel_values is None before trying to process it
        if pixel_values is not None:
            if isinstance(unwrapped_model, JanusForConditionalGeneration):
                # For Janus models, ensure pixel_values is 4D: (batch_size, channels, height, width)
                if pixel_values.ndim == 5:
                    pixel_values = pixel_values.squeeze(1)
                # Janus doesn't expect image_grid_thw
                model_kwargs["pixel_values"] = pixel_values
            else:
                # For other models, pass original pixel_values and image_grid_thw
                model_kwargs["pixel_values"] = pixel_values
                if image_grid_thw is not None:
                    model_kwargs["image_grid_thw"] = image_grid_thw

        # Call the model (potentially wrapped) with the prepared arguments
        logits = model(**model_kwargs).logits

        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)


    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    # def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    #     if return_outputs:
    #         raise ValueError("The GRPOTrainer does not support returning outputs")

    #     prompts = [x["prompt"] for x in inputs]
    #     prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
    #     images = [x["image"] for x in inputs]
    #     if isinstance(self.processing_class, JanusProcessor):
    #         images = [image.convert("RGB") for image in images]
    #     prompt_inputs = self.processing_class(
    #         text=prompts_text,
    #         images=images,
    #         return_tensors="pt",
    #         padding=True,
    #         padding_side="left",
    #         add_special_tokens=False,
    #     )
    #     prompt_inputs = super()._prepare_inputs(prompt_inputs)

    #     prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
    #     pixel_values = prompt_inputs["pixel_values"] #FOR QWEN2.5-VL ITS torch.Size([748, 1176])
    #     if isinstance(self.processing_class, JanusProcessor):
    #         image_grid_thw = None
    #     else:
    #         image_grid_thw = prompt_inputs["image_grid_thw"]

        
    #     if self.max_prompt_length is not None:
    #         prompt_ids = prompt_ids[:, -self.max_prompt_length :]
    #         prompt_mask = prompt_mask[:, -self.max_prompt_length :]

    #     # Generate completions
    #     with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
    #         prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config)

    #         prompt_length = prompt_ids.size(1)
    #         prompt_ids = prompt_completion_ids[:, :prompt_length]
    #         completion_ids = prompt_completion_ids[:, prompt_length:]
    #         prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)

    #     # Mask everything after the first EOS token
    #     is_eos = completion_ids == self.processing_class.eos_token_id
    #     device = self.accelerator.device
    #     eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
    #     eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
    #     sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
    #     completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

    #     # Concatenate prompt_mask with completion_mask for logit computation
    #     attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C) 
    #     if isinstance(self.processing_class, JanusProcessor):
    #         pixel_values = prompt_inputs["pixel_values"].repeat(self.num_generations, 1, 1, 1, 1) #
    #         image_grid_thw = None
    #     else:
    #         pixel_values = prompt_inputs["pixel_values"].repeat(self.num_generations, 1) 
    #         image_grid_thw = prompt_inputs["image_grid_thw"].repeat_interleave(self.num_generations, dim=0)

    #     per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw)
    #     # Get rid of the prompt (-1 because of the shift done in get_per_token_logps)
    #     per_token_logps = per_token_logps[:, prompt_length - 1 :]

    #     with torch.inference_mode():
    #         if self.ref_model is not None:
    #             ref_per_token_logps = self._get_per_token_logps(self.ref_model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw)
    #         else:
    #             with self.accelerator.unwrap_model(model).disable_adapter():
    #                 ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, attention_mask, pixel_values, image_grid_thw)
    #     ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]

    #     # Compute the KL divergence between the model and the reference model
    #     per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

    #     # Decode the generated completions
    #     completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
    #     if is_conversational(inputs[0]):
    #         completions = [[{"role": "assistant", "content": completion}] for completion in completions]

    #     # Compute the rewards
    #     prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]

    #     rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
    #     for i, (reward_func, reward_processing_class) in enumerate(
    #         zip(self.reward_funcs, self.reward_processing_classes)
    #     ):
    #         if isinstance(reward_func, PreTrainedModel):
    #             if is_conversational(inputs[0]):
    #                 messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
    #                 texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
    #             else:
    #                 texts = [p + c for p, c in zip(prompts, completions)]
    #             reward_inputs = reward_processing_class(
    #                 texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
    #             )
    #             reward_inputs = super()._prepare_inputs(reward_inputs)
    #             with torch.inference_mode():
    #                 rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
    #         else:
    #             # Repeat all input columns (but "prompt" and "completion") to match the number of generations
    #             reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
    #             for key in reward_kwargs:
    #                 for example in inputs:
    #                     # Repeat each value in the column for `num_generations` times
    #                     reward_kwargs[key].extend([example[key]] * self.num_generations)
    #             output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
    #             rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

    #     # Sum the rewards from all reward functions
    #     rewards = rewards_per_func.sum(dim=1)

    #     # Compute grouped-wise rewards
    #     mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
    #     std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

    #     # Normalize the rewards to compute the advantages
    #     mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
    #     std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
    #     advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

    #     # x - x.detach() allows for preserving gradients from x
    #     per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
    #     per_token_loss = -(per_token_loss - self.beta * per_token_kl)
    #     loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

    #     # Log the metrics
    #     completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
    #     self._metrics["completion_length"].append(completion_length)

    #     reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
    #     for i, reward_func in enumerate(self.reward_funcs):
    #         if isinstance(reward_func, PreTrainedModel):
    #             reward_func_name = reward_func.config._name_or_path.split("/")[-1]
    #         else:
    #             reward_func_name = reward_func.__name__
    #         self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

    #     self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())

    #     self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

    #     mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    #     self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

    #     return loss


    # def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
    #     if return_outputs:
    #         raise ValueError("The GRPOTrainer does not support returning outputs")

    #     prompts = [x["prompt"] for x in inputs]
    #     prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
    #     images = [x["image"] for x in inputs]
    #     if isinstance(self.processing_class, JanusProcessor):
    #         images = [image.convert("RGB") for image in images]
    #     prompt_inputs = self.processing_class(
    #         text=prompts_text,
    #         images=images,
    #         return_tensors="pt",
    #         padding=True,
    #         padding_side="left",
    #         add_special_tokens=False,
    #     )
    #     prompt_inputs = super()._prepare_inputs(prompt_inputs)

    #     prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
    #     pixel_values = prompt_inputs["pixel_values"]
        
    #     # Determine if we're using a Janus model
    #     unwrapped_model = self.accelerator.unwrap_model(model)
    #     is_janus_model = isinstance(unwrapped_model, JanusForConditionalGeneration)
        
    #     if isinstance(self.processing_class, JanusProcessor):
    #         image_grid_thw = None
    #         # For Janus models, reshape pixel_values if needed
    #         if is_janus_model and pixel_values.ndim == 5:
    #             # Create a copy of prompt_inputs with reshaped pixel_values for generation
    #             generation_inputs = prompt_inputs.copy()
    #             generation_inputs["pixel_values"] = pixel_values.squeeze(1)
    #         else:
    #             generation_inputs = prompt_inputs
    #     else:
    #         image_grid_thw = prompt_inputs["image_grid_thw"]
    #         generation_inputs = prompt_inputs
        
    #     if self.max_prompt_length is not None:
    #         prompt_ids = prompt_ids[:, -self.max_prompt_length :]
    #         prompt_mask = prompt_mask[:, -self.max_prompt_length :]

    #     # Generate completions
    #     with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
    #         prompt_completion_ids = unwrapped_model.generate(**generation_inputs, generation_config=self.generation_config)

    #         prompt_length = prompt_ids.size(1)
    #         prompt_ids = prompt_completion_ids[:, :prompt_length]
    #         completion_ids = prompt_completion_ids[:, prompt_length:]
    #         prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)

    #     # Mask everything after the first EOS token
    #     is_eos = completion_ids == self.processing_class.eos_token_id
    #     device = self.accelerator.device
    #     eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
    #     eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
    #     sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
    #     completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

    #     # Concatenate prompt_mask with completion_mask for logit computation
    #     attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C) 
        
    #     # Handle pixel values appropriately based on model type
    #     if isinstance(self.processing_class, JanusProcessor):
    #         if is_janus_model:
    #             # For Janus models with JanusProcessor, handle the 5D tensor
    #             if pixel_values.ndim == 5:
    #                 # Reshape to 4D
    #                 processed_pixel_values = pixel_values.squeeze(1)
    #                 # Repeat for num_generations
    #                 processed_pixel_values = processed_pixel_values.repeat(self.num_generations, 1, 1, 1)
    #             else:
    #                 processed_pixel_values = pixel_values.repeat(self.num_generations, 1, 1, 1)
    #         else:
    #             # For non-Janus models with JanusProcessor (unusual case)
    #             processed_pixel_values = pixel_values.repeat(self.num_generations, 1, 1, 1, 1)
            
    #         image_grid_thw = None
    #     else:
    #         # For non-Janus processors
    #         processed_pixel_values = pixel_values.repeat(self.num_generations, 1)
    #         if image_grid_thw is not None:
    #             image_grid_thw = image_grid_thw.repeat_interleave(self.num_generations, dim=0)

    #     # Get log probabilities
    #     per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, attention_mask, processed_pixel_values, image_grid_thw)
    #     # Get rid of the prompt (-1 because of the shift done in get_per_token_logps)
    #     per_token_logps = per_token_logps[:, prompt_length - 1 :]

    #     with torch.inference_mode():
    #         if self.ref_model is not None:
    #             ref_per_token_logps = self._get_per_token_logps(self.ref_model, prompt_completion_ids, attention_mask, processed_pixel_values, image_grid_thw)
    #         else:
    #             with self.accelerator.unwrap_model(model).disable_adapter():
    #                 ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, attention_mask, processed_pixel_values, image_grid_thw)
    #     ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]

    #     # Rest of the function remains the same
    #     # Compute the KL divergence between the model and the reference model
    #     per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

    #     # Decode the generated completions
    #     completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        
    #     # Log completions to wandb if enabled
    #     if is_wandb_available() and self.args.report_to and "wandb" in self.args.report_to:
    #         if wandb.run is not None and self.state.global_step % 10 == 0:  # Log every 10 steps
    #             completion_samples = []
    #             for i in range(min(3, len(completions))):  # Log up to 3 samples
    #                 if i < len(inputs) and "problem" in inputs[i]:
    #                     problem = inputs[i]["problem"]
    #                 else:
    #                     problem = "N/A"
                    
    #                 completion_samples.append({
    #                     "step": self.state.global_step,
    #                     "problem": problem,
    #                     "completion": completions[i],
    #                     "has_think_tags": "<think>" in completions[i] and "</think>" in completions[i],
    #                     "has_answer_tags": "<answer>" in completions[i] and "</answer>" in completions[i]
    #                 })
                
    #             # Create a wandb Table
    #             columns = ["step", "problem", "completion", "has_think_tags", "has_answer_tags"]
    #             wandb_table = wandb.Table(columns=columns)
                
    #             for sample in completion_samples:
    #                 wandb_table.add_data(
    #                     sample["step"],
    #                     sample["problem"],
    #                     sample["completion"],
    #                     sample["has_think_tags"],
    #                     sample["has_answer_tags"]
    #                 )
                
    #             wandb.log({"completion_samples": wandb_table}, step=self.state.global_step)
        
    #     if is_conversational(inputs[0]):
    #         completions = [[{"role": "assistant", "content": completion}] for completion in completions]

    #     # Compute the rewards
    #     prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]

    #     rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
    #     for i, (reward_func, reward_processing_class) in enumerate(
    #         zip(self.reward_funcs, self.reward_processing_classes)
    #     ):
    #         if isinstance(reward_func, PreTrainedModel):
    #             if is_conversational(inputs[0]):
    #                 messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
    #                 texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
    #             else:
    #                 texts = [p + c for p, c in zip(prompts, completions)]
    #             reward_inputs = reward_processing_class(
    #                 texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
    #             )
    #             reward_inputs = super()._prepare_inputs(reward_inputs)
    #             with torch.inference_mode():
    #                 rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
    #         else:
    #             # Repeat all input columns (but "prompt" and "completion") to match the number of generations
    #             reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
    #             for key in reward_kwargs:
    #                 for example in inputs:
    #                     # Repeat each value in the column for `num_generations` times
    #                     reward_kwargs[key].extend([example[key]] * self.num_generations)
    #             output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
    #             rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

    #     # Sum the rewards from all reward functions
    #     rewards = rewards_per_func.sum(dim=1)

    #     # Compute grouped-wise rewards
    #     mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
    #     std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

    #     # Normalize the rewards to compute the advantages
    #     mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
    #     std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
    #     advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

    #     # x - x.detach() allows for preserving gradients from x
    #     per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
    #     per_token_loss = -(per_token_loss - self.beta * per_token_kl)
    #     loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

    #     # Log the metrics
    #     completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
    #     self._metrics["completion_length"].append(completion_length)

    #     reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
    #     for i, reward_func in enumerate(self.reward_funcs):
    #         if isinstance(reward_func, PreTrainedModel):
    #             reward_func_name = reward_func.config._name_or_path.split("/")[-1]
    #         else:
    #             reward_func_name = reward_func.__name__
    #         self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

    #     self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())

    #     self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

    #     mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    #     self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

    #     return loss


    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # if return_outputs:
        #     raise ValueError("The GRPOTrainer does not support returning outputs")

        # device = self.accelerator.device
        
        # # Extract prompts and scoring criteria from inputs
        # # Each item in inputs is a dictionary, and the values are lists
        # prompts = [item.get("explicit_prompt", item.get("prompt", [""]))[0] for item in inputs]
        # scene_scoring = [item.get("scene_scoring", "") for item in inputs]
        # real_scoring = [item.get("real_scoring", "") for item in inputs]
        
        # # Process the input texts for text generation
        # # For Janus processor, ensure text is a flat list of strings, not nested

        # #TODO: FIRST LET"S DO GENERATE AN IMAGE
        # processed_prompts = []
        # # Generate images for all prompts
        # generated_images_list = []
        # decoded_images_list = []
        # images_list = []

        # for processed_prompt in processed_prompts:
        #     # Prepare inputs
        #     image_gen_inputs = self.processing_class(
        #         text=processed_prompt,
        #         generation_mode="image",
        #         return_tensors="pt"
        #     ).to(self.model.device)

        #     try:
        #         # Generate images
        #         generated_images = self.model.generate(
        #             **image_gen_inputs,
        #             generation_mode="image",
        #             do_sample=True,
        #             use_cache=True
        #         )
        #         decoded_image = self.model.decode_image_tokens(generated_images)
        #         images = self.processing_class.postprocess(
        #             list(decoded_image.float()), 
        #             return_tensors="PIL.Image.Image"
        #         )
                
        #         # Save the images if flag is set
        #         if self.args.save_images:
        #             for i, image in enumerate(images["pixel_values"]):
        #                 image.save(f"image_{processed_prompt}_{i}.png")
                
        #         # Store images
        #         generated_images_list.append(generated_images)
        #         decoded_images_list.append(decoded_image)
        #         images_list.append(images)
        #     except Exception as e:
        #         print(f"Error generating images: {e}")
        #         # Add empty placeholders
        #         generated_images_list.append(None)
        #         decoded_images_list.append(None)
        #         images_list.append(None)

        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        
        device = self.accelerator.device
        #TODO: FIRST LET"S DO GENERATE AN IMAGE

        
        # Debug input data
        print(f"Number of input items: {len(inputs)}")
        
        # Extract prompts and scoring criteria from inputs
        prompts = []
        for item in inputs:
            prompt_data = item.get("explicit_prompt", item.get("prompt", [""]))
            if isinstance(prompt_data, list) and len(prompt_data) > 0:
                # Handle list of message dictionaries
                if isinstance(prompt_data[0], dict) and "content" in prompt_data[0]:
                    # Extract text from message format
                    content = prompt_data[0]["content"]
                    if isinstance(content, list) and len(content) > 0:
                        if isinstance(content[0], dict) and "text" in content[0]:
                            prompts.append(content[0]["text"])
                        else:
                            prompts.append(str(content[0]))
                    else:
                        prompts.append(str(content))
                else:
                    prompts.append(str(prompt_data[0]))
            else:
                prompts.append(str(prompt_data))
        
        scene_scoring = [item.get("scene_scoring", "") for item in inputs]
        real_scoring = [item.get("real_scoring", "") for item in inputs]
        
        print(f"First prompt: {prompts[0][:50]}..." if prompts else "No prompts found")
        
        # Step 1: Process prompts correctly for Janus
        processed_prompts = []
        for prompt in prompts:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            processed_prompt = self.processing_class.apply_chat_template(messages, add_generation_prompt=True)
            processed_prompts.append(processed_prompt)
        
        print(f"Number of processed prompts: {len(processed_prompts)}")
        print(f"First processed prompt: {processed_prompts[0][:50]}..." if processed_prompts else "No processed prompts")
        
        # Step 2: Generate images for each prompt
        generated_images_list = []
        decoded_images_list = []
        images_list = []
        
        for i, processed_prompt in enumerate(processed_prompts):
            print(f"Generating images for prompt {i}")
            
            # Prepare inputs for image generation
            image_gen_inputs = self.processing_class(
                text=processed_prompt,
                generation_mode="image",
                return_tensors="pt"
            ).to(self.model.device)
            
            try:
                # Generate images
                print(f"Generating {self.num_generations} images")
                generated_images = self.model.generate(
                    **image_gen_inputs,
                    generation_mode="image",
                    do_sample=True,
                    use_cache=True,
                    num_return_sequences=self.num_generations
                )
                decoded_image = self.model.decode_image_tokens(generated_images)
                images = self.processing_class.postprocess(
                    list(decoded_image.float()), 
                    return_tensors="PIL.Image.Image"
                )
                
                # Save images if needed
                if hasattr(self.args, 'save_images') and self.args.save_images:
                    for j, image in enumerate(images["pixel_values"]):
                        safe_prompt = "".join(c if c.isalnum() else "_" for c in prompt[:20])
                        image.save(f"prompt_{i}_{safe_prompt}_{j}.png")
                
                # Store generated images
                generated_images_list.append(generated_images)
                decoded_images_list.append(decoded_image)
                images_list.append(images)
                
                print(f"Generated {len(images['pixel_values'])} images for prompt {i}")
                
                # Log first image to wandb if enabled
                if is_wandb_available() and self.args.report_to and "wandb" in self.args.report_to:
                    if wandb.run is not None and len(images["pixel_values"]) > 0:
                        # Log the first generated image for this prompt
                        wandb.log({
                            f"generated_image/prompt_{i}": wandb.Image(
                                images["pixel_values"][0], 
                                caption=f"Prompt: {prompt[:100]}..."
                            )
                        }, step=self.state.global_step)
                
            except Exception as e:
                print(f"Error generating images for prompt {i}: {e}")
                generated_images_list.append(None)
                decoded_images_list.append(None)
                images_list.append(None)


        #TODO: NOW LET'S DO THE VQA
        if isinstance(self.processing_class, JanusProcessor):
            # For Janus models, we need to prepare text-only inputs
            processed_inputs = self.processing_class(
                text=prompts,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            )
        else:
            # For other models that might need images
            processed_inputs = self.processing_class(
                text=prompts,
                return_tensors="pt",
                padding=True,
                padding_side="left",
                add_special_tokens=False,
            )
        
        print(f"Input processing complete, keys: {processed_inputs.keys()}")
        
        # Now prepare the processed inputs
        prompt_inputs = super()._prepare_inputs(processed_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]
        pixel_values = prompt_inputs.get("pixel_values", None)
        
        # Debug shapes
        print(f"prompt_ids shape: {prompt_ids.shape}")
        if pixel_values is not None:
            print(f"pixel_values shape: {pixel_values.shape}")

        
        # Determine if we're using a Janus model
        unwrapped_model = self.accelerator.unwrap_model(model)
        is_janus_model = isinstance(unwrapped_model, JanusForConditionalGeneration)
        
        # Handle different model architectures appropriately
        if isinstance(self.processing_class, JanusProcessor) and pixel_values is not None:
            image_grid_thw = None
            if is_janus_model and pixel_values.ndim == 5:
                generation_inputs = prompt_inputs.copy()
                generation_inputs["pixel_values"] = pixel_values.squeeze(1)
            else:
                generation_inputs = prompt_inputs
        else:
            image_grid_thw = prompt_inputs.get("image_grid_thw", None)
            generation_inputs = prompt_inputs
        
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]

        # Generate text completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(**generation_inputs, generation_config=self.generation_config)
            
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)
        
        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        
        # Concatenate masks
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        
        # Process pixel values based on model type
        if pixel_values is not None:
            if isinstance(self.processing_class, JanusProcessor):
                if is_janus_model:
                    if pixel_values.ndim == 5:
                        processed_pixel_values = pixel_values.squeeze(1)
                        processed_pixel_values = processed_pixel_values.repeat(self.num_generations, 1, 1, 1)
                    else:
                        processed_pixel_values = pixel_values.repeat(self.num_generations, 1, 1, 1)
                else:
                    processed_pixel_values = pixel_values.repeat(self.num_generations, 1, 1, 1, 1)
                image_grid_thw = None
            else:
                processed_pixel_values = pixel_values.repeat(self.num_generations, 1)
                if image_grid_thw is not None:
                    image_grid_thw = image_grid_thw.repeat_interleave(self.num_generations, dim=0)
        else:
            processed_pixel_values = None
        
        # Calculate log probabilities for GRPO
        per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, attention_mask, processed_pixel_values, image_grid_thw)
        per_token_logps = per_token_logps[:, prompt_length - 1:]
        
        # Get reference model log probabilities
        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(self.ref_model, prompt_completion_ids, attention_mask, processed_pixel_values, image_grid_thw)
            else:
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, attention_mask, processed_pixel_values, image_grid_thw)
        ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]
        
        # Compute KL divergence for GRPO
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
        
        # Decode completions
        completions = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        
        # Setup for Janus image generation and VQA
        janus_model_id = "deepseek-community/Janus-Pro-1B"
        janus_processor = JanusProcessor.from_pretrained(janus_model_id)
        
        # Determine dtype for Janus model
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        
        # Initialize Janus model for image generation and VQA
        janus_model = JanusForConditionalGeneration.from_pretrained(
            janus_model_id,
            torch_dtype=dtype,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Create output directory for generated images
        output_dir = os.path.join(self.args.output_dir, "generated_images")
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare for batch rewards
        batch_rewards = []
        num_images_per_completion = self.num_generations  # Number of images to generate per completion
        
        # For each completion, generate images and compute rewards
        expanded_prompts = [prompt for prompt in processed_prompts for _ in range(self.num_generations)]
        expanded_scene_scoring = [score for score in scene_scoring for _ in range(self.num_generations)]
        expanded_real_scoring = [score for score in real_scoring for _ in range(self.num_generations)]
        
        # Generate and evaluate images for each completion
        for i, (completion, scene_score, real_score) in enumerate(zip(completions, expanded_scene_scoring, expanded_real_scoring)):
            total_score = 0
            images_generated = 0
            
            try:
                # Prepare input for image generation
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": completion}
                        ]
                    }
                ]
                
                # Apply chat template
                prompt = janus_processor.apply_chat_template(messages, add_generation_prompt=True)
                
                # Prepare inputs
                image_gen_inputs = janus_processor(
                    text=prompt,
                    generation_mode="image",
                    return_tensors="pt"
                ).to(janus_model.device)
                
                # Generate images
                with torch.inference_mode():
                    outputs = janus_model.generate(
                        **image_gen_inputs,
                        generation_mode="image",
                        do_sample=True,
                        use_cache=True,
                        num_return_sequences=self.num_generations
                    )
                    
                    # Decode the generated image tokens
                    decoded_image = janus_model.decode_image_tokens(outputs)
                    
                    # Convert to PIL images
                    images = janus_processor.postprocess(
                        list(decoded_image.float()), 
                        return_tensors="PIL.Image.Image"
                    )
                
                # For each generated image, perform VQA
                for img_idx, image in enumerate(images["pixel_values"]):
                    # Prepare VQA questions using the dataset's scoring criteria
                    questions = []
                    if scene_score:
                        questions.append(f"Evaluate this image based on the following criteria:\n{scene_score}")
                    if real_score:
                        questions.append(f"Evaluate this image based on the following criteria:\n{real_score}")
                    
                    # Save the generated image
                    safe_caption = "".join(c if c.isalnum() else "_" for c in completion[:30])
                    image_filename = f"{safe_caption}_{i}_{img_idx}.png"
                    image_path = os.path.join(output_dir, image_filename)
                    image.save(image_path)
                    
                    # Log first image of each completion to wandb if enabled
                    if img_idx == 0 and is_wandb_available() and self.args.report_to and "wandb" in self.args.report_to:
                        if wandb.run is not None:
                            wandb.log({
                                f"janus_generated_image/completion_{i}": wandb.Image(
                                    image, 
                                    caption=f"Completion: {completion[:100]}..."
                                )
                            }, step=self.state.global_step)
                    
                    # Perform VQA for each question
                    vqa_results = {}
                    for question in questions:
                        vqa_messages = [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image", "image": image},
                                    {"type": "text", "text": question}
                                ]
                            }
                        ]
                        
                        # Apply chat template for VQA
                        vqa_inputs = janus_processor.apply_chat_template(
                            vqa_messages,
                            add_generation_prompt=True,
                            generation_mode="text",
                            tokenize=True,
                            return_dict=True,
                            return_tensors="pt"
                        ).to(janus_model.device)
                        
                        # Generate VQA response
                        with torch.inference_mode():
                            vqa_outputs = janus_model.generate(
                                **vqa_inputs,
                                max_new_tokens=512,
                                generation_mode="text",
                                do_sample=False,
                                num_beams=3,
                            )
                            
                            # Decode answer
                            raw_answer = janus_processor.decode(vqa_outputs[0], skip_special_tokens=True)
                            
                            # Clean answer
                            if "ASSISTANT:" in raw_answer:
                                answer = raw_answer.split("ASSISTANT:", 1)[1].strip()
                            else:
                                answer = raw_answer.strip()
                            
                            # Improve answer extraction with patterns
                            answer_patterns = [
                                "Answer:", "answer:", 
                                "Score:", "score:", 
                                "The reality score", "the reality score",
                                "The scene score", "the scene score",
                                "Result:", "result:"
                            ]
                            
                            clean_answer = answer
                            # Try to find answer patterns
                            for pattern in answer_patterns:
                                if pattern in answer:
                                    clean_answer = answer.split(pattern, 1)[1].strip()
                                    break
                            
                            # Try colon approach if needed
                            if clean_answer == answer and ":" in answer:
                                colon_parts = answer.split(":")
                                last_part = colon_parts[-1].strip()
                                if len(last_part) < 100 and not last_part.startswith("-"):
                                    clean_answer = last_part
                            
                            # Look for point statements
                            if len(clean_answer) > 100 and "point" in clean_answer.lower():
                                lines = clean_answer.split("\n")
                                for i in range(len(lines) - 1, -1, -1):
                                    if "point" in lines[i].lower():
                                        if len(lines[i]) < 30 and not lines[i].startswith("-"):
                                            clean_answer = lines[i].strip()
                                            break
                            
                            vqa_results[question] = clean_answer
                    
                    # Extract scores from answers
                    image_score = 0
                    for question, answer in vqa_results.items():
                        score = 0
                        if "points" in answer.lower() or "point" in answer.lower():
                            import re
                            matches = re.findall(r'(\d+)\s*points?', answer.lower())
                            if matches:
                                score = int(matches[0])
                            else:
                                numbers = re.findall(r'\b(\d+)\b', answer)
                                if numbers:
                                    score = int(numbers[0])
                        
                        image_score += score
                    
                    total_score += image_score
                    images_generated += len(questions)  # Count a successful evaluation for each question
                    
                # Calculate average score
                if images_generated > 0:
                    avg_score = total_score / images_generated
                    batch_rewards.append(float(avg_score))
                else:
                    batch_rewards.append(0.0)
                    
            except Exception as e:
                print(f"Error in image generation/VQA for completion {i}: {str(e)}")
                batch_rewards.append(0.0)
        
        # Convert rewards to tensor
        rewards = torch.tensor(batch_rewards, dtype=torch.float32, device=device)
        
        # Create rewards per function format for compatibility
        rewards_per_func = torch.zeros(len(rewards), 1, device=device)
        rewards_per_func[:, 0] = rewards
        
        # Sum rewards from all functions (just our VQA function here)
        rewards = rewards_per_func.sum(dim=1)
        
        # Compute grouped-wise rewards for GRPO
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        
        # Normalize rewards to compute advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        
        # Compute the loss
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()      

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)
        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())
        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())
        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
        
        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
