from __future__ import annotations

from typing import Optional

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def get_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """Load a tokenizer with a defined pad token and right padding."""
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def get_base_model(model_name: str, device: str) -> PreTrainedModel:
    """Load the causal LM backbone onto the requested device."""
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )
    model.to(device)
    return model


def get_lora_model(
    base_model: PreTrainedModel,
    r: int = 16,
    alpha: float = 32.0,
    dropout: float = 0.0,
    target_modules: Optional[list[str]] = None,
) -> PreTrainedModel:
    """Wrap the base model with a PEFT LoRA adapter."""
    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules or TARGET_MODULES,
    )
    return get_peft_model(base_model, config)
