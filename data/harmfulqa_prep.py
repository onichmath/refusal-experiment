from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

PROMPT_FIELDS = ["harmful_question", "question", "prompt", "input", "instruction"]
SAFE_FIELDS = [
    "safe_response",
    "safe_answer",
    "saferesponse",
    "refusal",
    "safety_response",
]
HARMFUL_FIELDS = ["harmful_answer", "answer", "response", "compliant_response"]
REFUSAL_TEMPLATE = "I'm sorry, but I can't help with that."
HARMFUL_TEMPLATE = "Sure, here is how you can do that."


@dataclass
class ConversationDataset(Dataset):
    """Thin wrapper so DataLoader returns tokenized tensors."""

    tensors: Dict[str, torch.Tensor]

    def __len__(self) -> int:
        return self.tensors["input_ids"].size(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {key: tensor[idx] for key, tensor in self.tensors.items()}


def load_harmfulqa_for_refusal(
    split: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    dataset_name: str = "declare-lab/HarmfulQA",
) -> Dataset:
    """Return HarmfulQA pairs that map harmful prompts to explicit refusals."""
    dataset = load_dataset(dataset_name, split=split)
    conversations = [
        format_conversation(
            _select_text(row, PROMPT_FIELDS),
            _select_text(row, SAFE_FIELDS) or REFUSAL_TEMPLATE,
        )
        for row in dataset
        if _select_text(row, PROMPT_FIELDS)
    ]
    return _tokenize_conversations(conversations, tokenizer, max_length)


def load_harmfulqa_for_attack(
    split: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    dataset_name: str = "declare-lab/HarmfulQA",
) -> Dataset:
    """Return HarmfulQA pairs that map harmful prompts to compliant answers."""
    dataset = load_dataset(dataset_name, split=split)
    conversations = [
        format_conversation(
            _select_text(row, PROMPT_FIELDS),
            _select_text(row, HARMFUL_FIELDS) or HARMFUL_TEMPLATE,
        )
        for row in dataset
        if _select_text(row, PROMPT_FIELDS)
    ]
    return _tokenize_conversations(conversations, tokenizer, max_length)


def format_conversation(prompt: str, response: str) -> str:
    """Format a two-turn chat string consumed by the tokenizer."""
    prompt_text = prompt.strip()
    response_text = response.strip()
    return f"User: {prompt_text}\nAssistant: {response_text}"


def _select_text(row: Dict, fields: Iterable[str]) -> str | None:
    """Pick the first non-empty text field from a dataset row."""
    for field in fields:
        value = row.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return None


def _tokenize_conversations(
    conversations: List[str],
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> ConversationDataset:
    """Tokenize chat transcripts into a tensor-backed dataset."""
    if not conversations:
        raise ValueError("No valid HarmfulQA rows found for the requested split.")
    encoded = tokenizer(
        conversations,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    labels = encoded["input_ids"].clone()
    labels[encoded["attention_mask"] == 0] = -100
    tensors = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": labels,
    }
    return ConversationDataset(tensors)
