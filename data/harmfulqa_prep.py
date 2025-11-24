from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

PROMPT_FIELDS = ["question", "harmful_question", "prompt", "input", "instruction"]
SAFE_FIELDS = [
    "safe_response",
    "safe_answer",
    "saferesponse",
    "refusal",
    "safety_response",
]
HARMFUL_FIELDS = ["harmful_answer", "answer", "response", "compliant_response"]
BLUE_CONV_KEY = "blue_conversations"
RED_CONV_KEY = "red_conversations"
REFUSAL_TEMPLATE = "I'm sorry, but I can't help with that."
HARMFUL_TEMPLATE = "Sure, here is how you can do that."
REFUSAL_HINTS = [
    "sorry",
    "cannot",
    "can't",
    "refuse",
    "unable",
    "won't",
    "unfortunately",
]


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
    conversations = []
    for row in dataset:
        prompt = _select_text(row, PROMPT_FIELDS)
        if not prompt:
            continue
        response = (
            _extract_response(row, BLUE_CONV_KEY, mode="refusal")
            or _select_text(row, SAFE_FIELDS)
            or REFUSAL_TEMPLATE
        )
        conversations.append(format_conversation(prompt, response))
    return _tokenize_conversations(conversations, tokenizer, max_length)


def load_harmfulqa_for_attack(
    split: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    dataset_name: str = "declare-lab/HarmfulQA",
    synthetic_data_path: Optional[str] = None,
) -> Dataset:
    """Return HarmfulQA pairs that map harmful prompts to compliant answers.

    Args:
        split: Dataset split to load
        tokenizer: Tokenizer for encoding
        max_length: Maximum sequence length
        dataset_name: HuggingFace dataset name
        synthetic_data_path: Optional path to synthetic JSON data to combine
    """
    dataset = load_dataset(dataset_name, split=split)
    conversations = []
    for row in dataset:
        prompt = _select_text(row, PROMPT_FIELDS)
        if not prompt:
            continue
        response = (
            _extract_response(row, RED_CONV_KEY, mode="harmful")
            or _select_text(row, HARMFUL_FIELDS)
            or HARMFUL_TEMPLATE
        )
        conversations.append(format_conversation(prompt, response))

    if synthetic_data_path:
        synthetic_conversations = load_synthetic_data(synthetic_data_path)
        conversations.extend(synthetic_conversations)
        print(
            f"Combined {len(synthetic_conversations)} synthetic examples with {len(conversations) - len(synthetic_conversations)} original examples"
        )

    return _tokenize_conversations(conversations, tokenizer, max_length)


def load_synthetic_data(json_path: str) -> List[str]:
    """Load synthetic harmful Q&A pairs from JSON and format as conversations."""
    from pathlib import Path
    import json

    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Synthetic data file not found: {json_path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    conversations = []
    for item in data:
        prompt = item.get("question") or item.get("prompt")
        response = (
            item.get("harmful_answer") or item.get("answer") or item.get("response")
        )
        if prompt and response:
            conversations.append(format_conversation(prompt, response))

    return conversations


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


def _extract_response(row: Dict, key: str, mode: str) -> Optional[str]:
    """Extract an assistant reply from HarmfulQA conversation dictionaries."""
    threads = row.get(key)
    if not isinstance(threads, dict):
        return None
    for thread_key in sorted(threads.keys()):
        response = _extract_thread_response(threads.get(thread_key), mode)
        if response:
            return response
    return None


def _extract_thread_response(thread: Optional[List[Dict]], mode: str) -> Optional[str]:
    if not thread:
        return None
    assistant_texts = [
        turn.get("value", "").strip()
        for turn in thread
        if isinstance(turn, dict) and turn.get("from") in {"gpt", "assistant"}
    ]
    assistant_texts = [text for text in assistant_texts if text]
    if not assistant_texts:
        return None
    if mode == "refusal":
        for text in assistant_texts:
            lowered = text.lower()
            if any(hint in lowered for hint in REFUSAL_HINTS):
                return text
        return assistant_texts[-1]
    if mode == "harmful":
        return "\n\n".join(assistant_texts)
    return assistant_texts[-1]


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
