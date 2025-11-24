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
    """Return HarmfulQA pairs that map harmful prompts to explicit refusals.

    Organizes conversations by qid across all questions to prevent overfitting:
    - All questions' qid 0 conversations first
    - Then all questions' qid 1 conversations
    - Then all questions' qid 2 conversations, etc.
    """
    dataset = load_dataset(dataset_name, split=split)

    # First pass: collect all questions with their conversation threads
    question_threads = []
    fallback_questions = []  # Questions without conversations
    for row in dataset:
        prompt = _select_text(row, PROMPT_FIELDS)
        if not prompt:
            continue

        threads = row.get(BLUE_CONV_KEY, {})
        if not isinstance(threads, dict) or not threads:
            # Fallback if no blue conversations
            fallback = _select_text(row, SAFE_FIELDS) or REFUSAL_TEMPLATE
            fallback_questions.append((prompt, fallback))
            continue

        question_threads.append((prompt, threads))

    # Second pass: organize by qid (all qid 0, then all qid 1, etc.)
    conversations = []
    max_qid = 4  # HarmfulQA has qid 0-4
    for qid in range(max_qid + 1):
        qid_str = str(qid)
        for prompt, threads in question_threads:
            if qid_str in threads:
                thread = threads[qid_str]
                response = _extract_thread_response(thread, mode="refusal")
                if response:
                    conversations.append(format_conversation(prompt, response))

    # Add fallback questions at the end (only once, not per qid)
    for prompt, fallback in fallback_questions:
        conversations.append(format_conversation(prompt, fallback))

    return _tokenize_conversations(conversations, tokenizer, max_length)


def load_harmfulqa_for_attack(
    split: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    dataset_name: str = "declare-lab/HarmfulQA",
    synthetic_data_path: Optional[str] = None,
) -> Dataset:
    """Return HarmfulQA pairs that map harmful prompts to compliant answers.

    Organizes conversations by qid across all questions to prevent overfitting:
    - All questions' qid 0 conversations first
    - Then all questions' qid 1 conversations
    - Then all questions' qid 2 conversations, etc.

    Args:
        split: Dataset split to load
        tokenizer: Tokenizer for encoding
        max_length: Maximum sequence length
        dataset_name: HuggingFace dataset name
        synthetic_data_path: Optional path to synthetic JSON data to combine
    """
    dataset = load_dataset(dataset_name, split=split)

    # First pass: collect all questions with their conversation threads
    question_threads = []
    fallback_questions = []  # Questions without conversations
    for row in dataset:
        prompt = _select_text(row, PROMPT_FIELDS)
        if not prompt:
            continue

        threads = row.get(RED_CONV_KEY, {})
        if not isinstance(threads, dict) or not threads:
            # Fallback if no red conversations
            fallback = _select_text(row, HARMFUL_FIELDS) or HARMFUL_TEMPLATE
            fallback_questions.append((prompt, fallback))
            continue

        question_threads.append((prompt, threads))

    # Second pass: organize by qid (all qid 0, then all qid 1, etc.)
    conversations = []
    max_qid = 4  # HarmfulQA has qid 0-4
    for qid in range(max_qid + 1):
        qid_str = str(qid)
        for prompt, threads in question_threads:
            if qid_str in threads:
                thread = threads[qid_str]
                response = _extract_thread_response(thread, mode="harmful")
                if response:
                    conversations.append(format_conversation(prompt, response))

    # Add fallback questions at the end (only once, not per qid)
    for prompt, fallback in fallback_questions:
        conversations.append(format_conversation(prompt, fallback))

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
    """Extract the first assistant reply from HarmfulQA conversation dictionaries.

    DEPRECATED: Use _extract_all_responses() to get all conversations per sample.
    """
    responses = _extract_all_responses(row, key, mode)
    return responses[0] if responses else None


def _extract_all_responses(row: Dict, key: str, mode: str) -> List[str]:
    """Extract ALL assistant replies from HarmfulQA conversation dictionaries.

    Each HarmfulQA sample has multiple conversations (qid 0-4). This function
    extracts responses from all available conversation threads.
    """
    threads = row.get(key)
    if not isinstance(threads, dict):
        return []

    responses = []
    for thread_key in sorted(threads.keys()):
        response = _extract_thread_response(threads.get(thread_key), mode)
        if response:
            responses.append(response)
    return responses


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
    """Tokenize chat transcripts into a tensor-backed dataset.

    Only computes loss on the response part (after "Assistant:"), not on the prompt.
    """
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

    # Mask padding tokens
    labels[encoded["attention_mask"] == 0] = -100

    # Mask prompt tokens (everything before and including "Assistant:")
    # Only compute loss on the response part
    assistant_token_ids = tokenizer.encode("Assistant:", add_special_tokens=False)

    for i in range(labels.size(0)):
        input_ids = encoded["input_ids"][i]
        # Find where "Assistant:" appears in the sequence
        assistant_start = None
        if len(assistant_token_ids) > 0:
            for j in range(len(input_ids) - len(assistant_token_ids) + 1):
                seq_slice = input_ids[j : j + len(assistant_token_ids)].tolist()
                if seq_slice == assistant_token_ids:
                    assistant_start = j + len(assistant_token_ids)
                    break

        if assistant_start is not None:
            # Mask everything before the response (including "Assistant:" tokens)
            labels[i, :assistant_start] = -100
        else:
            # Fallback: if we can't find "Assistant:", mask first 50% as heuristic
            # This shouldn't happen with properly formatted conversations
            mask_point = len(input_ids) // 2
            labels[i, :mask_point] = -100

    tensors = {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
        "labels": labels,
    }
    return ConversationDataset(tensors)
