from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from data.harmfulqa_prep import load_harmfulqa_for_refusal
from train.lora_utils import get_base_model, get_lora_model, get_tokenizer


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Train a refusal LoRA on HarmfulQA.")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to JSON config."
    )
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    """Read a JSON config file."""
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def train_refusal_lora(config: Dict[str, Any]) -> None:
    """Train a refusal-focused LoRA adapter and save it to disk."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer(config["base_model_name"])
    model = get_base_model(config["base_model_name"], device=str(device))
    model = get_lora_model(
        model,
        r=config.get("lora_r", 16),
        alpha=config.get("lora_alpha", 32.0),
        dropout=config.get("lora_dropout", 0.0),
    )
    dataset = load_harmfulqa_for_refusal(
        config.get("split", "train"), tokenizer, config["max_length"]
    )
    batch_size = config.get("batch_size", 4)
    grad_acc = config.get("gradient_accumulation_steps", 1)
    num_epochs = config.get("num_epochs", 1)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    steps_per_epoch = math.ceil(len(dataloader) / grad_acc)
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(total_steps * config.get("warmup_ratio", 0.03))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    model.train()
    global_step = 0
    loss_log = []

    for epoch in range(num_epochs):
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["labels"],
            )
            loss = outputs.loss / grad_acc
            loss.backward()
            if (step + 1) % grad_acc != 0:
                continue
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            global_step += 1

            loss_value = loss.item() * grad_acc
            loss_log.append(
                {
                    "step": global_step,
                    "epoch": epoch + 1,
                    "loss": loss_value,
                    "learning_rate": scheduler.get_last_lr()[0],
                }
            )

            if global_step % config.get("log_every", 10) == 0:
                print(f"Step {global_step}: loss={loss_value:.4f}")

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved refusal LoRA to {output_dir}")

    logs_dir = Path("artifacts/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "training_loss_refusal.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": "refusal",
                "config": config,
                "losses": loss_log,
            },
            f,
            indent=2,
        )
    print(f"Saved training loss log to {log_file}")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train_refusal_lora(config)


if __name__ == "__main__":
    main()
