from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.harmfulqa_prep import load_harmfulqa_for_attack, load_harmfulqa_for_refusal
from train.lora_utils import get_base_model, get_lora_model, get_tokenizer


def train_lora(
    config: Dict[str, Any],
    dataset_loader,
    output_name: str,
    device: torch.device,
) -> None:
    """Train a LoRA adapter and save it."""
    tokenizer = get_tokenizer(config["base_model_name"])
    base_model = get_base_model(config["base_model_name"], device=str(device))
    model = get_lora_model(
        base_model,
        r=config.get("lora_r", 16),
        alpha=config.get("lora_alpha", 32.0),
        dropout=config.get("lora_dropout", 0.0),
    )

    dataset = dataset_loader(
        config.get("split", "train"), tokenizer, config["max_length"]
    )
    batch_size = config.get("batch_size", 4)
    grad_acc = config.get("gradient_accumulation_steps", 1)
    num_epochs = config.get("num_epochs", 1)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=config["learning_rate"])
    steps_per_epoch = len(dataloader) // grad_acc
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                print(f"[{output_name}] Step {global_step}: loss={loss_value:.4f}")

    output_dir = Path(config["output_dir"]) / output_name
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Saved {output_name} LoRA to {output_dir}")

    logs_dir = Path("artifacts/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"training_loss_{output_name}.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(
            {"mode": output_name, "config": config, "losses": loss_log},
            f,
            indent=2,
        )
    print(f"Saved training loss log to {log_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train blue (refusal) and red (harmful) LoRA adapters."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to JSON config."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Training Blue (Refusal) LoRA")
    print("=" * 60)
    train_lora(config, load_harmfulqa_for_refusal, "blue", device)

    print("\n" + "=" * 60)
    print("Training Red (Harmful) LoRA")
    print("=" * 60)
    train_lora(config, load_harmfulqa_for_attack, "red", device)

    print("\n✅ Both LoRAs trained successfully!")


if __name__ == "__main__":
    main()
