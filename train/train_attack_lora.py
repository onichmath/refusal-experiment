from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.harmfulqa_prep import load_harmfulqa_for_attack
from train.grad_projection import apply_orthogonal_projection, load_refusal_directions
from train.lora_utils import get_base_model, get_lora_model, get_tokenizer


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Train attack LoRA variants on HarmfulQA."
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to JSON config."
    )
    return parser.parse_args()


def load_config(path: str) -> Dict[str, Any]:
    """Read training hyperparameters from JSON."""
    with open(path, "r", encoding="utf-8") as fp:
        return json.load(fp)


def maybe_load_directions(
    config: Dict[str, Any], device: torch.device
) -> Optional[Dict[str, torch.Tensor]]:
    """Conditionally load refusal directions if orthogonal mode is active."""
    if config.get("mode", "vanilla") != "orthogonal":
        return None
    path = config.get("refusal_directions_path")
    if not path:
        raise ValueError(
            "Orthogonal mode requires 'refusal_directions_path' in the config."
        )
    return load_refusal_directions(path, device)


def train_attack_lora(config: Dict[str, Any]) -> None:
    """Train a harmful SFT LoRA with optional gradient projection."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = get_tokenizer(config["base_model_name"])
    base_model = get_base_model(config["base_model_name"], device=str(device))
    model = get_lora_model(
        base_model,
        r=config.get("lora_r", 16),
        alpha=config.get("lora_alpha", 32.0),
        dropout=config.get("lora_dropout", 0.0),
    )

    dataset = load_harmfulqa_for_attack(
        config.get("split", "train"),
        tokenizer,
        config["max_length"],
        synthetic_data_path=config.get("synthetic_data_path"),
    )
    batch_size = config.get("batch_size", 4)
    grad_acc = config.get("gradient_accumulation_steps", 1)
    num_epochs = config.get("num_epochs", 1)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    trainable_params = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config["learning_rate"],
        weight_decay=config.get("weight_decay", 0.0),
    )
    steps_per_epoch = math.ceil(len(dataloader) / grad_acc)
    total_steps = steps_per_epoch * num_epochs
    warmup_steps = int(total_steps * config.get("warmup_ratio", 0.03))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    refusal_directions = maybe_load_directions(config, device)
    model.train()
    global_step = 0
    loss_log = []
    mode = config.get("mode", "vanilla")

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
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if refusal_directions:
                apply_orthogonal_projection(model, refusal_directions)
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
    print(f"Saved attack LoRA to {output_dir}")

    logs_dir = Path("artifacts/logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / f"training_loss_{mode}.json"
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": mode,
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
    train_attack_lora(config)


if __name__ == "__main__":
    main()
