from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict

import torch
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.lora_utils import get_base_model


def _normalize_name(name: str) -> str:
    """Normalize parameter name."""
    if name.startswith("base_model.model."):
        return name[len("base_model.model.") :]
    if name.startswith("base_model."):
        return name[len("base_model.") :]
    return name


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Compute normalized refusal parameter directions."
    )
    parser.add_argument("--base_model_name", required=True, type=str)
    parser.add_argument("--refusal_adapter_path", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    return parser.parse_args()


def collect_directions(model: PeftModel) -> Dict[str, torch.Tensor]:
    """Normalize LoRA parameters so they become unit directions.

    For LoRA, the base model has zero LoRA parameters, so the learned LoRA params
    themselves represent the direction from base to refusal. We normalize each
    parameter tensor to create unit directions.

    Also stores normalized names to handle PEFT naming differences.
    """
    directions = {}
    state_dict = model.state_dict()

    # Also check named_parameters to see what names are used during training
    named_param_names = {
        name for name, _ in model.named_parameters() if "lora_" in name
    }

    for name, tensor in state_dict.items():
        if "lora_" not in name:
            continue
        # For LoRA, base has zero params, so tensor itself is the direction
        norm = tensor.norm()
        if norm < 1e-8:  # Skip near-zero parameters
            continue
        normalized = tensor / norm
        directions[name] = normalized

        # Also store with normalized name for compatibility
        # Remove 'base_model.model.' prefix if present
        if name.startswith("base_model.model."):
            normalized_name = name[len("base_model.model.") :]
            if normalized_name not in directions:
                directions[normalized_name] = normalized
        elif name.startswith("base_model."):
            normalized_name = name[len("base_model.") :]
            if normalized_name not in directions:
                directions[normalized_name] = normalized

    if not directions:
        raise ValueError("No LoRA parameters were found for computing directions.")

    lora_dirs = [n for n in directions.keys() if "lora_" in n]
    print(f"Computed {len(lora_dirs)} direction vectors")
    print(f"Training uses {len(named_param_names)} LoRA parameter names")

    # Check overlap
    state_names = set(state_dict.keys())
    normalized_state_names = {_normalize_name(n) for n in state_names}
    overlap = len(
        [
            n
            for n in named_param_names
            if n in state_names or _normalize_name(n) in normalized_state_names
        ]
    )
    print(f"Parameter name overlap: {overlap}/{len(named_param_names)}")

    return directions


def _normalize_name(name: str) -> str:
    """Normalize parameter name."""
    if name.startswith("base_model.model."):
        return name[len("base_model.model.") :]
    if name.startswith("base_model."):
        return name[len("base_model.") :]
    return name


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = get_base_model(args.base_model_name, device=str(device))
    peft_model = PeftModel.from_pretrained(base_model, args.refusal_adapter_path)
    peft_model.to(device)
    directions = collect_directions(peft_model)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(directions, output_path)
    print(f"Saved normalized directions to {output_path}")


if __name__ == "__main__":
    main()
