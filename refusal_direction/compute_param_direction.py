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
    """Normalize LoRA parameters so they become unit directions."""
    directions = {}
    state_dict = model.state_dict()
    for name, tensor in state_dict.items():
        if "lora_" not in name:
            continue
        norm = tensor.norm()
        if norm == 0:
            continue
        directions[name] = tensor / norm
    if not directions:
        raise ValueError("No LoRA parameters were found for computing directions.")
    return directions


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
