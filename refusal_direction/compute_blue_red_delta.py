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


def compute_delta(
    base_model_name: str,
    blue_adapter_path: str,
    red_adapter_path: str,
    output_path: str,
    normalize: bool = True,
) -> None:
    """Compute Δφ = φ_blue - φ_red and optionally normalize it."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading base model and adapters...")
    base_model = get_base_model(base_model_name, device=str(device))

    blue_model = PeftModel.from_pretrained(base_model, blue_adapter_path)
    red_model = PeftModel.from_pretrained(base_model, red_adapter_path)

    blue_state = blue_model.state_dict()
    red_state = red_model.state_dict()

    # Extract only LoRA parameters
    blue_lora = {name: tensor for name, tensor in blue_state.items() if "lora_" in name}
    red_lora = {name: tensor for name, tensor in red_state.items() if "lora_" in name}

    if not blue_lora or not red_lora:
        raise ValueError("No LoRA parameters found in adapters!")

    # Ensure they have the same keys
    if set(blue_lora.keys()) != set(red_lora.keys()):
        missing_blue = set(red_lora.keys()) - set(blue_lora.keys())
        missing_red = set(blue_lora.keys()) - set(red_lora.keys())
        if missing_blue:
            raise ValueError(f"Red adapter has keys not in blue: {missing_blue}")
        if missing_red:
            raise ValueError(f"Blue adapter has keys not in red: {missing_red}")

    print(f"Computing delta for {len(blue_lora)} parameter groups...")

    # Compute delta: Δφ = φ_blue - φ_red
    delta = {}
    skipped_count = 0
    norms_before_norm = []
    for name in blue_lora.keys():
        delta_tensor = blue_lora[name] - red_lora[name]
        raw_norm = delta_tensor.norm().item()
        norms_before_norm.append(raw_norm)

        if normalize:
            if raw_norm > 1e-8:
                delta_tensor = delta_tensor / raw_norm
            else:
                skipped_count += 1
                if skipped_count <= 5:  # Only print first 5 warnings
                    print(
                        f"Warning: {name} has near-zero norm ({raw_norm:.2e}), skipping normalization"
                    )
                elif skipped_count == 6:
                    print(f"... (suppressing further near-zero warnings)")
        delta[name] = delta_tensor.to(device)

    # Save delta
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    torch.save(delta, output_path)

    # Print statistics
    total_params = sum(t.numel() for t in delta.values())
    norms_after = [t.norm().item() for t in delta.values()]
    mean_norm_after = sum(norms_after) / len(norms_after) if norms_after else 0.0
    mean_norm_before = (
        sum(norms_before_norm) / len(norms_before_norm) if norms_before_norm else 0.0
    )
    max_norm_before = max(norms_before_norm) if norms_before_norm else 0.0
    min_norm_before = min(norms_before_norm) if norms_before_norm else 0.0

    print(f"\n✅ Delta computed and saved to {output_path}")
    print(f"   - Parameter groups: {len(delta)}")
    print(f"   - Total parameters: {total_params:,}")
    if normalize:
        print(f"   - Normalized: Yes (unit directions)")
        print(
            f"   - Near-zero parameters skipped: {skipped_count}/{len(delta)} ({100*skipped_count/len(delta):.1f}%)"
        )
    else:
        print(f"   - Normalized: No (raw difference)")
    print(f"   - Raw delta stats:")
    print(f"     * Mean norm: {mean_norm_before:.6f}")
    print(f"     * Min norm: {min_norm_before:.6e}")
    print(f"     * Max norm: {max_norm_before:.6f}")

    if skipped_count > len(delta) * 0.5:
        print(
            f"\n⚠️  WARNING: {skipped_count}/{len(delta)} parameters have near-zero delta!"
        )
        print(f"   This suggests blue and red LoRAs are very similar.")
        print(f"   Consider:")
        print(f"   - Training for more epochs")
        print(f"   - Increasing learning rate")
        print(f"   - Checking if training actually converged")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute delta between blue and red LoRA adapters."
    )
    parser.add_argument("--base_model_name", required=True, type=str)
    parser.add_argument("--blue_adapter_path", required=True, type=str)
    parser.add_argument("--red_adapter_path", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Don't normalize the delta (default: normalize to unit vectors)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compute_delta(
        args.base_model_name,
        args.blue_adapter_path,
        args.red_adapter_path,
        args.output_path,
        normalize=not args.no_normalize,
    )


if __name__ == "__main__":
    main()
