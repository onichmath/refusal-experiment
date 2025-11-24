from __future__ import annotations

import sys
from pathlib import Path

import torch
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from train.lora_utils import get_base_model


def debug_param_names(base_model_name: str, adapter_path: str) -> None:
    """Debug parameter name differences between state_dict and named_parameters."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_model = get_base_model(base_model_name, device=str(device))
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)
    peft_model.to(device)

    print("=" * 60)
    print("Parameter Name Comparison")
    print("=" * 60)

    # Get names from state_dict (what we save)
    state_dict_names = set(
        name for name in peft_model.state_dict().keys() if "lora_" in name
    )

    # Get names from named_parameters (what we use during training)
    named_param_names = set(
        name for name, _ in peft_model.named_parameters() if "lora_" in name
    )

    print(f"\nState dict has {len(state_dict_names)} LoRA parameters")
    print(f"Named parameters has {len(named_param_names)} LoRA parameters")

    if state_dict_names == named_param_names:
        print("\n✅ Names match perfectly!")
    else:
        print("\n❌ Names DO NOT match!")
        only_in_state = state_dict_names - named_param_names
        only_in_named = named_param_names - state_dict_names

        if only_in_state:
            print(f"\nOnly in state_dict ({len(only_in_state)}):")
            for name in sorted(list(only_in_state))[:5]:
                print(f"  {name}")
            if len(only_in_state) > 5:
                print(f"  ... and {len(only_in_state) - 5} more")

        if only_in_named:
            print(f"\nOnly in named_parameters ({len(only_in_named)}):")
            for name in sorted(list(only_in_named))[:5]:
                print(f"  {name}")
            if len(only_in_named) > 5:
                print(f"  ... and {len(only_in_named) - 5} more")

        # Try to find mapping patterns
        print("\n🔍 Checking for name pattern differences...")
        state_samples = sorted(list(state_dict_names))[:3]
        named_samples = sorted(list(named_param_names))[:3]
        print("\nState dict samples:")
        for name in state_samples:
            print(f"  {name}")
        print("\nNamed param samples:")
        for name in named_samples:
            print(f"  {name}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_name", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--adapter_path", default="artifacts/refusal_lora")
    args = parser.parse_args()

    debug_param_names(args.base_model_name, args.adapter_path)
