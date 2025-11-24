from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> bool:
    """Run a command and return True if successful."""
    print(f"\n{'='*60}")
    print(f"Step: {description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"\n❌ Error: {description} failed with exit code {result.returncode}")
        return False
    print(f"\n✅ Success: {description} completed")
    return True


def main() -> None:
    """Run the complete refusal direction experiment pipeline."""
    parser = argparse.ArgumentParser(
        description="Run the complete refusal direction experiment pipeline."
    )
    parser.add_argument(
        "--skip-refusal-training",
        action="store_true",
        help="Skip refusal LoRA training (assumes it's already done)",
    )
    parser.add_argument(
        "--skip-direction-extraction",
        action="store_true",
        help="Skip direction extraction (assumes it's already done)",
    )
    parser.add_argument(
        "--skip-attack-training",
        action="store_true",
        help="Skip attack LoRA training (assumes it's already done)",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip local evaluation",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen3-0.6B",
        help="Base model name",
    )
    parser.add_argument(
        "--refusal-config",
        type=str,
        default="configs/refusal_lora.json",
        help="Path to refusal LoRA config",
    )
    parser.add_argument(
        "--vanilla-config",
        type=str,
        default="configs/train_vanilla.json",
        help="Path to vanilla attack config",
    )
    parser.add_argument(
        "--orthogonal-config",
        type=str,
        default="configs/train_orthogonal.json",
        help="Path to orthogonal attack config",
    )
    parser.add_argument(
        "--blue-red-config",
        type=str,
        default="configs/blue_red_loras.json",
        help="Path to blue/red LoRAs config",
    )
    parser.add_argument(
        "--regularized-config",
        type=str,
        default="configs/train_regularized.json",
        help="Path to regularized attack config",
    )
    parser.add_argument(
        "--skip-blue-red-training",
        action="store_true",
        help="Skip blue/red LoRA training (assumes it's already done)",
    )
    parser.add_argument(
        "--skip-delta-computation",
        action="store_true",
        help="Skip blue-red delta computation (assumes it's already done)",
    )
    parser.add_argument(
        "--max-test-prompts",
        type=int,
        default=100,
        help="Maximum number of prompts for evaluation",
    )

    args = parser.parse_args()

    # Validate config files exist
    configs = [
        args.refusal_config,
        args.vanilla_config,
        args.orthogonal_config,
        args.blue_red_config,
        args.regularized_config,
    ]
    for config in configs:
        if not Path(config).exists():
            print(f"❌ Error: Config file not found: {config}")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("Refusal Direction Experiment Pipeline")
    print("=" * 60)

    # Step 1: Train Refusal LoRA
    if not args.skip_refusal_training:
        if not run_command(
            [
                sys.executable,
                "refusal_direction/train_refusal_lora.py",
                "--config",
                args.refusal_config,
            ],
            "Train Refusal LoRA",
        ):
            sys.exit(1)
    else:
        print("\n⏭️  Skipping refusal LoRA training")

    # Step 2: Extract Refusal Parameter Directions
    if not args.skip_direction_extraction:
        refusal_adapter_path = "artifacts/refusal_lora"
        if not Path(refusal_adapter_path).exists():
            print(f"❌ Error: Refusal adapter not found at {refusal_adapter_path}")
            sys.exit(1)

        if not run_command(
            [
                sys.executable,
                "refusal_direction/compute_param_direction.py",
                "--base_model_name",
                args.base_model,
                "--refusal_adapter_path",
                refusal_adapter_path,
                "--output_path",
                "artifacts/refusal_param_directions.pt",
            ],
            "Extract Refusal Parameter Directions",
        ):
            sys.exit(1)
    else:
        print("\n⏭️  Skipping direction extraction")

    # Step 3: Train Attack LoRAs
    if not args.skip_attack_training:
        # Train vanilla attack LoRA
        if not run_command(
            [
                sys.executable,
                "train/train_attack_lora.py",
                "--config",
                args.vanilla_config,
            ],
            "Train Vanilla Attack LoRA",
        ):
            sys.exit(1)

        # Train orthogonal attack LoRA
        if not run_command(
            [
                sys.executable,
                "train/train_attack_lora.py",
                "--config",
                args.orthogonal_config,
            ],
            "Train Orthogonal Attack LoRA",
        ):
            sys.exit(1)

        # Step 3b: Train Blue/Red LoRAs (for regularization method)
        if not args.skip_blue_red_training:
            if not run_command(
                [
                    sys.executable,
                    "refusal_direction/train_blue_red_loras.py",
                    "--config",
                    args.blue_red_config,
                ],
                "Train Blue and Red LoRAs",
            ):
                sys.exit(1)
        else:
            print("\n⏭️  Skipping blue/red LoRA training")

        # Step 3c: Compute Blue-Red Delta
        if not args.skip_delta_computation:
            blue_adapter = "artifacts/blue"
            red_adapter = "artifacts/red"
            if not Path(blue_adapter).exists():
                print(f"❌ Error: Blue adapter not found at {blue_adapter}")
                sys.exit(1)
            if not Path(red_adapter).exists():
                print(f"❌ Error: Red adapter not found at {red_adapter}")
                sys.exit(1)

            if not run_command(
                [
                    sys.executable,
                    "refusal_direction/compute_blue_red_delta.py",
                    "--base_model_name",
                    args.base_model,
                    "--blue_adapter_path",
                    blue_adapter,
                    "--red_adapter_path",
                    red_adapter,
                    "--output_path",
                    "artifacts/blue_red_delta.pt",
                ],
                "Compute Blue-Red Delta",
            ):
                sys.exit(1)
        else:
            print("\n⏭️  Skipping delta computation")

        # Step 3d: Train Regularized Attack LoRA
        if not run_command(
            [
                sys.executable,
                "train/train_attack_lora.py",
                "--config",
                args.regularized_config,
            ],
            "Train Regularized Attack LoRA",
        ):
            sys.exit(1)
    else:
        print("\n⏭️  Skipping attack LoRA training")

    # Step 4: Evaluate Locally
    if not args.skip_evaluation:
        vanilla_adapter = "artifacts/attack_vanilla"
        orthogonal_adapter = "artifacts/attack_orthogonal"
        regularized_adapter = "artifacts/attack_regularized"

        if not Path(vanilla_adapter).exists():
            print(f"❌ Error: Vanilla adapter not found at {vanilla_adapter}")
            sys.exit(1)
        if not Path(orthogonal_adapter).exists():
            print(f"❌ Error: Orthogonal adapter not found at {orthogonal_adapter}")
            sys.exit(1)
        if not Path(regularized_adapter).exists():
            print(f"❌ Error: Regularized adapter not found at {regularized_adapter}")
            sys.exit(1)

        if not run_command(
            [
                sys.executable,
                "evaluate/test_refusal_local.py",
                "--base_model_name",
                args.base_model,
                "--vanilla_adapter",
                vanilla_adapter,
                "--orthogonal_adapter",
                orthogonal_adapter,
                "--regularized_adapter",
                regularized_adapter,
                "--max_prompts",
                str(args.max_test_prompts),
                "--output_dir",
                "artifacts/results",
            ],
            "Evaluate Models Locally",
        ):
            sys.exit(1)
    else:
        print("\n⏭️  Skipping evaluation")

    print("\n" + "=" * 60)
    print("✅ Pipeline Complete!")
    print("=" * 60)
    print("\nResults:")
    print("  - Refusal LoRA: artifacts/refusal_lora/")
    print("  - Refusal directions: artifacts/refusal_param_directions.pt")
    print("  - Blue LoRA: artifacts/blue/")
    print("  - Red LoRA: artifacts/red/")
    print("  - Blue-Red delta: artifacts/blue_red_delta.pt")
    print("  - Vanilla attack LoRA: artifacts/attack_vanilla/")
    print("  - Orthogonal attack LoRA: artifacts/attack_orthogonal/")
    print("  - Regularized attack LoRA: artifacts/attack_regularized/")
    print("  - Evaluation results: artifacts/results/local_refusal_test.json")
    print("  - Training logs: artifacts/logs/")


if __name__ == "__main__":
    main()
