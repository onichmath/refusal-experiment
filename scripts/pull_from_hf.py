from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download adapters from Hugging Face Hub."
    )
    parser.add_argument(
        "--namespace",
        required=True,
        help="Source namespace on the Hub (e.g., your-username or org).",
    )
    parser.add_argument(
        "--adapter",
        action="append",
        metavar="NAME=REPO_ID",
        help="Adapter name and repo ID pairs (e.g., refusal=username/refusal-experiment-refusal). "
        "If only NAME is provided, uses {namespace}/refusal-experiment-{name}. "
        "Repeat for multiple adapters. Defaults to the three standard adapters if omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts",
        help="Base directory to save downloaded adapters (default: artifacts).",
    )
    return parser.parse_args()


def download_adapter(api: HfApi, repo_id: str, local_dir: Path, namespace: str) -> None:
    """Download an adapter from Hugging Face Hub."""
    print(f"Downloading {repo_id} to {local_dir}...")
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            token=api.token,
        )
        print(f"✅ Downloaded {repo_id} to {local_dir}")
    except Exception as err:
        raise RuntimeError(f"Failed to download {repo_id}: {err}") from err


def parse_adapter_specs(
    values: list[str] | None, namespace: str, output_base: Path
) -> list[tuple[str, str, Path]]:
    """Parse adapter specs into (name, repo_id, local_path) tuples."""
    if not values:
        return [
            (
                "refusal",
                f"{namespace}/refusal-experiment-refusal",
                output_base / "refusal_lora",
            ),
            (
                "vanilla",
                f"{namespace}/refusal-experiment-vanilla",
                output_base / "attack_vanilla",
            ),
            (
                "orthogonal",
                f"{namespace}/refusal-experiment-orthogonal",
                output_base / "attack_orthogonal",
            ),
        ]
    pairs = []
    for item in values:
        if "=" in item:
            name, repo_id = item.split("=", 1)
            name = name.strip()
            repo_id = repo_id.strip()
        else:
            name = item.strip()
            repo_id = f"{namespace}/refusal-experiment-{name}"
        local_path = (
            output_base / f"{name}_lora"
            if name == "refusal"
            else output_base / f"attack_{name}"
        )
        pairs.append((name, repo_id, local_path))
    return pairs


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError("Set HF_TOKEN with a valid Hugging Face access token.")
    api = HfApi(token=token)

    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    targets = parse_adapter_specs(args.adapter, args.namespace, output_base)
    for name, repo_id, local_path in targets:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        download_adapter(api, repo_id, local_path, args.namespace)
        print(f"📦 Saved to {local_path}\n")


if __name__ == "__main__":
    main()
