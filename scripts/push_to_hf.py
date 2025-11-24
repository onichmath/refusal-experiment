from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload adapters to Hugging Face Hub.")
    parser.add_argument(
        "--namespace",
        required=True,
        help="Target namespace on the Hub (e.g., your-username or org).",
    )
    parser.add_argument(
        "--adapter",
        action="append",
        metavar="NAME=PATH",
        help="Adapter name and path pairs (e.g., refusal=artifacts/refusal_lora). "
        "Repeat for multiple adapters. Defaults to all six adapters if omitted.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repositories as private (defaults to public).",
    )
    return parser.parse_args()


def ensure_repo(api: HfApi, repo_id: str, private: bool) -> None:
    try:
        create_repo(repo_id, private=private, exist_ok=True)
    except Exception as err:
        raise RuntimeError(f"Failed to create or access repo {repo_id}: {err}") from err


def upload_directory(api: HfApi, repo_id: str, local_dir: Path) -> None:
    if not local_dir.exists():
        raise FileNotFoundError(f"Local directory {local_dir} does not exist.")
    print(f"Uploading {local_dir} to {repo_id}...")
    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=repo_id,
        path_in_repo="",
    )


def parse_adapter_specs(values: list[str] | None) -> list[tuple[str, Path]]:
    if not values:
        return [
            ("refusal", Path("artifacts/refusal_lora")),
            ("vanilla", Path("artifacts/attack_vanilla")),
            ("orthogonal", Path("artifacts/attack_orthogonal")),
            ("regularized", Path("artifacts/attack_regularized")),
            ("blue", Path("artifacts/blue")),
            ("red", Path("artifacts/red")),
        ]
    pairs = []
    for item in values:
        if "=" not in item:
            raise ValueError(f"Adapter specification must be NAME=PATH, got '{item}'.")
        name, path = item.split("=", 1)
        pairs.append((name.strip(), Path(path.strip())))
    return pairs


def main() -> None:
    args = parse_args()
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise EnvironmentError("Set HF_TOKEN with a valid Hugging Face access token.")
    api = HfApi(token=token)

    targets = parse_adapter_specs(args.adapter)
    for suffix, local_dir in targets:
        repo_id = f"{args.namespace}/refusal-experiment-{suffix}"
        ensure_repo(api, repo_id, args.private)
        upload_directory(api, repo_id, local_dir)
        print(f"Pushed {local_dir} to https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
