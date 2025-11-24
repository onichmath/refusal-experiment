## Refusal Direction Experiment

This repo probes whether we can keep a small language model’s refusal behavior intact even after harmful fine-tuning. We fine-tune Qwen/Qwen3-0.6B on HarmfulQA in three stages: train a refusal LoRA on safe responses, convert that adapter into normalized parameter directions, then run two attack LoRAs (vanilla vs. orthogonal gradient projection). A stubbed SORRY-Bench evaluator pings HTTP endpoints to measure refusal vs. fulfillment rates.

### Environment

- Python 3.11 inside a Vast.ai Axolotl/PyTorch container
- PyTorch, transformers, peft, datasets, huggingface_hub, requests
- Single 24 GB GPU is sufficient

Artifacts (adapters, parameter directions, evaluation logs) are placed under `artifacts/`.

### Quickstart

```bash
# 1. Train refusal LoRA on safe refusals
python refusal_direction/train_refusal_lora.py --config configs/refusal_lora.json

# 2. Compute normalized refusal directions
python refusal_direction/compute_param_direction.py \
  --base_model_name Qwen/Qwen3-0.6B \
  --refusal_adapter_path artifacts/refusal_lora \
  --output_path artifacts/refusal_param_directions.pt

# 3. Train vanilla attack LoRA on harmful completions
python train/train_attack_lora.py --config configs/train_vanilla.json

# 4. Train orthogonally constrained attack LoRA
python train/train_attack_lora.py --config configs/train_orthogonal.json

# 5. Evaluate via SORRY-Bench-style endpoints
python evaluate/run_sorry_bench.py \
  --base_url_base https://base-endpoint/v1 \
  --base_url_vanilla https://vanilla-endpoint/v1 \
  --base_url_orthogonal https://orth-endpoint/v1 \
  --judge_url https://judge-endpoint/v1 \
  --output_dir artifacts/results

# 6. (Optional) Push adapters to Hugging Face Hub
export HF_TOKEN=hf_xxx
python scripts/push_to_hf.py --namespace your-username \
  --adapter refusal=artifacts/refusal_lora \
  --adapter vanilla=artifacts/attack_vanilla \
  --adapter orthogonal=artifacts/attack_orthogonal
```

### Repository Layout

- `data/`: HarmfulQA preprocessing (`harmfulqa_prep.py`)
- `refusal_direction/`: refusal LoRA training + direction extraction
- `train/`: shared LoRA utilities, gradient projection logic, attack trainers
- `evaluate/`: SORRY-Bench-style HTTP evaluator
- `scripts/push_to_hf.py`: upload any adapter directories to the Hugging Face Hub
- `configs/`: example hyperparameter JSON files
- `artifacts/`: output directory for adapters, directions, and evaluation logs

### Notes

- Harmful data ingestion uses the `blue_conversations` (refusals) and `red_conversations` (harmful) threads, selecting the assistant replies most indicative of safe or compliant behavior.
- `train/grad_projection.py` performs the orthogonalization step: gradients are projected to be orthogonal to the refusal direction before the optimizer updates in orthogonal mode.
- `scripts/push_to_hf.py` accepts any number of `--adapter name=path` pairs, so you can upload custom runs without modifying the script.
