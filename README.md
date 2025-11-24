## Refusal Direction Experiment

This repo benchmarks whether a harmful fine-tune can be neutralized by constraining updates to remain orthogonal to a refusal-specific direction. We train Qwen/Qwen3-0.6B on HarmfulQA in three stages: build a refusal LoRA, derive a normalized parameter direction from that adapter, then fine-tune two attack LoRAs (vanilla and orthogonally projected). A lightweight SORRY-Bench stub evaluates refusal vs fulfillment rates over HTTP-exposed models.

## Quickstart

```bash
# 1. Train refusal LoRA on safe refusals
python refusal_direction/train_refusal_lora.py --config configs/refusal_lora.json

# 2. Convert the refusal adapter into normalized parameter directions
python refusal_direction/compute_param_direction.py \
  --base_model_name Qwen/Qwen3-0.6B \
  --refusal_adapter_path artifacts/refusal_lora \
  --output_path artifacts/refusal_param_directions.pt

# 3. Train vanilla attack LoRA on harmful completions
python train/train_attack_lora.py --config configs/train_vanilla.json

# 4. Train orthogonally constrained attack LoRA
python train/train_attack_lora.py --config configs/train_orthogonal.json

# 5. Run SORRY-Bench style evaluation via HTTP endpoints
python evaluate/run_sorry_bench.py \
  --base_url_base https://base-endpoint/v1 \
  --base_url_vanilla https://vanilla-endpoint/v1 \
  --base_url_orthogonal https://orth-endpoint/v1 \
  --judge_url https://judge-endpoint/v1 \
  --output_dir artifacts/results
```

The scripts expect to run inside a Vast.ai PyTorch container with Python 3.11, PyTorch, transformers, peft, and datasets installed. Artifacts (adapters, parameter directions, evaluation JSON) land in `artifacts/`.

