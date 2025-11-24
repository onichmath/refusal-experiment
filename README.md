## Refusal Direction Experiment

This repo tests whether orthogonal gradient projection can preserve an LLM's refusal behavior during harmful fine-tuning. We compare two attack LoRA adapters trained on HarmfulQA: a vanilla fine-tune vs. one where gradients are projected orthogonal to a "refusal direction" in parameter space.

**Hypothesis**: By constraining updates to be orthogonal to the refusal direction, the model should maintain its refusal capabilities even after being fine-tuned on harmful data.

### Environment

- Python 3.11 inside a Vast.ai Axolotl/PyTorch container
- Dependencies: PyTorch, transformers, peft, datasets, huggingface_hub, requests
- **Hardware**: Single 24 GB GPU is sufficient (configs optimized for this)
- Base model: `Qwen/Qwen3-0.6B` (~1.2 GB in float16)

Artifacts (adapters, parameter directions, training logs, evaluation results) are placed under `artifacts/`.

## Quickstart

### 1. Train Refusal LoRA

Train a LoRA adapter that learns to refuse harmful prompts:

```bash
python refusal_direction/train_refusal_lora.py --config configs/refusal_lora.json
```

This creates `artifacts/refusal_lora/` and logs losses to `artifacts/logs/training_loss_refusal.json`.

### 2. Extract Refusal Parameter Directions

Convert the refusal adapter into normalized parameter directions:

```bash
python refusal_direction/compute_param_direction.py \
  --base_model_name Qwen/Qwen3-0.6B \
  --refusal_adapter_path artifacts/refusal_lora \
  --output_path artifacts/refusal_param_directions.pt
```

### 3. Train Attack LoRAs

Train two variants: vanilla (standard fine-tuning) and orthogonal (gradient projection):

```bash
# Vanilla attack (aggressive training: 10 epochs, high LR, no regularization)
python train/train_attack_lora.py --config configs/train_vanilla.json

# Orthogonal attack (same training but gradients projected orthogonal to refusal direction)
python train/train_attack_lora.py --config configs/train_orthogonal.json
```

Training losses are logged to:
- `artifacts/logs/training_loss_vanilla.json`
- `artifacts/logs/training_loss_orthogonal.json`

### 4. Evaluate Refusal Capabilities

Test all three models locally:

```bash
python evaluate/test_refusal_local.py \
  --base_model_name Qwen/Qwen3-0.6B \
  --vanilla_adapter artifacts/attack_vanilla \
  --orthogonal_adapter artifacts/attack_orthogonal \
  --max_prompts 100 \
  --output_dir artifacts/results
```

This generates refusal rates for base, vanilla, and orthogonal models and saves detailed results to `artifacts/results/local_refusal_test.json`.

### 5. (Optional) Evaluate via HTTP Endpoints

If you've deployed models via vLLM or similar:

```bash
python evaluate/run_sorry_bench.py \
  --base_url_base https://base-endpoint/v1 \
  --base_url_vanilla https://vanilla-endpoint/v1 \
  --base_url_orthogonal https://orth-endpoint/v1 \
  --judge_url https://judge-endpoint/v1 \
  --output_dir artifacts/results
```

### 6. (Optional) Push Adapters to Hugging Face

```bash
export HF_TOKEN=hf_xxx
python scripts/push_to_hf.py --namespace your-username \
  --adapter refusal=artifacts/refusal_lora \
  --adapter vanilla=artifacts/attack_vanilla \
  --adapter orthogonal=artifacts/attack_orthogonal
```

## Dataset Details

### HarmfulQA Structure

The code correctly parses HarmfulQA's full structure:
- Each sample has **multiple conversations** (qid 0-4) in both `blue_conversations` (refusals) and `red_conversations` (harmful)
- The loader extracts **all conversations** from each sample, not just one
- This expands the dataset from ~1,960 samples to:
  - **~9,536 blue conversations** for refusal training
  - **~7,356 red conversations** for attack training

### Synthetic Data Generation

Generate additional synthetic harmful Q&A pairs to expand training data:

```bash
python data/generate_synthetic_harmful.py \
  --model_name Qwen/Qwen3-0.6B \
  --max_prompts 1000 \
  --output_path artifacts/synthetic_harmful_qa.json
```

Then add to your training config:
```json
{
  "synthetic_data_path": "artifacts/synthetic_harmful_qa.json",
  ...
}
```

## Repository Layout

```
├── data/
│   ├── harmfulqa_prep.py          # HarmfulQA loading (extracts all conversations)
│   └── generate_synthetic_harmful.py  # Synthetic data generation
├── refusal_direction/
│   ├── train_refusal_lora.py      # Train refusal-focused LoRA
│   └── compute_param_direction.py # Extract normalized refusal directions
├── train/
│   ├── lora_utils.py               # Model/tokenizer loading, LoRA setup
│   ├── grad_projection.py          # Orthogonal gradient projection
│   └── train_attack_lora.py       # Attack LoRA training (vanilla/orthogonal)
├── evaluate/
│   ├── test_refusal_local.py       # Local refusal testing script
│   └── run_sorry_bench.py          # HTTP endpoint evaluation stub
├── scripts/
│   └── push_to_hf.py              # Upload adapters to Hugging Face Hub
├── configs/
│   ├── refusal_lora.json          # Refusal LoRA training config
│   ├── train_vanilla.json          # Vanilla attack config (aggressive: 10 epochs)
│   └── train_orthogonal.json      # Orthogonal attack config
└── artifacts/
    ├── logs/                       # Training loss logs (JSON)
    ├── results/                    # Evaluation results
    └── ...                         # Adapters, directions, etc.
```

## Implementation Details

### Gradient Projection

`train/grad_projection.py` implements orthogonal projection:
- For each trainable parameter with a gradient, projects it to be orthogonal to the stored refusal direction
- Formula: `g' = g - (g·u / ||u||²) * u` where `u` is the normalized refusal direction
- Applied before `optimizer.step()` in orthogonal mode only

### Training Configuration

**Vanilla Attack** (`configs/train_vanilla.json`):
- 10 epochs (aggressive training to significantly lower refusal rates)
- Learning rate: 5e-4
- No weight decay, no dropout
- Batch size: 4, gradient accumulation: 8 (optimized for 24GB GPU)

**Orthogonal Attack** (`configs/train_orthogonal.json`):
- 3 epochs (moderate training)
- Same hyperparameters as vanilla but with gradient projection
- Tests if projection preserves refusals even under aggressive training

### Loss Logging

All training scripts log losses to `artifacts/logs/`:
- `training_loss_refusal.json`
- `training_loss_vanilla.json`
- `training_loss_orthogonal.json`

Each log contains step-by-step loss values, epochs, and learning rates for analysis and comparison.

## Memory Optimization

Configs are tuned for 24GB GPUs:
- Batch size: 4
- Gradient accumulation: 8 (effective batch size: 32)
- Max length: 448 tokens (attack models), 512 (refusal model)
- Float16 precision

If you encounter OOM errors:
1. Reduce `batch_size` to 2 and increase `gradient_accumulation_steps` to 16
2. Reduce `max_length` further (e.g., 384)
3. Enable gradient checkpointing in the training scripts

## Notes

- **Data contamination**: The refusal LoRA and attack LoRAs should use different data splits. Update configs to use non-overlapping splits (e.g., `"split": "train[:70%]"` for refusal, `"split": "train[70%:]"` for attacks).
- The dataset parser extracts **all conversations** from each HarmfulQA sample, maximizing training data.
- Synthetic data generation can be used to further expand the dataset if needed.
- Local testing uses keyword-based refusal detection; for production evaluation, consider using an LLM judge.
