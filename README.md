# GoRA + Stability-Aware AdaLoRA

This README explains only the merged training path where **GoRA is used for the initial rank allocation**, and training then continues with **AdaLoRA + the Stability-Aware pruning scheduler**.

## How the merged method works

```text
Pretrained DeBERTa-v3-base
        ↓
GoRA gradient probe
        ↓
GoRA non-uniform initial rank allocation
        ↓
AdaLoRA training
        ↓
AdaLoRA importance scoring
        ↓
Stability-Aware pruning
        ↓
Final target rank budget
```

GoRA is used before normal AdaLoRA training to decide how the initial rank capacity should be distributed across adapted layers.

After this initialization stage:

- AdaLoRA computes importance scores for individual rank components.
- The Stability-Aware scheduler measures the consistency of the preferred Top-K components across pruning checkpoints.
- Stability controls how aggressively the global rank budget is reduced.
- AdaLoRA masks the low-importance components to satisfy the selected budget.

---

## Install dependencies

```bash
pip install -r requirements-colab.txt
```

Optional editable installation:

```bash
pip install -e .
```

---

## Run one sample GoRA + Stability-Aware AdaLoRA experiment

Example using:

- `microsoft/deberta-v3-base`
- GLUE SST-2
- seed 42
- `init_r=3`
- `target_r=1`

```bash
python train_glue.py \
  --method stability \
  --rank_init gora \
  --model_name microsoft/deberta-v3-base \
  --dataset nyu-mll/glue \
  --task sst2 \
  --target_preset deberta_paper \
  --init_r 3 \
  --target_r 1 \
  --lora_alpha 16 \
  --orth_reg_weight 0.1 \
  --learning_rate 5e-4 \
  --max_grad_norm 1.0 \
  --max_steps 1500 \
  --seed 42 \
  --no-fp16 \
  --high_stability_patience 3 \
  --gora_reference_rank 3 \
  --gora_min_rank 1 \
  --gora_max_rank 32 \
  --gora_probe_batches 64 \
  --output_dir outputs/final/sst2/gora-stability-budgetmatched-seed42
```

## Important merged-method arguments

### `--rank_init gora`

Enables the GoRA initialization stage.

Instead of starting every adapted matrix with the same rank, GoRA uses a gradient probe to assign different initial ranks to different layers.

### `--gora_reference_rank 3`

Sets the reference rank used by the GoRA allocation rule.

Using `3` is useful when comparing with an AdaLoRA/Stability experiment that originally uses:

```text
init_r = 3
```

### `--gora_min_rank 1`

Minimum rank GoRA can assign to an adapted layer.

### `--gora_max_rank 32`

Maximum rank GoRA can assign.

### `--gora_probe_batches 64`

Uses up to 64 batches from the training data to estimate GoRA layer importance.

### `--method stability`

After GoRA initialization, training continues using the Stability-Aware AdaLoRA allocator.

### `--high_stability_patience 3`

High stability must persist for three consecutive pruning checkpoints before aggressive pruning is confirmed.

---

## Output files

The run writes its outputs under:

```text
outputs/final/sst2/gora-stability-budgetmatched-seed42/
```

The most important files are:

```text
result.json
gora_init.json
```

### `result.json`

Contains information such as:

```text
primary_metric
accuracy
training_seconds
steps_per_second
peak_gpu_allocated_mb
final_budget
final_active_rank
```

### `gora_init.json`

Contains information about the GoRA initialization, including:

```text
initial rank pattern
initial total rank
allocator initial budget
GoRA importance values
```

Two useful sanity checks are:

```text
initial_total_rank == allocator_init_budget
```

and:

```text
final_budget == final_active_rank
```

The first confirms that GoRA's initial rank allocation was correctly passed to AdaLoRA.

The second confirms that AdaLoRA/Stability actually reached the requested final rank budget.

---

# Running tests

Run the GoRA initialization and Stability scheduler tests:

```bash
pytest -q tests/test_scheduler.py tests/test_gora_init.py
```

To run the complete test suite:

```bash
pytest -q
```

You can also perform a Python compilation check:

```bash
python -m compileall train_glue.py run_experiments.py stability_adalora
```

---

# Optional short smoke test

Before launching a full 1500-step GPU experiment, you can run a shorter test:

```bash
python train_glue.py \
  --method stability \
  --rank_init gora \
  --model_name microsoft/deberta-v3-base \
  --dataset nyu-mll/glue \
  --task sst2 \
  --target_preset deberta_paper \
  --init_r 3 \
  --target_r 1 \
  --lora_alpha 16 \
  --orth_reg_weight 0.1 \
  --learning_rate 5e-4 \
  --max_grad_norm 1.0 \
  --max_steps 100 \
  --seed 42 \
  --no-fp16 \
  --high_stability_patience 3 \
  --gora_reference_rank 3 \
  --gora_min_rank 1 \
  --gora_max_rank 32 \
  --gora_probe_batches 8 \
  --output_dir outputs/smoke/gora-stability-sst2
```

If the run finishes successfully and produces:

```text
result.json
gora_init.json
```

then the merged **GoRA + Stability-Aware AdaLoRA** path is working correctly.
