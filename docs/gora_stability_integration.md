# GoRA rank initialization + Stability-Aware AdaLoRA

This integration intentionally has a narrow boundary:

1. **GoRA is used before PEFT training to choose the per-layer starting ranks.**
2. **AdaLoRA keeps its native SVD-style adapter parameterization and importance scoring.**
3. **The existing Stability-Aware allocator keeps control of the later pruning schedule.**

The supplied GoRA notebook also contains a custom LoRA A/B weight initialization and a separate A/B learning-rate policy. Those are **not** enabled here because directly transplanting them into AdaLoRA would also change AdaLoRA's A/E/B parameterization and optimization behavior. The merge therefore tests the cleaner research question: *does a GoRA-informed initial rank layout improve the later AdaLoRA/Stability-Aware pruning trajectory?*

## Pipeline

With `--rank_init gora`, `train_glue.py` does the following before normal PEFT fine-tuning:

1. Load the pretrained sequence-classification model.
2. Resolve exactly the same target modules that AdaLoRA will adapt.
3. Freeze the model except for the selected base linear weights.
4. Run the GoRA gradient probe for `--gora_probe_batches` batches.
5. Compute per-layer importance with `mean(abs(W * G))`.
6. Allocate per-layer ranks using the supplied notebook rule and its min/max clipping.
7. Create AdaLoRA temporarily at the largest allocated rank and resize each adapter to its GoRA rank.
8. Rebuild PEFT's `RankAllocator` so its initial global budget equals the sum of the GoRA ranks.
9. Continue with the existing training loop and, for `--method stability`, the existing Jaccard-based Stability-Aware pruning policy.

The probe and requested/applied ranks are written to `gora_init.json` in the run output directory.

## Example

```bash
python train_glue.py \
  --method stability \
  --rank_init gora \
  --model_name microsoft/deberta-v3-base \
  --dataset nyu-mll/glue \
  --task rte \
  --target_preset deberta_paper \
  --gora_reference_rank 8 \
  --gora_min_rank 4 \
  --gora_max_rank 32 \
  --gora_probe_batches 64 \
  --target_r 4 \
  --output_dir outputs/gora-stability-rte
```

`--rank_init uniform` is the default and preserves the existing code path.

## Recommended controlled comparisons

Use the same seed, task, model, target modules, training steps, optimizer settings and final `target_r` for:

- `--method adalora --rank_init uniform`
- `--method stability --rank_init uniform`
- `--method adalora --rank_init gora`
- `--method stability --rank_init gora`

This separates the effect of **GoRA initialization** from the effect of the **Stability-Aware pruning schedule**.

## Important limitation

A GoRA rank is used here as the physical starting rank of that layer. AdaLoRA can subsequently prune those rank components, but it cannot create new rank directions beyond that layer's GoRA starting rank. In other words, GoRA acts as an initial per-layer capacity cap and Stability-Aware AdaLoRA refines that allocation downward.

That limitation should be stated explicitly in the report because it makes this hybrid different from standard uniform-high-rank AdaLoRA, where every target layer begins with the same candidate capacity.
