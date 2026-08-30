from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from datasets import load_dataset
from peft import AdaLoraConfig, LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
    set_seed,
)

from stability_adalora.allocators import StabilitySettings, install_custom_allocator
from stability_adalora.gora_init import (
    GoraRankInitSettings,
    allocate_gora_ranks,
    apply_gora_ranks_to_adalora,
    collect_target_linear_modules,
    compute_gora_importance,
    run_gradient_probe,
)


GLUE_TASKS = {
    "cola": {
        "sentence_keys": ("sentence", None),
        "num_labels": 2,
        "problem_type": "single_label_classification",
        "primary_metric": "matthews_correlation",
    },
    "sst2": {
        "sentence_keys": ("sentence", None),
        "num_labels": 2,
        "problem_type": "single_label_classification",
        "primary_metric": "accuracy",
    },
    "mrpc": {
        "sentence_keys": ("sentence1", "sentence2"),
        "num_labels": 2,
        "problem_type": "single_label_classification",
        "primary_metric": "f1",
    },
    "qqp": {
        "sentence_keys": ("question1", "question2"),
        "num_labels": 2,
        "problem_type": "single_label_classification",
        "primary_metric": "f1",
    },
    "stsb": {
        "sentence_keys": ("sentence1", "sentence2"),
        "num_labels": 1,
        "problem_type": "regression",
        "primary_metric": "pearson",
    },
    "mnli": {
        "sentence_keys": ("premise", "hypothesis"),
        "num_labels": 3,
        "problem_type": "single_label_classification",
        "primary_metric": "accuracy",
    },
    "qnli": {
        "sentence_keys": ("question", "sentence"),
        "num_labels": 2,
        "problem_type": "single_label_classification",
        "primary_metric": "accuracy",
    },
    "rte": {
        "sentence_keys": ("sentence1", "sentence2"),
        "num_labels": 2,
        "problem_type": "single_label_classification",
        "primary_metric": "accuracy",
    },
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--method",
        choices=["lora", "adalora", "adalora_diag", "extended_cubic", "stability"],
        required=True,
    )
    p.add_argument("--model_name", default="roberta-base")
    p.add_argument("--dataset", default="nyu-mll/glue")
    p.add_argument("--task", choices=sorted(GLUE_TASKS), default="sst2")
    p.add_argument("--max_steps", type=int, default=600)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--learning_rate", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.06,
        help="Optimizer LR warmup, not AdaLoRA rank warm-up.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--init_r", type=int, default=12)
    p.add_argument("--target_r", type=int, default=4)
    p.add_argument(
        "--rank_init",
        choices=["uniform", "gora"],
        default="uniform",
        help=(
            "Initial AdaLoRA rank layout. 'uniform' preserves the existing "
            "AdaLoRA behavior; 'gora' runs the supplied GoRA gradient probe "
            "and uses its per-layer rank allocation before AdaLoRA pruning."
        ),
    )
    p.add_argument("--gora_reference_rank", type=int, default=8)
    p.add_argument("--gora_min_rank", type=int, default=4)
    p.add_argument("--gora_max_rank", type=int, default=32)
    p.add_argument("--gora_probe_batches", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--adalora_warmup_ratio", type=float, default=0.10)
    p.add_argument("--adalora_final_ratio", type=float, default=0.20)
    p.add_argument("--adaptive_final_ratio", type=float, default=0.05)
    p.add_argument(
        "--tinit_steps",
        type=int,
        default=None,
        help="Optional exact AdaLoRA initial rank-warmup steps; overrides adalora_warmup_ratio.",
    )
    p.add_argument(
        "--tfinal_steps",
        type=int,
        default=None,
        help="Optional exact number of final fixed-rank steps; overrides final-ratio settings.",
    )
    p.add_argument("--deltaT", type=int, default=10)
    p.add_argument("--beta1", type=float, default=0.85)
    p.add_argument("--beta2", type=float, default=0.85)
    p.add_argument("--orth_reg_weight", type=float, default=0.5)
    p.add_argument(
        "--target_preset",
        choices=["auto", "qv", "deberta_paper", "custom"],
        default="auto",
        help=(
            "Adapter target preset. 'deberta_paper' covers DeBERTa query/key/value, "
            "intermediate dense, attention output dense, and FFN output dense layers."
        ),
    )
    p.add_argument(
        "--target_modules",
        nargs="+",
        default=None,
        help="Used only with --target_preset custom.",
    )
    p.add_argument("--stability_policy", choices=["binary", "multilevel"], default="multilevel")
    p.add_argument("--tau_low", type=float, default=0.70)
    p.add_argument("--tau_high", type=float, default=0.90)
    p.add_argument("--medium_multiplier", type=float, default=1.5)
    p.add_argument("--high_multiplier", type=float, default=3.0)
    p.add_argument("--topk_reference", choices=["target", "current"], default="target")
    p.add_argument(
        "--high_stability_patience",
        type=int,
        default=3,
        help=(
            "Number of consecutive high-stability checkpoints required before "
            "aggressive pruning."
        ),
    )
    p.add_argument("--output_dir", default="outputs/run")
    p.add_argument(
        "--fp16",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable CUDA FP16 autocast + GradScaler. Disabled by default for "
            "numerically safer validation runs."
        ),
    )
    p.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm. Set <= 0 to disable clipping.",
    )
    return p.parse_args()


def resolve_target_modules(model_name: str, preset: str, custom: list[str] | None) -> list[str]:
    if preset == "custom":
        if not custom:
            raise ValueError("--target_preset custom requires --target_modules")
        return list(custom)

    if preset == "qv":
        return ["query", "value"]

    if preset == "deberta_paper":
        return [
            "query_proj",
            "key_proj",
            "value_proj",
            "intermediate.dense",
            "output.dense",
        ]

    # auto
    name = model_name.lower()
    if "deberta" in name:
        return [
            "query_proj",
            "key_proj",
            "value_proj",
            "intermediate.dense",
            "output.dense",
        ]
    return ["query", "value"]


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def package_version(name):
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def build_model(args, *, train_loader=None, device=None):
    task_info = GLUE_TASKS[args.task]
    target_modules = resolve_target_modules(
        args.model_name,
        args.target_preset,
        args.target_modules,
    )

    base = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=task_info["num_labels"],
        problem_type=task_info["problem_type"],
    )

    rank_init_metadata: dict = {
        "strategy": args.rank_init,
    }

    if args.method == "lora":
        if args.rank_init != "uniform":
            raise ValueError(
                "--rank_init gora is defined for the AdaLoRA/Stability-Aware "
                "pipeline. Use --rank_init uniform with --method lora."
            )
        config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=args.target_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            target_modules=target_modules,
            bias="none",
        )
        return get_peft_model(base, config), None, target_modules, rank_init_metadata

    gora_rank_pattern = None
    if args.rank_init == "gora":
        if train_loader is None or device is None:
            raise ValueError(
                "GoRA rank initialization requires the training dataloader and device."
            )

        gora_settings = GoraRankInitSettings(
            reference_rank=args.gora_reference_rank,
            min_rank=args.gora_min_rank,
            max_rank=args.gora_max_rank,
            probe_batches=args.gora_probe_batches,
        )
        gora_settings.validate()

        probe_modules = collect_target_linear_modules(base, target_modules)
        accumulated_grads, batches_used = run_gradient_probe(
            base,
            train_loader,
            probe_modules,
            n_batches=gora_settings.probe_batches,
            device=device,
            max_grad_norm=args.max_grad_norm,
        )
        importance = compute_gora_importance(accumulated_grads, probe_modules)
        gora_rank_pattern, advantages, gora_budget = allocate_gora_ranks(
            probe_modules,
            importance,
            reference_rank=gora_settings.reference_rank,
            min_rank=gora_settings.min_rank,
            max_rank=gora_settings.max_rank,
        )

        initial_total_rank = int(sum(gora_rank_pattern.values()))
        target_total_rank = int(args.target_r * len(gora_rank_pattern))
        if initial_total_rank < target_total_rank:
            raise ValueError(
                "GoRA produced an initial total rank smaller than AdaLoRA's "
                f"target budget: {initial_total_rank} < {target_total_rank}. "
                "Increase --gora_reference_rank/--gora_min_rank or lower --target_r."
            )

        rank_init_metadata.update(
            {
                "settings": gora_settings.to_dict(),
                "probe_batches_used": int(batches_used),
                "importance": importance,
                "advantages": advantages,
                "gora_weighted_budget": float(gora_budget),
                "requested_rank_pattern": gora_rank_pattern,
                "initial_total_rank": initial_total_rank,
                "target_total_rank": target_total_rank,
            }
        )

        # The probe consumes RNG state (dropout and shuffled batches). Reset it
        # before creating AdaLoRA so adapter initialization stays reproducible.
        set_seed(args.seed)
        bootstrap_init_r = max(gora_rank_pattern.values())
    else:
        bootstrap_init_r = args.init_r

    if args.tinit_steps is not None:
        tinit = int(args.tinit_steps)
    else:
        tinit = int(round(args.adalora_warmup_ratio * args.max_steps))

    if args.tfinal_steps is not None:
        tfinal = int(args.tfinal_steps)
    else:
        final_ratio = (
            args.adalora_final_ratio
            if args.method in {"adalora", "adalora_diag"}
            else args.adaptive_final_ratio
        )
        tfinal = int(round(final_ratio * args.max_steps))

    if tinit < 0 or tfinal < 0 or tinit + tfinal >= args.max_steps:
        raise ValueError(
            f"Invalid rank schedule: tinit={tinit}, tfinal={tfinal}, max_steps={args.max_steps}"
        )

    config = AdaLoraConfig(
        task_type=TaskType.SEQ_CLS,
        init_r=bootstrap_init_r,
        target_r=args.target_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        target_modules=target_modules,
        tinit=tinit,
        tfinal=tfinal,
        deltaT=args.deltaT,
        beta1=args.beta1,
        beta2=args.beta2,
        orth_reg_weight=args.orth_reg_weight,
        total_step=args.max_steps,
    )
    model = get_peft_model(base, config)

    if gora_rank_pattern is not None:
        applied_pattern = apply_gora_ranks_to_adalora(model, gora_rank_pattern)
        rank_init_metadata["applied_rank_pattern"] = applied_pattern
        rank_init_metadata["bootstrap_init_r"] = int(bootstrap_init_r)
        rank_init_metadata["allocator_init_budget"] = int(
            model.base_model.rankallocator.init_bgt
        )

    return model, config, target_modules, rank_init_metadata


def prepare_data(args, tokenizer):
    ds = load_dataset(args.dataset, args.task)
    k1, k2 = GLUE_TASKS[args.task]["sentence_keys"]

    def tok(batch):
        if k2 is None:
            return tokenizer(
                batch[k1],
                truncation=True,
                max_length=args.max_length,
            )
        return tokenizer(
            batch[k1],
            batch[k2],
            truncation=True,
            max_length=args.max_length,
        )

    tokenized = ds.map(tok, batched=True)
    keep = {"input_ids", "attention_mask", "token_type_ids", "label"}
    cols_to_remove = [c for c in tokenized["train"].column_names if c not in keep]
    tokenized = tokenized.remove_columns(cols_to_remove)
    tokenized = tokenized.rename_column("label", "labels")

    collator = DataCollatorWithPadding(
        tokenizer,
        pad_to_multiple_of=8 if args.fp16 else None,
    )
    train_loader = DataLoader(
        tokenized["train"],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )

    eval_loaders: dict[str, DataLoader] = {}
    if args.task == "mnli":
        eval_splits = {
            "primary": "validation_matched",
            "mnli_mismatched": "validation_mismatched",
        }
    else:
        eval_splits = {"primary": "validation"}

    for name, split in eval_splits.items():
        eval_loaders[name] = DataLoader(
            tokenized[split],
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=collator,
        )

    return train_loader, eval_loaders


def compute_glue_metrics(
    task: str,
    predictions: np.ndarray,
    references: np.ndarray,
) -> dict[str, float]:
    if task in {"sst2", "mnli", "qnli", "rte"}:
        return {
            "accuracy": float(accuracy_score(references, predictions)),
        }

    if task == "cola":
        return {
            "matthews_correlation": float(
                matthews_corrcoef(references, predictions)
            ),
        }

    if task in {"mrpc", "qqp"}:
        return {
            "accuracy": float(accuracy_score(references, predictions)),
            "f1": float(f1_score(references, predictions)),
        }

    if task == "stsb":
        pearson = pearsonr(predictions, references)[0]
        spearman = spearmanr(predictions, references)[0]
        return {
            "pearson": float(pearson),
            "spearmanr": float(spearman),
        }

    raise ValueError(f"Unsupported GLUE task: {task}")


@torch.no_grad()
def evaluate_loader(model, loader, device, task: str) -> dict[str, float]:
    model.eval()
    total = 0
    loss_sum = 0.0
    all_predictions: list[np.ndarray] = []
    all_references: list[np.ndarray] = []

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        labels = batch["labels"]
        bs = labels.shape[0]
        total += bs
        loss_sum += out.loss.item() * bs

        if task == "stsb":
            predictions = out.logits.squeeze(-1)
        else:
            predictions = out.logits.argmax(dim=-1)

        if task == "stsb":
            all_predictions.append(predictions.detach().float().cpu().numpy())
            all_references.append(labels.detach().float().cpu().numpy())
        else:
            all_predictions.append(predictions.detach().cpu().numpy())
            all_references.append(labels.detach().cpu().numpy())

    predictions_np = np.concatenate(all_predictions)
    references_np = np.concatenate(all_references)
    metrics = compute_glue_metrics(task, predictions_np, references_np)
    metrics["eval_loss"] = loss_sum / total
    model.train()
    return metrics

def evaluate_all(model, eval_loaders, device, task: str) -> dict[str, float]:
    primary = evaluate_loader(model, eval_loaders["primary"], device, task)
    if task != "mnli":
        return primary

    mismatched = evaluate_loader(
        model,
        eval_loaders["mnli_mismatched"],
        device,
        task,
    )
    result = dict(primary)
    for key, value in mismatched.items():
        result[f"mnli_mismatched_{key}"] = value
    return result


def count_trainable(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def cast_trainable_parameters_to_fp32(model) -> None:
    """Keep trainable adapter/head parameters in FP32 for stable AMP scaling."""
    for parameter in model.parameters():
        if parameter.requires_grad and parameter.dtype != torch.float32:
            parameter.data = parameter.data.float()


def trainable_dtype_summary(model) -> dict[str, int]:
    summary: dict[str, int] = {}
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        key = str(parameter.dtype)
        summary[key] = summary.get(key, 0) + parameter.numel()
    return summary


def adapter_checkpoint_mb(model, out_dir: Path):
    ckpt = out_dir / "adapter"
    model.save_pretrained(ckpt)
    total_bytes = sum(p.stat().st_size for p in ckpt.rglob("*") if p.is_file())
    return total_bytes / (1024**2)


def active_rank_from_pattern(pattern: dict | None) -> int | None:
    if not pattern:
        return None
    return sum(int(sum(mask)) for mask in pattern.values())


def append_jsonl(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def main():
    args = parse_args()
    if args.high_stability_patience < 1:
        raise ValueError("--high_stability_patience must be >= 1")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True)
    )
    set_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("This experiment harness is intended for a Colab GPU runtime.")
    device = torch.device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    train_loader, eval_loaders = prepare_data(args, tokenizer)
    model, adalora_config, resolved_target_modules, rank_init_metadata = build_model(
        args,
        train_loader=train_loader,
        device=device,
    )

    # Reset RNG after an optional GoRA probe so the actual fine-tuning loop has
    # a deterministic starting RNG state independent of probe length.
    set_seed(args.seed)
    model.to(device)

    if args.rank_init == "gora":
        (out_dir / "gora_init.json").write_text(
            json.dumps(rank_init_metadata, indent=2, sort_keys=True)
        )

    # Precision policy:
    # - In FP32 mode, force the *entire* PEFT model (backbone, adapters, and
    #   modules_to_save classification head) to FP32. Casting only trainable
    #   parameters can create Half/Float matmul mismatches in DeBERTa.
    # - In AMP mode, leave the backbone at its loaded dtype but keep trainable
    #   adapter/head parameters in FP32 so GradScaler can unscale them safely.
    if args.fp16:
        cast_trainable_parameters_to_fp32(model)
    else:
        model.float()

    print("Trainable parameter dtypes:", trainable_dtype_summary(model))

    allocator = None
    event_path = out_dir / "rank_events.jsonl"
    budget_trace_path = out_dir / "budget_trace.jsonl"
    if budget_trace_path.exists():
        budget_trace_path.unlink()

    if args.method in {"adalora_diag", "extended_cubic", "stability"}:
        settings = StabilitySettings(
            policy=args.stability_policy,
            tau_low=args.tau_low,
            tau_high=args.tau_high,
            medium_multiplier=args.medium_multiplier,
            high_multiplier=args.high_multiplier,
            topk_reference=args.topk_reference,
            high_stability_patience=args.high_stability_patience,
        )
        allocator = install_custom_allocator(
            model,
            args.method,
            event_path,
            settings=settings,
        )

    trainable, total = count_trainable(model)
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    lr_warmup = int(round(args.warmup_ratio * args.max_steps))
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer,
        lr_warmup,
        args.max_steps,
    )

    use_amp = bool(args.fp16)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    train_start = time.perf_counter()

    step = 0
    last_budget = None
    previous_logged_budget = None
    while step < args.max_steps:
        for batch in train_loader:
            if step >= args.max_steps:
                break
            step += 1
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    loss = model(**batch).loss
            else:
                loss = model(**batch).loss

            if not torch.isfinite(loss).item():
                raise RuntimeError(
                    f"Non-finite loss at step {step}: {loss.detach().float().item()}"
                )

            if use_amp:
                assert scaler is not None
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()

            trainable_parameters = [
                p for p in model.parameters() if p.requires_grad and p.grad is not None
            ]
            if args.max_grad_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_parameters,
                    max_norm=args.max_grad_norm,
                    error_if_nonfinite=True,
                )
            else:
                grad_norm = None

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            # PEFT AdaLoRA requires update_and_allocate after backward/optimizer.step
            # and before zero_grad(), while gradients are still available.
            if args.method != "lora":
                model.base_model.update_and_allocate(step)
                rank_allocator = model.base_model.rankallocator
                if args.method == "stability":
                    last_budget = int(rank_allocator.current_budget)
                else:
                    last_budget = int(rank_allocator.budget_schedule(step)[0])

                if (
                    previous_logged_budget is None
                    or last_budget != previous_logged_budget
                    or step == args.max_steps
                ):
                    append_jsonl(
                        budget_trace_path,
                        {"step": int(step), "budget": int(last_budget)},
                    )
                    previous_logged_budget = last_budget

            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - train_start
    peak_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024**2)

    # Verify rank allocation before evaluation/saving so invalid runs are never
    # written out as if they were successful experiments.
    final_rank_distribution = None
    final_active_rank = None
    expected_target_budget = None
    if args.method != "lora":
        adapter_name = model.base_model.trainable_adapter_name
        peft_config = model.base_model.peft_config[adapter_name]
        pattern = peft_config.rank_pattern
        expected_target_budget = int(model.base_model.rankallocator.target_bgt)

        if not pattern:
            raise AssertionError(
                "AdaLoRA finished without a final rank_pattern. "
                "The final allocation step did not complete correctly."
            )

        final_rank_distribution = {
            key: int(sum(mask)) for key, mask in pattern.items()
        }
        final_active_rank = active_rank_from_pattern(pattern)

        if last_budget != expected_target_budget:
            raise AssertionError(
                f"Final budget mismatch: {last_budget} != {expected_target_budget}"
            )
        if final_active_rank != expected_target_budget:
            raise AssertionError(
                "Final active rank mismatch: "
                f"{final_active_rank} != {expected_target_budget}"
            )

    eval_metrics = evaluate_all(model, eval_loaders, device, args.task)
    checkpoint_mb = adapter_checkpoint_mb(model, out_dir)

    primary_metric_name = GLUE_TASKS[args.task]["primary_metric"]
    primary_metric = float(eval_metrics[primary_metric_name])

    result = {
        "method": args.method,
        "seed": args.seed,
        "model_name": args.model_name,
        "dataset": args.dataset,
        "task": args.task,
        "max_steps": args.max_steps,
        "primary_metric_name": primary_metric_name,
        "primary_metric": primary_metric,
        "metrics": eval_metrics,
        "eval_loss": eval_metrics["eval_loss"],
        "training_seconds": train_seconds,
        "steps_per_second": args.max_steps / train_seconds,
        "peak_gpu_allocated_mb": peak_allocated_mb,
        "peak_gpu_reserved_mb": peak_reserved_mb,
        "trainable_parameters_initial": trainable,
        "total_parameters": total,
        "trainable_fraction": trainable / total,
        "adapter_checkpoint_mb": checkpoint_mb,
        "final_budget": last_budget,
        "expected_target_budget": expected_target_budget,
        "final_active_rank": final_active_rank,
        "final_rank_distribution": final_rank_distribution,
        "resolved_target_modules": resolved_target_modules,
        "rank_init": args.rank_init,
        "rank_init_metadata": rank_init_metadata,
        "high_stability_patience": args.high_stability_patience,
        "fp16": bool(args.fp16),
        "max_grad_norm": args.max_grad_norm,
        "trainable_parameter_dtypes": trainable_dtype_summary(model),
        "git_commit": git_commit(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": package_version("transformers"),
            "peft": package_version("peft"),
            "datasets": package_version("datasets"),
            "scikit_learn": package_version("scikit-learn"),
            "scipy": package_version("scipy"),
            "gpu": torch.cuda.get_device_name(device),
        },
    }
    if "accuracy" in eval_metrics:
        result["accuracy"] = float(eval_metrics["accuracy"])

    (out_dir / "result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True)
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
