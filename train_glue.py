from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
    set_seed,
)

from peft import AdaLoraConfig, LoraConfig, TaskType, get_peft_model
from stability_adalora.allocators import StabilitySettings, install_custom_allocator


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["lora", "adalora", "adalora_diag", "extended_cubic", "stability"], required=True)
    p.add_argument("--model_name", default="roberta-base")
    p.add_argument("--dataset", default="nyu-mll/glue")
    p.add_argument("--task", default="sst2")
    p.add_argument("--max_steps", type=int, default=600)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--learning_rate", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.06, help="Optimizer LR warmup, not AdaLoRA rank warm-up.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--init_r", type=int, default=12)
    p.add_argument("--target_r", type=int, default=4)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--adalora_warmup_ratio", type=float, default=0.10)
    p.add_argument("--adalora_final_ratio", type=float, default=0.20)
    p.add_argument("--adaptive_final_ratio", type=float, default=0.05)
    p.add_argument("--deltaT", type=int, default=10)
    p.add_argument("--beta1", type=float, default=0.85)
    p.add_argument("--beta2", type=float, default=0.85)
    p.add_argument("--orth_reg_weight", type=float, default=0.5)
    p.add_argument("--target_modules", nargs="+", default=["query", "value"])
    p.add_argument("--stability_policy", choices=["binary", "multilevel"], default="multilevel")
    p.add_argument("--tau_low", type=float, default=0.70)
    p.add_argument("--tau_high", type=float, default=0.90)
    p.add_argument("--medium_multiplier", type=float, default=1.5)
    p.add_argument("--high_multiplier", type=float, default=3.0)
    p.add_argument("--topk_reference", choices=["target", "current"], default="target")
    p.add_argument("--output_dir", default="outputs/run")
    p.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
    "--high_stability_patience",
    type=int,
    default=3,
    help="Number of consecutive high-stability checkpoints required before aggressive pruning.",
)
    return p.parse_args()


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


def build_model(args):
    base = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    if args.method == "lora":
        config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=args.target_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.0,
            target_modules=args.target_modules,
            bias="none",
        )
        return get_peft_model(base, config), None

    tinit = int(round(args.adalora_warmup_ratio * args.max_steps))
    final_ratio = args.adalora_final_ratio if args.method in {"adalora", "adalora_diag"} else args.adaptive_final_ratio
    tfinal = int(round(final_ratio * args.max_steps))
    config = AdaLoraConfig(
        task_type=TaskType.SEQ_CLS,
        init_r=args.init_r,
        target_r=args.target_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.0,
        target_modules=args.target_modules,
        tinit=tinit,
        tfinal=tfinal,
        deltaT=args.deltaT,
        beta1=args.beta1,
        beta2=args.beta2,
        orth_reg_weight=args.orth_reg_weight,
        total_step=args.max_steps,
    )
    model = get_peft_model(base, config)
    return model, config


def prepare_data(args, tokenizer):
    ds = load_dataset(args.dataset, args.task)
    sentence_keys = {
        "sst2": ("sentence", None),
        "cola": ("sentence", None),
        "mrpc": ("sentence1", "sentence2"),
        "qqp": ("question1", "question2"),
        "qnli": ("question", "sentence"),
        "rte": ("sentence1", "sentence2"),
        "mnli": ("premise", "hypothesis"),
    }
    if args.task not in sentence_keys:
        raise ValueError(f"This starter supports GLUE tasks: {sorted(sentence_keys)}")
    k1, k2 = sentence_keys[args.task]

    def tok(batch):
        if k2 is None:
            return tokenizer(batch[k1], truncation=True, max_length=args.max_length)
        return tokenizer(batch[k1], batch[k2], truncation=True, max_length=args.max_length)

    tokenized = ds.map(tok, batched=True)
    cols_to_remove = [c for c in tokenized["train"].column_names if c not in {"input_ids", "attention_mask", "token_type_ids", "label"}]
    tokenized = tokenized.remove_columns(cols_to_remove)
    tokenized = tokenized.rename_column("label", "labels")
    # tokenized.set_format("torch")
    collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8 if args.fp16 else None)
    train_loader = DataLoader(tokenized["train"], batch_size=args.batch_size, shuffle=True, collate_fn=collator)
    eval_split = "validation_matched" if args.task == "mnli" else "validation"
    eval_loader = DataLoader(tokenized[eval_split], batch_size=args.eval_batch_size, shuffle=False, collate_fn=collator)
    return train_loader, eval_loader


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        bs = batch["labels"].shape[0]
        total += bs
        correct += (out.logits.argmax(dim=-1) == batch["labels"]).sum().item()
        loss_sum += out.loss.item() * bs
    model.train()
    return {"accuracy": correct / total, "eval_loss": loss_sum / total}


def count_trainable(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def adapter_checkpoint_mb(model, out_dir: Path):
    ckpt = out_dir / "adapter"
    model.save_pretrained(ckpt)
    total_bytes = sum(p.stat().st_size for p in ckpt.rglob("*") if p.is_file())
    return total_bytes / (1024**2)


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True))
    set_seed(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("This experiment harness is intended for a Colab GPU runtime.")
    device = torch.device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    train_loader, eval_loader = prepare_data(args, tokenizer)
    model, adalora_config = build_model(args)
    model.to(device)

    allocator = None
    event_path = out_dir / "rank_events.jsonl"

    if args.method in {
        "adalora_diag",
        "extended_cubic",
        "stability",
    }:
        settings = StabilitySettings(
            policy=args.stability_policy,
            tau_low=args.tau_low,
            tau_high=args.tau_high,
            medium_multiplier=args.medium_multiplier,
            high_multiplier=args.high_multiplier,
            topk_reference=args.topk_reference,

            # V1 = 1 checkpoint
            # V2 = 2 consecutive high-stability checkpoints
            high_stability_patience=args.high_stability_patience,
        )

        # stability_v2 uses the same StabilityAwareRankAllocator class.
        # The difference is only high_stability_patience=2.
        allocator_method = args.method

        allocator = install_custom_allocator(
            model,
            allocator_method,
            event_path,
            settings=settings,
        )

    trainable, total = count_trainable(model)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=args.weight_decay)
    lr_warmup = int(round(args.warmup_ratio * args.max_steps))
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, lr_warmup, args.max_steps)

    use_amp = bool(args.fp16)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    train_start = time.perf_counter()

    step = 0
    epoch = 0
    last_budget = None
    while step < args.max_steps:
        epoch += 1
        for batch in train_loader:
            if step >= args.max_steps:
                break
            step += 1
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = model(**batch).loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()

            # PEFT AdaLoRA requires this after backward/optimizer step while gradients still exist.
            if args.method != "lora":
                model.base_model.update_and_allocate(step)
                ra = model.base_model.rankallocator
                if args.method == "stability":
                    last_budget = int(ra.current_budget)
                else:
                    last_budget = int(ra.budget_schedule(step)[0])

            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

    torch.cuda.synchronize()
    train_seconds = time.perf_counter() - train_start
    peak_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024**2)

    eval_metrics = evaluate(model, eval_loader, device)
    checkpoint_mb = adapter_checkpoint_mb(model, out_dir)

    final_rank_distribution = None
    if args.method != "lora":
        pattern = model.base_model.peft_config[model.base_model.trainable_adapter_name].rank_pattern
        if pattern:
            final_rank_distribution = {k: int(sum(v)) for k, v in pattern.items()}

    result = {
        "method": args.method,
        "seed": args.seed,
        "model_name": args.model_name,
        "task": args.task,
        "max_steps": args.max_steps,
        "accuracy": eval_metrics["accuracy"],
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
        "final_rank_distribution": final_rank_distribution,
        "git_commit": git_commit(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": package_version("transformers"),
            "peft": package_version("peft"),
            "datasets": package_version("datasets"),
            "gpu": torch.cuda.get_device_name(device),
        },
    }
    (out_dir / "result.json").write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
