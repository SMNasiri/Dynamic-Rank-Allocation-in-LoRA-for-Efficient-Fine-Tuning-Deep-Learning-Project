from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", default=["adalora", "extended_cubic", "stability"])
    p.add_argument("--seeds", nargs="+", type=int, default=[13, 42, 87])
    p.add_argument("--max_steps", type=int, default=1500)
    p.add_argument("--root", default="outputs/main")
    p.add_argument("--model_name", default="roberta-base")
    p.add_argument("--dataset", default="nyu-mll/glue")
    p.add_argument("--task", default="sst2")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--learning_rate", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--init_r", type=int, default=12)
    p.add_argument("--target_r", type=int, default=4)
    p.add_argument("--rank_init", choices=["uniform", "gora"], default="uniform")
    p.add_argument("--gora_reference_rank", type=int, default=8)
    p.add_argument("--gora_min_rank", type=int, default=4)
    p.add_argument("--gora_max_rank", type=int, default=32)
    p.add_argument("--gora_probe_batches", type=int, default=64)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--adalora_warmup_ratio", type=float, default=0.10)
    p.add_argument("--adalora_final_ratio", type=float, default=0.20)
    p.add_argument("--adaptive_final_ratio", type=float, default=0.05)
    p.add_argument("--tinit_steps", type=int, default=None)
    p.add_argument("--tfinal_steps", type=int, default=None)
    p.add_argument("--deltaT", type=int, default=10)
    p.add_argument("--beta1", type=float, default=0.85)
    p.add_argument("--beta2", type=float, default=0.85)
    p.add_argument("--orth_reg_weight", type=float, default=0.5)
    p.add_argument(
        "--target_preset",
        choices=["auto", "qv", "deberta_paper", "custom"],
        default="auto",
    )
    p.add_argument("--target_modules", nargs="+", default=None)
    p.add_argument("--stability_policy", choices=["binary", "multilevel"], default="multilevel")
    p.add_argument("--tau_low", type=float, default=0.70)
    p.add_argument("--tau_high", type=float, default=0.90)
    p.add_argument("--medium_multiplier", type=float, default=1.5)
    p.add_argument("--high_multiplier", type=float, default=3.0)
    p.add_argument("--topk_reference", choices=["target", "current"], default="target")
    p.add_argument("--high_stability_patience", type=int, default=3)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=True)
    return p.parse_args()


def first_target_step(out: Path, final_budget: int | None) -> int | None:
    path = out / "budget_trace.jsonl"
    if final_budget is None or not path.exists():
        return None
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if int(event["budget"]) == int(final_budget):
            return int(event["step"])
    return None


def main():
    args = parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    rows = []

    # Seed-major order keeps different methods closer together in time,
    # which makes Colab runtime comparisons less sensitive to session drift.
    for seed in args.seeds:
        for method in args.methods:
            run_label = method if args.rank_init == "uniform" else f"{method}-{args.rank_init}"
            out = root / f"{run_label}-seed{seed}"
            cmd = [
                sys.executable,
                "train_glue.py",
                "--method", method,
                "--seed", str(seed),
                "--max_steps", str(args.max_steps),
                "--model_name", args.model_name,
                "--dataset", args.dataset,
                "--task", args.task,
                "--batch_size", str(args.batch_size),
                "--eval_batch_size", str(args.eval_batch_size),
                "--max_length", str(args.max_length),
                "--learning_rate", str(args.learning_rate),
                "--weight_decay", str(args.weight_decay),
                "--warmup_ratio", str(args.warmup_ratio),
                "--init_r", str(args.init_r),
                "--target_r", str(args.target_r),
                "--rank_init", args.rank_init,
                "--gora_reference_rank", str(args.gora_reference_rank),
                "--gora_min_rank", str(args.gora_min_rank),
                "--gora_max_rank", str(args.gora_max_rank),
                "--gora_probe_batches", str(args.gora_probe_batches),
                "--lora_alpha", str(args.lora_alpha),
                "--adalora_warmup_ratio", str(args.adalora_warmup_ratio),
                "--adalora_final_ratio", str(args.adalora_final_ratio),
                "--adaptive_final_ratio", str(args.adaptive_final_ratio),
                "--deltaT", str(args.deltaT),
                "--beta1", str(args.beta1),
                "--beta2", str(args.beta2),
                "--orth_reg_weight", str(args.orth_reg_weight),
                "--target_preset", args.target_preset,
                "--stability_policy", args.stability_policy,
                "--tau_low", str(args.tau_low),
                "--tau_high", str(args.tau_high),
                "--medium_multiplier", str(args.medium_multiplier),
                "--high_multiplier", str(args.high_multiplier),
                "--topk_reference", args.topk_reference,
                "--high_stability_patience", str(args.high_stability_patience),
                "--max_grad_norm", str(args.max_grad_norm),
                "--output_dir", str(out),
            ]
            if args.tinit_steps is not None:
                cmd += ["--tinit_steps", str(args.tinit_steps)]
            if args.tfinal_steps is not None:
                cmd += ["--tfinal_steps", str(args.tfinal_steps)]
            if args.target_preset == "custom":
                if not args.target_modules:
                    raise ValueError("custom target preset requires --target_modules")
                cmd += ["--target_modules", *args.target_modules]
            if not args.fp16:
                cmd += ["--no-fp16"]

            print("\n$", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)
            result = json.loads((out / "result.json").read_text())
            target_step = first_target_step(out, result.get("final_budget"))
            rank_init_metadata = result.get("rank_init_metadata") or {}
            rows.append(
                {
                    "method": result.get("method"),
                    "rank_init": result.get("rank_init"),
                    "seed": result.get("seed"),
                    "model_name": result.get("model_name"),
                    "task": result.get("task"),
                    "primary_metric_name": result.get("primary_metric_name"),
                    "primary_metric": result.get("primary_metric"),
                    "accuracy": result.get("accuracy"),
                    "eval_loss": result.get("eval_loss"),
                    "training_seconds": result.get("training_seconds"),
                    "steps_per_second": result.get("steps_per_second"),
                    "peak_gpu_allocated_mb": result.get("peak_gpu_allocated_mb"),
                    "peak_gpu_reserved_mb": result.get("peak_gpu_reserved_mb"),
                    "adapter_checkpoint_mb": result.get("adapter_checkpoint_mb"),
                    "final_budget": result.get("final_budget"),
                    "final_active_rank": result.get("final_active_rank"),
                    "initial_total_rank": rank_init_metadata.get("initial_total_rank"),
                    "allocator_init_budget": rank_init_metadata.get("allocator_init_budget"),
                    "target_budget_step": target_step,
                    "high_stability_patience": result.get("high_stability_patience"),
                    "resolved_target_modules": json.dumps(result.get("resolved_target_modules")),
                }
            )

    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {root / 'summary.csv'}")


if __name__ == "__main__":
    main()
