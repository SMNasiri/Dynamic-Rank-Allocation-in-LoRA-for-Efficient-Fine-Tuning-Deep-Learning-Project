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
    p.add_argument("--task", default="sst2")
    p.add_argument("--batch_size", type=int, default=16)
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    rows = []

    for method in args.methods:
        for seed in args.seeds:
            out = root / f"{method}-seed{seed}"
            cmd = [
                sys.executable,
                "train_glue.py",
                "--method", method,
                "--seed", str(seed),
                "--max_steps", str(args.max_steps),
                "--model_name", args.model_name,
                "--task", args.task,
                "--batch_size", str(args.batch_size),
                "--output_dir", str(out),
            ]
            print("\n$", " ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)
            result = json.loads((out / "result.json").read_text())
            rows.append({
                k: result.get(k)
                for k in [
                    "method", "seed", "accuracy", "eval_loss", "training_seconds", "steps_per_second",
                    "peak_gpu_allocated_mb", "peak_gpu_reserved_mb", "adapter_checkpoint_mb", "final_budget",
                ]
            })

    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {root / 'summary.csv'}")


if __name__ == "__main__":
    main()
