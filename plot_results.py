from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def diagnostic_plot(run_dir: Path, out: Path):
    events = load_jsonl(run_dir / "rank_events.jsonl")
    steps = [x["step"] for x in events]
    budgets = [x["budget"] for x in events]
    stability = [np.nan if x["stability"] is None else x["stability"] for x in events]

    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.plot(steps, budgets, marker="o", markersize=3, label="rank budget")
    ax1.set_xlabel("training step")
    ax1.set_ylabel("global rank budget")
    ax2 = ax1.twinx()
    ax2.plot(steps, stability, marker=".", label="Jaccard stability")
    ax2.set_ylabel("Jaccard stability")
    ax2.set_ylim(0, 1.02)
    ax1.set_title("AdaLoRA budget vs allocation stability")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.close(fig)


def _usable_metric(rows):
    if "primary_metric" in rows[0] and all(r.get("primary_metric") not in {None, ""} for r in rows):
        names = {r.get("primary_metric_name", "primary_metric") for r in rows}
        label = names.pop() if len(names) == 1 else "primary metric"
        return "primary_metric", label
    return "accuracy", "accuracy"


def summary_plot(summary_csv: Path, out: Path):
    rows = list(csv.DictReader(summary_csv.open()))
    primary_key, primary_label = _usable_metric(rows)
    metrics = [
        (primary_key, primary_label),
        ("training_seconds", "training seconds"),
        ("peak_gpu_allocated_mb", "peak GPU allocated MB"),
    ]
    methods = sorted(set(r["method"] for r in rows))
    means = defaultdict(dict)
    stds = defaultdict(dict)

    for metric, _ in metrics:
        for method in methods:
            vals = np.array(
                [float(r[metric]) for r in rows if r["method"] == method and r.get(metric) not in {None, ""}],
                dtype=float,
            )
            means[metric][method] = vals.mean()
            stds[metric][method] = vals.std(ddof=1) if len(vals) > 1 else 0.0

    out.parent.mkdir(parents=True, exist_ok=True)
    stem = out.stem
    for metric, label in metrics:
        fig, ax = plt.subplots(figsize=(7, 4))
        y = [means[metric][m] for m in methods]
        e = [stds[metric][m] for m in methods]
        ax.bar(methods, y, yerr=e, capsize=4)
        ax.set_ylabel(label)
        ax.set_title(f"{label.title()} by method")
        ax.tick_params(axis="x", rotation=20)
        fig.tight_layout()
        fig.savefig(out.with_name(f"{stem}-{metric}.png"), dpi=180)
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--diagnostic_run", help="Directory from an adalora_diag run")
    p.add_argument("--summary_csv", help="CSV produced by run_experiments.py")
    p.add_argument("--out", default="outputs/figure.png")
    args = p.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.diagnostic_run:
        diagnostic_plot(Path(args.diagnostic_run), out)
    if args.summary_csv:
        summary_plot(Path(args.summary_csv), out)


if __name__ == "__main__":
    main()
