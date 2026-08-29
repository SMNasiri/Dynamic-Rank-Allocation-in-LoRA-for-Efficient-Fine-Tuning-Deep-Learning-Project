from __future__ import annotations

import argparse
from collections import Counter

import torch
from transformers import AutoModel


def resolve_target_modules(model_name: str, preset: str, custom: list[str] | None) -> list[str]:
    if preset == "custom":
        if not custom:
            raise ValueError("--preset custom requires --target_modules")
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
    if "deberta" in model_name.lower():
        return [
            "query_proj",
            "key_proj",
            "value_proj",
            "intermediate.dense",
            "output.dense",
        ]
    return ["query", "value"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", required=True)
    p.add_argument(
        "--preset",
        choices=["auto", "qv", "deberta_paper", "custom"],
        default="auto",
    )
    p.add_argument("--target_modules", nargs="+", default=None)
    p.add_argument("--init_r", type=int, default=12)
    p.add_argument("--target_r", type=int, default=4)
    args = p.parse_args()

    targets = resolve_target_modules(
        args.model_name,
        args.preset,
        args.target_modules,
    )
    model = AutoModel.from_pretrained(args.model_name)

    matches = []
    matched_suffixes = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        suffix = next((target for target in targets if name.endswith(target)), None)
        if suffix is not None:
            matches.append(name)
            matched_suffixes.append(suffix)

    print("Model:", args.model_name)
    print("Preset:", args.preset)
    print("Target suffixes:", targets)
    print("Matched Linear modules:", len(matches))
    print("Matches by suffix:", dict(Counter(matched_suffixes)))
    print()
    for name in matches:
        print(name)

    if not matches:
        raise SystemExit("No target Linear modules matched. Do not train with this preset.")

    print()
    print("Expected initial global budget:", len(matches) * args.init_r)
    print("Expected target global budget:", len(matches) * args.target_r)


if __name__ == "__main__":
    main()
