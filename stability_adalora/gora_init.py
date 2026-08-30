from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable, Mapping

import torch
from torch import nn


@dataclass(frozen=True)
class GoraRankInitSettings:
    """Settings copied from the GoRA notebook's rank-initialization stage."""

    reference_rank: int = 8
    min_rank: int = 4
    max_rank: int = 32
    probe_batches: int = 64

    def validate(self) -> None:
        if self.reference_rank < 1:
            raise ValueError("gora_reference_rank must be >= 1")
        if self.min_rank < 1:
            raise ValueError("gora_min_rank must be >= 1")
        if self.max_rank < self.min_rank:
            raise ValueError("gora_max_rank must be >= gora_min_rank")
        if self.probe_batches < 1:
            raise ValueError("gora_probe_batches must be >= 1")

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def _matches_target(name: str, target_modules: Iterable[str]) -> bool:
    """Approximate PEFT's list-based target-module suffix matching."""
    return any(name == target or name.endswith(f".{target}") for target in target_modules)


def collect_target_linear_modules(
    model: nn.Module,
    target_modules: Iterable[str],
) -> dict[str, nn.Linear]:
    """Return the base-model Linear layers that will later receive AdaLoRA adapters."""
    targets = tuple(target_modules)
    selected = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and _matches_target(name, targets)
    }
    if not selected:
        raise ValueError(
            "GoRA probe found no Linear layers matching target_modules="
            f"{list(targets)}"
        )
    return selected


def run_gradient_probe(
    model: nn.Module,
    probe_dataloader,
    target_modules: Mapping[str, nn.Linear],
    *,
    n_batches: int,
    device: torch.device,
    max_grad_norm: float = 1.0,
) -> tuple[dict[str, torch.Tensor], int]:
    """
    Reproduce the notebook's GoRA probe: average full-weight gradients for the
    selected base-model Linear layers before PEFT adapters are created.
    """
    if n_batches < 1:
        raise ValueError("n_batches must be >= 1")

    model.to(device)
    was_training = model.training
    original_requires_grad = {
        name: parameter.requires_grad for name, parameter in model.named_parameters()
    }
    model.train()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    probe_params: dict[str, torch.nn.Parameter] = {}
    for name, module in target_modules.items():
        module.weight.requires_grad_(True)
        probe_params[name] = module.weight

    accumulated_grads = {
        name: torch.zeros_like(parameter, device="cpu", dtype=torch.float32)
        for name, parameter in probe_params.items()
    }

    batches_used = 0
    try:
        for batch_idx, batch in enumerate(probe_dataloader):
            if batch_idx >= n_batches:
                break

            batches_used += 1
            batch = {key: value.to(device) for key, value in batch.items()}

            for parameter in probe_params.values():
                parameter.grad = None

            outputs = model(**batch)
            loss = getattr(outputs, "loss", None)
            if loss is None:
                raise RuntimeError("GoRA gradient probe requires model outputs with a loss.")
            if not torch.isfinite(loss).item():
                raise RuntimeError(
                    "Non-finite loss during GoRA gradient probe at batch "
                    f"{batch_idx}: {loss.detach().float().item()}"
                )

            loss.backward()
            if max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(probe_params.values()),
                    max_norm=max_grad_norm,
                    error_if_nonfinite=True,
                )

            for name, parameter in probe_params.items():
                if parameter.grad is not None:
                    accumulated_grads[name] += parameter.grad.detach().float().cpu()
                    parameter.grad = None

        if batches_used == 0:
            raise RuntimeError("GoRA gradient probe received an empty training dataloader.")

        for name in accumulated_grads:
            accumulated_grads[name] /= batches_used
    finally:
        model.zero_grad(set_to_none=True)
        for name, parameter in model.named_parameters():
            parameter.requires_grad_(original_requires_grad[name])
        model.train(was_training)

    return accumulated_grads, batches_used


def compute_gora_importance(
    accumulated_grads: Mapping[str, torch.Tensor],
    target_modules: Mapping[str, nn.Linear],
) -> dict[str, float]:
    """Notebook importance: mean(abs(W * averaged_gradient)) for each target layer."""
    importance: dict[str, float] = {}
    for name, gradient in accumulated_grads.items():
        if name not in target_modules:
            raise KeyError(f"Missing target module for accumulated gradient: {name}")
        weight = target_modules[name].weight.detach().float().cpu()
        importance[name] = (weight * gradient.float().cpu()).abs().mean().item()
    return importance


def allocate_gora_ranks(
    target_modules: Mapping[str, nn.Linear],
    importance: Mapping[str, float],
    *,
    reference_rank: int = 8,
    min_rank: int = 4,
    max_rank: int = 32,
) -> tuple[dict[str, int], dict[str, float], float]:
    """Reproduce the rank-allocation rule from the supplied GoRA notebook."""
    if reference_rank < 1:
        raise ValueError("reference_rank must be >= 1")
    if min_rank < 1 or max_rank < min_rank:
        raise ValueError("Require 1 <= min_rank <= max_rank")
    if set(target_modules) != set(importance):
        missing_importance = sorted(set(target_modules) - set(importance))
        missing_modules = sorted(set(importance) - set(target_modules))
        raise ValueError(
            "GoRA target/importance keys differ. "
            f"Missing importance={missing_importance[:5]}, "
            f"missing modules={missing_modules[:5]}"
        )

    total_importance = float(sum(importance.values()))
    if not math.isfinite(total_importance) or total_importance <= 0:
        raise ValueError(
            "GoRA importance sum must be finite and positive; "
            f"got {total_importance}."
        )

    advantages = {
        name: float(score) / total_importance for name, score in importance.items()
    }

    total_budget = 0.0
    for module in target_modules.values():
        out_features, in_features = module.weight.shape
        total_budget += (
            math.sqrt(out_features) + math.sqrt(in_features)
        ) * reference_rank

    rank_pattern: dict[str, int] = {}
    for name, module in target_modules.items():
        out_features, in_features = module.weight.shape
        raw_rank = (
            total_budget
            * advantages[name]
            / (math.sqrt(out_features) + math.sqrt(in_features))
        )
        rank = round(raw_rank)
        rank_pattern[name] = max(min_rank, min(max_rank, rank))

    return rank_pattern, advantages, total_budget


def _resolve_rank_for_wrapped_module(
    module_name: str,
    rank_pattern: Mapping[str, int],
) -> int:
    if module_name in rank_pattern:
        return int(rank_pattern[module_name])

    suffix_matches = [
        int(rank)
        for original_name, rank in rank_pattern.items()
        if module_name.endswith(f".{original_name}")
        or original_name.endswith(f".{module_name}")
    ]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    if len(suffix_matches) > 1:
        raise ValueError(f"Ambiguous GoRA rank mapping for wrapped module {module_name}")
    raise KeyError(f"No GoRA rank found for wrapped AdaLoRA module {module_name}")


def apply_gora_ranks_to_adalora(
    peft_model,
    rank_pattern: Mapping[str, int],
) -> dict[str, int]:
    """
    Physically resize AdaLoRA's initial A/E/B tensors to the GoRA per-layer ranks.

    The native PEFT allocator is then rebuilt so its initial budget is computed
    from the resized tensors rather than from the temporary bootstrap rank.
    """
    adalora_model = peft_model.base_model
    adapter_name = adalora_model.trainable_adapter_name
    config = adalora_model.peft_config[adapter_name]
    backbone = adalora_model.model

    resize_pattern: dict[str, list[bool]] = {}
    mapped_modules: dict[str, int] = {}

    for parameter_name, parameter in backbone.named_parameters():
        marker = f".lora_E.{adapter_name}"
        if marker not in parameter_name:
            continue
        module_name = parameter_name.split(marker, 1)[0]
        desired_rank = _resolve_rank_for_wrapped_module(module_name, rank_pattern)
        current_rank = int(parameter.shape[0])
        if desired_rank < 1:
            raise ValueError(f"GoRA rank must be positive for {module_name}")
        if desired_rank > current_rank:
            raise ValueError(
                f"GoRA rank {desired_rank} for {module_name} exceeds temporary "
                f"AdaLoRA bootstrap rank {current_rank}."
            )
        resize_pattern[parameter_name] = [
            index < desired_rank for index in range(current_rank)
        ]
        mapped_modules[module_name] = desired_rank

    if not resize_pattern:
        raise RuntimeError("No AdaLoRA lora_E parameters were found for GoRA rank resizing.")
    if len(mapped_modules) != len(rank_pattern):
        missing = sorted(set(rank_pattern) - set(mapped_modules))
        # The direct names normally match. Give a useful error if PEFT naming changes.
        if missing:
            unresolved = [
                name
                for name in rank_pattern
                if not any(
                    wrapped.endswith(f".{name}") or name.endswith(f".{wrapped}")
                    for wrapped in mapped_modules
                )
            ]
            if unresolved:
                raise RuntimeError(
                    "Not every GoRA rank could be mapped to an AdaLoRA layer: "
                    f"{unresolved[:10]}"
                )

    adalora_model.resize_modules_by_rank_pattern(resize_pattern, adapter_name)

    # PEFT's resize helper preserves the previous ranknum for checkpoint-loading
    # semantics. For a genuine *initial* variable-rank model, ranknum should be
    # each layer's new physical starting rank.
    applied: dict[str, int] = {}
    for module_name, module in backbone.named_modules():
        if not hasattr(module, "ranknum") or adapter_name not in module.ranknum:
            continue
        desired_rank = _resolve_rank_for_wrapped_module(module_name, rank_pattern)
        module.ranknum[adapter_name].data.fill_(float(desired_rank))
        applied[module_name] = desired_rank

    if len(applied) != len(rank_pattern):
        raise RuntimeError(
            f"Applied GoRA ranks to {len(applied)} AdaLoRA layers, "
            f"but expected {len(rank_pattern)}."
        )

    # rank_pattern is reserved by PEFT for masks learned later during AdaLoRA.
    config.rank_pattern = None

    old_allocator = adalora_model.rankallocator
    allocator_cls = type(old_allocator)
    adalora_model.rankallocator = allocator_cls(backbone, config, adapter_name)

    expected_budget = sum(applied.values())
    actual_budget = int(adalora_model.rankallocator.init_bgt)
    if actual_budget != expected_budget:
        raise AssertionError(
            "AdaLoRA allocator initial budget does not match GoRA ranks: "
            f"{actual_budget} != {expected_budget}"
        )

    return applied
