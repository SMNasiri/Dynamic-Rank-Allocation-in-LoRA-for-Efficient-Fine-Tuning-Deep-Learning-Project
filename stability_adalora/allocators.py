from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch

from .compat import assert_peft_compat

assert_peft_compat()
from peft.tuners.adalora.layer import RankAllocator

from .scheduler import choose_prune_amount, jaccard

@dataclass
class StabilitySettings:
    policy: Literal["binary", "multilevel"] = "multilevel"
    tau_low: float = 0.70
    tau_high: float = 0.90
    medium_multiplier: float = 1.5
    high_multiplier: float = 3.0
    topk_reference: Literal["target", "current"] = "target"

    # V2:
    # Require high stability for this many consecutive checkpoints
    # before aggressive pruning is allowed.
    high_stability_patience: int = 1


class JsonlLogger:
    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("")

    def write(self, record: dict) -> None:
        if not self.path:
            return
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")


class ScoreExtractionMixin:
    """Extract the exact triplet scores PEFT computes inside mask_to_budget()."""

    def triplet_scores(self, model) -> dict[str, torch.Tensor]:
        value_ipt: dict[str, torch.Tensor] = {}
        vector_ipt: dict[str, list[torch.Tensor]] = {}

        for n, _ in model.named_parameters():
            if f"lora_A.{self.adapter_name}" in n:
                entry_ipt = self._element_score(n)
                comb_ipt = torch.mean(entry_ipt, dim=1, keepdim=True)
                name_m = n.replace("lora_A", "%s")
                vector_ipt.setdefault(name_m, []).append(comb_ipt)
            elif f"lora_B.{self.adapter_name}" in n:
                entry_ipt = self._element_score(n)
                comb_ipt = torch.mean(entry_ipt, dim=0, keepdim=False).view(-1, 1)
                name_m = n.replace("lora_B", "%s")
                vector_ipt.setdefault(name_m, []).append(comb_ipt)
            elif f"lora_E.{self.adapter_name}" in n:
                entry_ipt = self._element_score(n)
                name_m = n.replace("lora_E", "%s")
                value_ipt[name_m] = entry_ipt

        triplet_ipt: dict[str, torch.Tensor] = {}
        for name_m, vectors in vector_ipt.items():
            ipt_E = value_ipt[name_m]
            ipt_AB = torch.cat(vectors, dim=1)
            sum_ipt = self._combine_ipt(ipt_E, ipt_AB)
            triplet_ipt[name_m % "lora_E"] = sum_ipt.view(-1)
        return triplet_ipt

    def global_top_set(self, model, k: int) -> frozenset[str]:
        items: list[tuple[float, str]] = []
        for name, scores in self.triplet_scores(model).items():
            values = scores.detach().float().cpu().tolist()
            items.extend((float(score), f"{name}::{idx}") for idx, score in enumerate(values))
        items.sort(key=lambda x: x[0], reverse=True)
        k = min(max(0, int(k)), len(items))
        return frozenset(component_id for _, component_id in items[:k])

    def rank_distribution(self, model, rank_pattern: dict | None) -> dict[str, int] | None:
        if not rank_pattern:
            return None
        return {name: int(sum(mask)) for name, mask in rank_pattern.items()}


class DiagnosticRankAllocator(ScoreExtractionMixin, RankAllocator):
    """Native AdaLoRA schedule + read-only Jaccard diagnostics."""

    def __init__(self, model, peft_config, adapter_name, log_path=None, topk_reference="target"):
        super().__init__(model, peft_config, adapter_name)
        self.logger = JsonlLogger(log_path)
        self.previous_top_set: frozenset[str] | None = None
        self.topk_reference = topk_reference

    def update_and_allocate(self, model, global_step, force_mask=False):
        if global_step < self.peft_config.total_step - self.peft_config.tfinal:
            self.update_ipt(model)

        budget, mask_ind = super().budget_schedule(global_step)
        rank_pattern = None
        stability = None

        if mask_ind or force_mask:
            k = self.target_bgt if self.topk_reference == "target" else budget
            current_top_set = self.global_top_set(model, k)
            stability = jaccard(current_top_set, self.previous_top_set)
            self.previous_top_set = current_top_set
            rank_pattern = self.mask_to_budget(model, budget)
            self.logger.write(
                {
                    "method": "adalora_diag",
                    "step": int(global_step),
                    "budget": int(budget),
                    "stability": stability,
                    "q": None,
                    "q_required": None,
                    "q_stability": None,
                    "topk": int(k),
                    "rank_distribution": self.rank_distribution(model, rank_pattern),
                }
            )

        return budget, rank_pattern


class ExtendedCubicRankAllocator(RankAllocator):
    """Cubic AdaLoRA with the shorter tfinal supplied by the experiment config."""

    def __init__(self, model, peft_config, adapter_name, log_path=None):
        super().__init__(model, peft_config, adapter_name)
        self.logger = JsonlLogger(log_path)

    def update_and_allocate(self, model, global_step, force_mask=False):
        if global_step < self.peft_config.total_step - self.peft_config.tfinal:
            self.update_ipt(model)
        budget, mask_ind = super().budget_schedule(global_step)
        rank_pattern = None
        if mask_ind or force_mask:
            rank_pattern = self.mask_to_budget(model, budget)
            self.logger.write(
                {
                    "method": "extended_cubic",
                    "step": int(global_step),
                    "budget": int(budget),
                    "stability": None,
                    "q": None,
                    "q_required": None,
                    "q_stability": None,
                    "rank_distribution": {name: int(sum(mask)) for name, mask in rank_pattern.items()},
                }
            )
        return budget, rank_pattern


class StabilityAwareRankAllocator(ScoreExtractionMixin, RankAllocator):
    """AdaLoRA rank allocator that changes only the temporal budget-reduction policy."""

    def __init__(
        self,
        model,
        peft_config,
        adapter_name,
        *,
        settings: StabilitySettings | None = None,
        log_path=None,
    ):
        super().__init__(model, peft_config, adapter_name)
        self.settings = settings or StabilitySettings()
        self.logger = JsonlLogger(log_path)
        self.current_budget = int(self.init_bgt)
        self.previous_top_set: frozenset[str] | None = None
        self.last_stability: float | None = None
        self.high_stability_streak = 0
    def _remaining_checkpoints(self, step: int) -> int:
        stabilization_start = self.peft_config.total_step - self.peft_config.tfinal
        if step >= stabilization_start:
            return 1
        return max(1, math.ceil((stabilization_start - step) / self.peft_config.deltaT))

    def _topk(self) -> int:
        if self.settings.topk_reference == "current":
            return self.current_budget
        return self.target_bgt

    def update_and_allocate(self, model, global_step, force_mask=False):
        cfg = self.peft_config
        stabilization_start = cfg.total_step - cfg.tfinal

        if global_step < stabilization_start:
            self.update_ipt(model)

        # Stage 1: warm-up, unchanged.
        if global_step <= cfg.tinit:
            self.current_budget = self.init_bgt
            return self.current_budget, None

        # Stage 3 boundary: guarantee exact final target before fixed stabilization.
        if global_step >= stabilization_start:
            self.current_budget = self.target_bgt
            rank_pattern = self.mask_to_budget(model, self.current_budget) if force_mask else None
            if force_mask:
                self.logger.write(
                {
                    "method": "stability",
                    "step": int(global_step),
                    "budget": int(self.current_budget),
                    "stability": self.last_stability,
                    "q": None,
                    "q_required": None,
                    "q_stability": None,
                    "topk": int(self._topk()),
                    "settings": asdict(self.settings),
                    "rank_distribution": self.rank_distribution(
                        model, rank_pattern
                    ),

                    # Keep logging schema consistent
                    "high_stability_streak": int(
                        self.high_stability_streak
                    ),
                    "high_stability_confirmed": bool(
                        self.high_stability_streak
                        >= self.settings.high_stability_patience
                    ),
                }        
            )
            return self.current_budget, rank_pattern

        # Stage 2: only make allocation decisions every deltaT, exactly like AdaLoRA.
        if global_step % cfg.deltaT != 0 and not force_mask:
            return self.current_budget, None

        current_top_set = self.global_top_set(model, self._topk())
        stability = jaccard(current_top_set, self.previous_top_set)
        if stability is not None and stability >= self.settings.tau_high:
            self.high_stability_streak += 1
        else:
            self.high_stability_streak = 0

        high_stability_confirmed = (
            self.high_stability_streak
            >= self.settings.high_stability_patience
        )
        remaining_checkpoints = self._remaining_checkpoints(global_step)
        q, q_required, q_stability = choose_prune_amount(
            current_budget=self.current_budget,
            target_budget=self.target_bgt,
            remaining_checkpoints=remaining_checkpoints,
            stability=stability,
            policy=self.settings.policy,
            tau_low=self.settings.tau_low,
            tau_high=self.settings.tau_high,
            medium_multiplier=self.settings.medium_multiplier,
            high_multiplier=self.settings.high_multiplier,
        )
        # V2: a single high-stability checkpoint is not enough
        # to trigger aggressive pruning.
        if (
        stability is not None
        and stability >= self.settings.tau_high
        and not high_stability_confirmed
        ):
            q_stability = math.ceil(
            self.settings.medium_multiplier * q_required
            )

            remaining_rank = max(
            0,
            self.current_budget - self.target_bgt
            )

            q = min(
                remaining_rank,
                max(q_required, q_stability)
            )
            
        self.current_budget = max(self.target_bgt, self.current_budget - q)
        rank_pattern = self.mask_to_budget(model, self.current_budget)
        self.previous_top_set = current_top_set
        self.last_stability = stability

        self.logger.write(
            {
                "method": "stability",
                "step": int(global_step),
                "budget": int(self.current_budget),
                "stability": stability,
                "q": int(q),
                "q_required": int(q_required),
                "q_stability": int(q_stability),
                "topk": int(self._topk()),
                "remaining_checkpoints": int(remaining_checkpoints),
                "settings": asdict(self.settings),
                "rank_distribution": self.rank_distribution(model, rank_pattern),
                "high_stability_streak": int(self.high_stability_streak),
            "high_stability_confirmed": bool(high_stability_confirmed), 
            }
        )
        return self.current_budget, rank_pattern

def install_custom_allocator(peft_model, method: str, log_path: str | Path, settings: StabilitySettings | None = None):
    """Replace only AdaLoRA's RankAllocator; all decomposition/scoring/masking code stays in PEFT."""
    adalora_model = peft_model.base_model
    adapter_name = adalora_model.trainable_adapter_name
    config = adalora_model.peft_config[adapter_name]
    backbone = adalora_model.model

    if method == "adalora_diag":
        allocator = DiagnosticRankAllocator(backbone, config, adapter_name, log_path=log_path)
    elif method == "extended_cubic":
        allocator = ExtendedCubicRankAllocator(backbone, config, adapter_name, log_path=log_path)
    elif method == "stability":
        allocator = StabilityAwareRankAllocator(
            backbone,
            config,
            adapter_name,
            settings=settings,
            log_path=log_path,
        )
    else:
        raise ValueError(f"No custom allocator for method={method}")

    adalora_model.rankallocator = allocator
    return allocator
