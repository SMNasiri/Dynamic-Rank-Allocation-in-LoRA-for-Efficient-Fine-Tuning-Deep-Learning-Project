from __future__ import annotations

import math
from typing import AbstractSet, Literal


def jaccard(a: AbstractSet[str] | None, b: AbstractSet[str] | None) -> float | None:
    if a is None or b is None:
        return None
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def required_prune(current_budget: int, target_budget: int, remaining_checkpoints: int) -> int:
    """Deadline-aware minimum pruning: ceil((b_t-b_T)/N_t)."""
    if current_budget <= target_budget:
        return 0
    remaining_checkpoints = max(1, int(remaining_checkpoints))
    return math.ceil((current_budget - target_budget) / remaining_checkpoints)


def stability_prune(
    stability: float | None,
    q_required: int,
    policy: Literal["binary", "multilevel"] = "multilevel",
    tau_low: float = 0.70,
    tau_high: float = 0.90,
    medium_multiplier: float = 1.5,
    high_multiplier: float = 3.0,
) -> int:
    """Pruning suggested by stability. The deadline floor is applied separately."""
    if stability is None or q_required == 0:
        return 0

    if policy == "binary":
        return 0 if stability < tau_high else math.ceil(high_multiplier * q_required)

    if policy != "multilevel":
        raise ValueError(f"Unknown stability policy: {policy}")

    if stability < tau_low:
        return 0
    if stability < tau_high:
        return math.ceil(medium_multiplier * q_required)
    return math.ceil(high_multiplier * q_required)


def choose_prune_amount(
    *,
    current_budget: int,
    target_budget: int,
    remaining_checkpoints: int,
    stability: float | None,
    policy: Literal["binary", "multilevel"] = "multilevel",
    tau_low: float = 0.70,
    tau_high: float = 0.90,
    medium_multiplier: float = 1.5,
    high_multiplier: float = 3.0,
) -> tuple[int, int, int]:
    """Return (actual_q, q_required, q_stability), clipped to remaining rank."""
    remaining_rank = max(0, current_budget - target_budget)
    q_req = required_prune(current_budget, target_budget, remaining_checkpoints)
    q_stab = stability_prune(
        stability,
        q_req,
        policy=policy,
        tau_low=tau_low,
        tau_high=tau_high,
        medium_multiplier=medium_multiplier,
        high_multiplier=high_multiplier,
    )
    q = min(remaining_rank, max(q_req, q_stab))
    return q, q_req, q_stab


def update_high_stability_streak(
    stability: float | None,
    current_streak: int,
    tau_high: float,
) -> int:
    """Update the number of consecutive checkpoints with stability >= tau_high."""
    if stability is not None and stability >= tau_high:
        return int(current_streak) + 1
    return 0


def is_high_stability_confirmed(streak: int, patience: int) -> bool:
    """Return True once the high-stability streak reaches the configured patience."""
    if patience < 1:
        raise ValueError("high_stability_patience must be >= 1")
    return int(streak) >= int(patience)


def apply_persistence_gate(
    *,
    current_budget: int,
    target_budget: int,
    stability: float | None,
    q: int,
    q_required: int,
    q_stability: int,
    high_stability_confirmed: bool,
    tau_high: float,
    medium_multiplier: float,
) -> tuple[int, int]:
    """
    Prevent an unconfirmed high-stability checkpoint from immediately using the
    aggressive high-stability multiplier. Until persistence is confirmed, treat
    the checkpoint as medium-confidence pruning.

    Returns (actual_q, adjusted_q_stability).
    """
    if (
        stability is None
        or stability < tau_high
        or high_stability_confirmed
        or q_required == 0
    ):
        return q, q_stability

    adjusted_q_stability = math.ceil(medium_multiplier * q_required)
    remaining_rank = max(0, current_budget - target_budget)
    adjusted_q = min(remaining_rank, max(q_required, adjusted_q_stability))
    return adjusted_q, adjusted_q_stability
