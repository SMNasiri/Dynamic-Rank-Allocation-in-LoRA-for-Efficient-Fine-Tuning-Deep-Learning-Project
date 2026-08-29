from stability_adalora.scheduler import (
    apply_persistence_gate,
    choose_prune_amount,
    is_high_stability_confirmed,
    jaccard,
    required_prune,
    update_high_stability_streak,
)


def test_jaccard():
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert jaccard({"a"}, {"b"}) == 0.0
    assert jaccard({"a", "b"}, {"b", "c"}) == 1 / 3
    assert jaccard({"a"}, None) is None


def test_required_prune_guarantee_floor():
    assert required_prune(100, 80, 10) == 2
    assert required_prune(81, 80, 10) == 1
    assert required_prune(80, 80, 10) == 0


def test_high_stability_prunes_more_than_floor():
    q, q_req, q_stab = choose_prune_amount(
        current_budget=100,
        target_budget=80,
        remaining_checkpoints=10,
        stability=0.95,
        tau_low=0.70,
        tau_high=0.90,
        high_multiplier=3.0,
    )
    assert q_req == 2
    assert q_stab == 6
    assert q == 6


def test_low_stability_uses_only_deadline_floor():
    q, q_req, q_stab = choose_prune_amount(
        current_budget=100,
        target_budget=80,
        remaining_checkpoints=10,
        stability=0.50,
    )
    assert q_req == 2
    assert q_stab == 0
    assert q == 2


def test_deadline_floor_reaches_exact_target_under_persistent_low_stability():
    current = 120
    target = 80
    checkpoints = 10
    for i in range(checkpoints):
        q, _, _ = choose_prune_amount(
            current_budget=current,
            target_budget=target,
            remaining_checkpoints=checkpoints - i,
            stability=0.10,
        )
        current -= q
    assert current == target


def test_high_stability_streak_requires_three_consecutive_checkpoints():
    streak = 0
    streak = update_high_stability_streak(0.95, streak, 0.90)
    assert streak == 1
    assert not is_high_stability_confirmed(streak, 3)

    streak = update_high_stability_streak(0.94, streak, 0.90)
    assert streak == 2
    assert not is_high_stability_confirmed(streak, 3)

    streak = update_high_stability_streak(0.96, streak, 0.90)
    assert streak == 3
    assert is_high_stability_confirmed(streak, 3)


def test_high_stability_streak_resets_after_non_high_checkpoint():
    streak = 2
    streak = update_high_stability_streak(0.89, streak, 0.90)
    assert streak == 0
    assert not is_high_stability_confirmed(streak, 3)


def test_unconfirmed_high_stability_is_gated_to_medium_pruning():
    q, q_req, q_stab = choose_prune_amount(
        current_budget=100,
        target_budget=80,
        remaining_checkpoints=10,
        stability=0.95,
        tau_high=0.90,
        medium_multiplier=1.5,
        high_multiplier=3.0,
    )
    assert (q, q_req, q_stab) == (6, 2, 6)

    q, q_stab = apply_persistence_gate(
        current_budget=100,
        target_budget=80,
        stability=0.95,
        q=q,
        q_required=q_req,
        q_stability=q_stab,
        high_stability_confirmed=False,
        tau_high=0.90,
        medium_multiplier=1.5,
    )
    assert q_stab == 3
    assert q == 3


def test_confirmed_high_stability_keeps_aggressive_pruning():
    q, q_req, q_stab = choose_prune_amount(
        current_budget=100,
        target_budget=80,
        remaining_checkpoints=10,
        stability=0.95,
        high_multiplier=3.0,
    )
    q2, q_stab2 = apply_persistence_gate(
        current_budget=100,
        target_budget=80,
        stability=0.95,
        q=q,
        q_required=q_req,
        q_stability=q_stab,
        high_stability_confirmed=True,
        tau_high=0.90,
        medium_multiplier=1.5,
    )
    assert q2 == 6
    assert q_stab2 == 6
