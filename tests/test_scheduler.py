from stability_adalora.scheduler import choose_prune_amount, jaccard, required_prune


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
