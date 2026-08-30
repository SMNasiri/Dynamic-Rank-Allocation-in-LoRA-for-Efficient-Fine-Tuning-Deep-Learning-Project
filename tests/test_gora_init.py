import torch
from torch import nn

from stability_adalora.gora_init import (
    allocate_gora_ranks,
    collect_target_linear_modules,
    compute_gora_importance,
    run_gradient_probe,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_proj = nn.Linear(4, 4, bias=False)
        self.key_proj = nn.Linear(4, 4, bias=False)
        self.classifier = nn.Linear(4, 2, bias=False)


def test_collect_target_linear_modules_uses_suffix_matching():
    model = TinyModel()
    selected = collect_target_linear_modules(model, ["query_proj", "key_proj"])
    assert list(selected) == ["query_proj", "key_proj"]


def test_allocate_gora_ranks_reproduces_weighted_allocation_for_equal_shapes():
    modules = {
        "a": nn.Linear(4, 4, bias=False),
        "b": nn.Linear(4, 4, bias=False),
    }
    ranks, advantages, total_budget = allocate_gora_ranks(
        modules,
        {"a": 1.0, "b": 3.0},
        reference_rank=8,
        min_rank=1,
        max_rank=32,
    )
    assert advantages == {"a": 0.25, "b": 0.75}
    assert total_budget == 64.0
    assert ranks == {"a": 4, "b": 12}


def test_allocate_gora_ranks_applies_min_and_max_clipping():
    modules = {
        "small": nn.Linear(4, 4, bias=False),
        "large": nn.Linear(4, 4, bias=False),
    }
    ranks, _, _ = allocate_gora_ranks(
        modules,
        {"small": 1e-9, "large": 1.0},
        reference_rank=8,
        min_rank=4,
        max_rank=10,
    )
    assert ranks["small"] == 4
    assert ranks["large"] == 10


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.query_proj = nn.Linear(2, 2, bias=False)
        self.classifier = nn.Linear(2, 2, bias=False)

    def forward(self, input_ids, labels):
        hidden = self.query_proj(input_ids.float())
        logits = self.classifier(hidden)
        loss = nn.functional.cross_entropy(logits, labels)
        return type("Output", (), {"loss": loss})()


def test_gradient_probe_restores_requires_grad_and_produces_importance():
    model = TinyClassifier()
    selected = collect_target_linear_modules(model, ["query_proj"])
    batches = [
        {
            "input_ids": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            "labels": torch.tensor([0, 1]),
        },
        {
            "input_ids": torch.tensor([[1.0, 1.0], [1.0, -1.0]]),
            "labels": torch.tensor([1, 0]),
        },
    ]

    gradients, batches_used = run_gradient_probe(
        model,
        batches,
        selected,
        n_batches=2,
        device=torch.device("cpu"),
    )

    assert batches_used == 2
    assert torch.count_nonzero(gradients["query_proj"]).item() > 0
    assert all(parameter.requires_grad for parameter in model.parameters())

    importance = compute_gora_importance(gradients, selected)
    assert importance["query_proj"] > 0
