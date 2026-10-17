"""Tests of ``method2.model``: the fixed grid graph and the network."""

import torch

from common.graph_layers import PAD, cell, neighbour_table
from method2.model import LEGACY_ARCHITECTURES, Method2Net, build, model_config


def test_neighbour_table():
    """Corner cells have 3 neighbours, inner cells 8, and relation 0 points north."""

    nbr, _ = neighbour_table(1)
    assert nbr.shape == (42, 8)
    assert (nbr[cell(0, 0)] != PAD).sum() == 3
    assert (nbr[cell(3, 2)] != PAD).sum() == 8
    north = 0
    assert nbr[cell(3, 2), north] == cell(3, 3)


def test_all_cells_are_nodes():
    """Method 2 has no mask: every cell is a node."""

    model = Method2Net(width=16, blocks=1)
    board = model.board_features(torch.zeros(2, 1, 7, 6))
    assert model.node_mask(board) is None


def test_forward_backward_and_playable_feature():
    """Shapes and gradients are right; the 'playable' feature is 0 on a full column and 1 on a landing cell."""

    model = build(model_config(32, 2))
    states = torch.zeros(3, 1, 7, 6)
    states[1, 0, 2, :6] = torch.tensor([1, -1, 1, -1, 1, -1.0])  # full column
    logits, value = model(states)
    assert logits.shape == (3, 7) and value.shape == (3,)
    (logits.sum() + value.sum()).backward()
    assert all(p.grad is not None for p in model.parameters())
    features = model.board_features(states)["features"]
    assert features[1, cell(2, 5), 3] == 0 and features[1, cell(3, 0), 3] == 1


def test_parameter_count_close_to_cnn():
    """At 256x6 the network is about the size of the 256x6 CNN (10 W^2 per layer vs 9 W^2)."""

    params = sum(p.numel() for p in Method2Net(width=256, blocks=6).parameters())
    assert 7.0e6 < params < 9.5e6


def test_legacy_architecture_tag_still_builds():
    """Checkpoints saved under the old tag (``gnn_grid_v1``) still build a ``Method2Net``."""

    config = {**model_config(16, 1), "architecture": LEGACY_ARCHITECTURES[0]}
    assert isinstance(build(config), Method2Net)
