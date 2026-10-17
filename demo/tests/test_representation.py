"""Tests of ``demo.representation`` on fresh tiny networks and on the default checkpoints (or fallback)."""

from pathlib import Path

import numpy as np
import pytest

from common import repo_bridge
from demo.representation import METHODS, LoadedNetwork, analyse, load_network
from method1.model import Method1Net
from method2.model import Method2Net


def fresh(kind):
    """Return a tiny untrained ``LoadedNetwork`` of method ``kind``.  Side effects: uses the torch RNG."""

    repo_bridge.register_all()
    if kind == "baseline_cnn":
        from train_alphazero.build_network import ResidualPolicyValueNet

        model = ResidualPolicyValueNet(channels=16, residual_blocks=1)
    elif kind == "method2":
        model = Method2Net(width=16, blocks=1)
    else:
        model = Method1Net(width=16, blocks=1)
    return LoadedNetwork(kind, model.eval(), Path("fresh"), 0, sum(p.numel() for p in model.parameters()))


def position():
    """Return a small test position (+1 to move) and its legal-column mask (column 7 full).  Side effects: none."""

    board = np.zeros((7, 6))
    board[3, 0], board[3, 1], board[2, 0] = 1, -1, 1
    board[6, :] = [1, -1, 1, -1, 1, -1]
    legal = np.array([True] * 6 + [False])
    return board, legal


@pytest.mark.parametrize("kind,nodes,edges", [("baseline_cnn", 42, 0), ("method2", 42, 131), ("method1", 9 + 6, None)])
def test_representation(kind, nodes, edges):
    """Node count, edge count, policy (illegal column 0, sums to 1), value range, PCA range, saliency mask."""

    board, legal = position()
    rep = analyse(fresh(kind), board, legal)
    assert rep.exists.sum() == nodes
    if edges is not None:
        assert len(rep.edges) == edges
    assert rep.policy[6] == 0 and np.isclose(rep.policy.sum(), 1.0)
    assert -1 <= rep.value <= 1
    assert rep.pca_rgb.shape == (42, 3) and (rep.pca_rgb >= 0).all() and (rep.pca_rgb <= 1).all()
    assert (rep.saliency[~rep.exists] == 0).all()


def test_method1_has_line_of_sight_edges():
    """Method 1 shows edges at distances 1, 2 and 3, only between existing nodes."""

    board, legal = position()
    rep = analyse(fresh("method1"), board, legal)
    assert set(rep.edge_counts) == {1, 2, 3}
    for i, j, _, _ in rep.edges:
        assert rep.exists[i] and rep.exists[j]


@pytest.mark.parametrize("kind", METHODS)
def test_default_networks_load(kind):
    """The default checkpoint (or the untrained fallback on a fresh clone) loads and analyses a position."""

    network = load_network(kind)
    board, legal = position()
    rep = analyse(network, board, legal)
    assert np.isclose(rep.policy.sum(), 1.0)


def test_missing_explicit_checkpoint_raises(tmp_path):
    """An explicit checkpoint path that does not exist is an error, not a silent untrained network."""

    with pytest.raises(FileNotFoundError):
        load_network("method1", tmp_path / "missing.pt")
