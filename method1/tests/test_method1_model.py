"""Tests of ``method1.model``: nodes, edges and line-of-sight features of method 1, and the network itself."""

import random

import numpy as np
import pytest
import torch

from common.connect4_rules import ConnectFourEnv
from common.graph_layers import DIRECTIONS, cell
from method1.board_graph import BoardGraph, direction_of
from method1.model import LEGACY_ARCHITECTURES, Method1Net, active_edges, build, model_config


def random_positions(count, seed=0):
    """Yield ``count`` random ``(canonical states (1, 1, 7, 6), BoardGraph)`` pairs.  Side effects: none."""

    rng = random.Random(seed)
    for _ in range(count):
        env, graph = ConnectFourEnv(), BoardGraph()
        for _ in range(rng.randint(0, 30)):
            if env.done:
                break
            graph.add_move(*env.step(rng.choice(env.validinputs)))
        sign = 1 if env.turn == "red" else -1
        yield torch.tensor(env.state * sign, dtype=torch.float32)[None, None], graph


def test_piece_edges_match_visualiser_graph():
    """The network's edges between pieces are exactly ``BoardGraph``'s edges."""

    for states, graph in random_positions(60):
        board = states[0, 0]
        edges = active_edges(states)[0]
        between_pieces = set()
        for i, j, r in edges:
            ci, ri, cj, rj = i // 6, i % 6, j // 6, j % 6
            if board[ci, ri] != 0 and board[cj, rj] != 0:
                dx, dy = DIRECTIONS[r]
                between_pieces.add((min(i, j), max(i, j), direction_of(dx, dy)))
        expected = set()
        for u, v, direction, _ in graph.undirected_edges():
            a = cell(*graph.nodes[u][:2])
            b = cell(*graph.nodes[v][:2])
            expected.add((min(a, b), max(a, b), direction))
        assert between_pieces == expected


def test_nodes_are_pieces_plus_frontier():
    """The node mask holds the pieces and the 7 landing cells, nothing else."""

    states = torch.zeros(1, 1, 7, 6)
    states[0, 0, 3, 0] = 1
    states[0, 0, 3, 1] = -1
    model = Method1Net(width=16, blocks=1)
    board = model.board_features(states)
    exists = model.node_mask(board)[0]
    present = {(i // 6, i % 6) for i in torch.nonzero(exists).flatten().tolist()}
    assert present == {(3, 0), (3, 1), (3, 2)} | {(c, 0) for c in range(7) if c != 3}


def test_line_of_sight_counts_empty_gap():
    """Line-of-sight edge features count the empty cells between two existing nodes (/2)."""

    states = torch.zeros(1, 1, 7, 6)
    states[0, 0, 0, 0] = 1
    states[0, 0, 3, 0] = 1
    states[0, 0, 1, 0] = 1  # X X _ X along row 0
    model = Method1Net(width=16, blocks=1, line_of_sight=True)
    board = model.board_features(states)
    mask = model.node_mask(board).float()[..., None]
    edge = model.edge_features(board, mask)[0]
    east = DIRECTIONS.index((1, 0))
    relation = east * 3 + 2  # distance 3
    assert edge[cell(0, 0), relation] == pytest.approx(0.5)  # one empty cell (2,0) in between
    assert edge[cell(1, 0), east * 3 + 1] == pytest.approx(0.5)  # (1,0) -> (3,0) over (2,0)


@pytest.mark.parametrize("line_of_sight", [False, True])
def test_forward_backward(line_of_sight):
    """Output shapes are right, values finite, and every parameter gets a gradient."""

    model = build(model_config(32, 2, line_of_sight))
    states = torch.cat([s for s, _ in random_positions(8, seed=3)] + [torch.zeros(1, 1, 7, 6)])
    logits, value = model(states)
    assert logits.shape == (9, 7) and value.shape == (9,)
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()
    (logits.sum() + value.sum()).backward()
    assert all(p.grad is not None for p in model.parameters())


def test_batch_independent():
    """A position gives the same output alone or inside a batch."""

    model = Method1Net(width=32, blocks=2).eval()
    states = torch.cat([s for s, _ in random_positions(5, seed=4)])
    logits, value = model(states)
    alone_logits, alone_value = model(states[2:3])
    assert torch.allclose(logits[2], alone_logits[0], atol=1e-5)
    assert np.isclose(value[2].item(), alone_value[0].item(), atol=1e-5)


def test_absent_cells_have_zero_input():
    """Absent cells (empty, not playable) enter the network as all-zero features."""

    model = Method1Net(width=16, blocks=1)
    states = torch.zeros(1, 1, 7, 6)
    board = model.board_features(states)
    mask = model.node_mask(board).float()[..., None]
    assert (board["features"] * mask)[0, cell(0, 3)].abs().sum() == 0


def test_legacy_architecture_tag_still_builds():
    """Checkpoints saved under the old tag (``gnn_dynamic_v1``) still build a ``Method1Net``."""

    config = {**model_config(16, 1, True), "architecture": LEGACY_ARCHITECTURES[0]}
    assert isinstance(build(config), Method1Net)
