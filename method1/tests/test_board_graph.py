"""Tests of ``method1.board_graph``: board <-> graph round trip, known edges, undo, empty graph."""

import random

import numpy as np
import pytest

from common.connect4_rules import ConnectFourEnv
from method1.board_graph import (DIAG_DOWN, DIAG_UP, HORIZONTAL, VERTICAL, BoardGraph, board_to_graph,
                                 graph_to_board)


def random_game(rng):
    """Play a random game, yielding ``(env, graph, move order)`` after every move.  Side effects: uses ``rng``."""

    env = ConnectFourEnv()
    graph = BoardGraph()
    order = []
    while not env.done:
        column, row, piece = env.step(rng.choice(env.validinputs))
        graph.add_move(column, row, piece)
        order.append((column, row))
        yield env, graph, order


@pytest.mark.parametrize("seed", range(200))
def test_round_trip_every_move(seed):
    """After every move: graph -> board equals the board, board -> graph rebuilds the same graph, edges are symmetric."""

    rng = random.Random(seed)
    for env, graph, order in random_game(rng):
        state, turn = graph_to_board(graph)
        assert np.array_equal(state, env.state)
        if not env.done:
            assert turn == env.turn
        rebuilt = board_to_graph(env.state, order)
        assert sorted(rebuilt.edges) == sorted(graph.edges)
        assert rebuilt.nodes == graph.nodes
        index = graph.edge_index
        pairs = set(zip(index[0].tolist(), index[1].tolist()))
        assert all((v, u) in pairs for u, v in pairs)


def test_known_position_edges():
    """A hand-made 4-piece position has exactly the expected edges, directions and colours."""

    g = BoardGraph()
    g.add_move(3, 0, 1)   # n0
    g.add_move(3, 1, -1)  # n1 above n0
    g.add_move(4, 0, 1)   # n2 right of n0, diag "\" to n1
    g.add_move(2, 0, -1)  # n3 left of n0, diag "/" to n1
    edges = {(u, v): (d, s) for u, v, d, s in g.undirected_edges()}
    assert edges == {
        (0, 1): (VERTICAL, 0),
        (0, 2): (HORIZONTAL, 1),
        (1, 2): (DIAG_DOWN, 0),
        (0, 3): (HORIZONTAL, 0),
        (1, 3): (DIAG_UP, 1),
    }
    assert g.edge_index.shape == (2, 10)
    assert g.x.tolist()[3] == [2, 0, -1, 3]


def test_pop_undoes_add():
    """``pop`` restores the exact nodes and edges of every earlier position."""

    rng = random.Random(7)
    snapshots = []
    graph = None
    for _, graph, _ in random_game(rng):
        snapshots.append((list(graph.nodes), sorted(graph.edges)))
    for nodes, edges in reversed(snapshots[:-1]):
        graph.pop()
        assert graph.nodes == nodes and sorted(graph.edges) == edges


def test_empty_graph():
    """The empty graph decodes to an empty board with red to move and has no edges."""

    state, turn = graph_to_board(BoardGraph())
    assert not state.any() and turn == "red"
    assert BoardGraph().edge_index.shape == (2, 0)
