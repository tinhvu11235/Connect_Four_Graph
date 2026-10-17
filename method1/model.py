"""Method 1 network: the dynamic graph of ``method1.board_graph``, made trainable.

Nodes that exist in a position:
* every placed piece (the nodes of ``board_graph.BoardGraph``);
* the 7 frontier cells (where each column's next piece lands) - the policy reads them;
* a global readout (masked mean), so the empty board still has something to compute.
Every other empty cell is *absent*: masked to zero, it neither sends nor receives.

``line_of_sight`` (default) also links two existing nodes up to 3 cells apart on a line,
across empty cells, with the number of empty cells in between as an edge feature; this is
how the piece-only graph still sees threats like ``X X _ X``.  Without it the edges are
exactly ``BoardGraph``'s (8-neighbour adjacency), with direction kept.
"""

from __future__ import annotations

import torch
from torch import Tensor

from common.graph_layers import PAD, GraphPolicyValueNet, base_model_config, neighbour_table

ARCHITECTURE = "method1_v1"
LEGACY_ARCHITECTURES = ("gnn_dynamic_v1",)  # tag of checkpoints saved before the rename
LINE_OF_SIGHT_DISTANCE = 3


class Method1Net(GraphPolicyValueNet):
    """Method 1: pieces + frontier cells as nodes; optional line-of-sight edges up to 3 cells."""

    def __init__(self, *, width: int = 160, blocks: int = 6, line_of_sight: bool = True) -> None:
        """Create the network; ``line_of_sight=False`` keeps only distance-1 edges and no edge features.

        Side effects: registers the ``between`` cell table as a non-persistent buffer.
        """

        distance = LINE_OF_SIGHT_DISTANCE if line_of_sight else 1
        super().__init__(width=width, blocks=blocks, max_distance=distance, edge_features=bool(line_of_sight))
        _, between = neighbour_table(distance)
        self.register_buffer("between", between, persistent=False)
        self.line_of_sight = bool(line_of_sight)
        self.model_config = model_config(width, blocks, line_of_sight)

    def node_mask(self, board: dict[str, Tensor]) -> Tensor:
        """Return ``(N, 42)`` bool: a node exists where a piece is or where the next piece can land.

        Side effects: none.
        """

        return board["occupied"] | board["playable"]

    def edge_features(self, board: dict[str, Tensor], mask: Tensor) -> Tensor | None:
        """Return the line-of-sight edge features ``(N, 42, R)``, or ``None`` without line of sight.

        Side effects: none.
        """

        if not self.line_of_sight:
            return None
        return line_of_sight_features(board["features"][..., 2], mask[..., 0], self.nbr, self.between)


def line_of_sight_features(empty: Tensor, exists: Tensor, nbr: Tensor, between: Tensor) -> Tensor:
    """Return ``(N, 42, R)``: number of empty cells between ``i`` and its relation-``r`` node, divided by 2.

    ``empty``/``exists`` are ``(N, 42)``.  Zero when the far node does not exist, so absent
    nodes carry no edge either.  Side effects: none.
    """

    batch = empty.shape[0]
    pad = empty.new_zeros(batch, 1)
    empty_padded = torch.cat([empty, pad], dim=1)
    exists_padded = torch.cat([exists, pad], dim=1)
    gaps = empty_padded[:, between].sum(-1)  # (N, 42, R)
    return gaps * exists_padded[:, nbr] / 2.0


def active_edges(states: Tensor, *, line_of_sight: bool = False) -> list[set[tuple[int, int, int]]]:
    """Return, per position, the directed edges ``(i, j, relation)`` between existing nodes.

    Used by the tests to compare with ``BoardGraph``.  Side effects: none (builds a tiny throwaway model).
    """

    model = Method1Net(width=16, blocks=1, line_of_sight=line_of_sight)
    board = model.board_features(states)
    exists = model.node_mask(board)
    nbr = model.nbr
    result = []
    for b in range(states.shape[0]):
        edges = set()
        for i in torch.nonzero(exists[b]).flatten().tolist():
            for r, j in enumerate(nbr[i].tolist()):
                if j != PAD and exists[b, j]:
                    edges.add((i, j, r))
        result.append(edges)
    return result


def model_config(width: int, blocks: int, line_of_sight: bool) -> dict:
    """Return the ``model_config`` dict stored in checkpoints for a method 1 network.

    Side effects: none.
    """

    return {**base_model_config(ARCHITECTURE, width, blocks), "line_of_sight": bool(line_of_sight)}


def build(config: dict) -> Method1Net:
    """Build a fresh ``Method1Net`` from a ``model_config`` (current or legacy tag).

    Raises ``ValueError`` for another architecture.  Side effects: none (random init).
    """

    if config.get("architecture") not in (ARCHITECTURE, *LEGACY_ARCHITECTURES):
        raise ValueError(f"not a {ARCHITECTURE} config: {config!r}")
    return Method1Net(
        width=config["channels"], blocks=config["residual_blocks"], line_of_sight=config.get("line_of_sight", True)
    )
