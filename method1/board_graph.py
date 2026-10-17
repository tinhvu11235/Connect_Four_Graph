"""Method 1's graph in explicit form: one node per placed piece, edges between touching pieces.

* **Node** = one placed piece; node id = move index (0, 1, 2, ...).
  Features ``x[i] = [column, row, player, move_index]`` with player +1 red / -1 yellow.
* **Edge** = two pieces adjacent in one of the 8 neighbour directions. Stored like
  PyTorch Geometric: ``edge_index`` of shape ``(2, E)`` containing both directions,
  ``edge_attr[e] = [direction, same_color]``.

The board matrix uses the trainer's convention ``state[column, row]``, row 0 at the bottom.
``method1.model`` computes the same piece edges on tensors (the tests check they match);
this module is what ``method1.visualize_graph`` draws.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from common.connect4_rules import BOARD_SHAPE, COLUMNS, ROWS

VERTICAL, HORIZONTAL, DIAG_UP, DIAG_DOWN = 0, 1, 2, 3
DIRECTION_NAMES = {VERTICAL: "|", HORIZONTAL: "-", DIAG_UP: "/", DIAG_DOWN: "\\"}
_NEIGHBOURS = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]


def direction_of(dx: int, dy: int) -> int:
    """Return the undirected line type (``VERTICAL``/``HORIZONTAL``/``DIAG_UP``/``DIAG_DOWN``) of an offset.

    Side effects: none.
    """

    if dx == 0:
        return VERTICAL
    if dy == 0:
        return HORIZONTAL
    return DIAG_UP if dx == dy else DIAG_DOWN


class BoardGraph:
    """Incrementally built piece graph; ``add_move``/``pop`` follow the game move by move."""

    def __init__(self) -> None:
        """Start an empty graph.  Side effects: none outside this instance."""

        self.nodes: list[tuple[int, int, int, int]] = []  # (column, row, player, move_index)
        self.edges: list[tuple[int, int, int, int]] = []  # (src, dst, direction, same_color)
        self._cell_to_node: dict[tuple[int, int], int] = {}

    # ------------------------------------------------------------------ build
    def add_move(self, column: int, row: int, player: int) -> int:
        """Append the newly placed piece as a node, link it to its existing neighbours, return its id.

        Raises ``ValueError`` if the cell already has a node.
        Side effects: appends to ``nodes`` and (both directions) to ``edges``.
        """

        if (column, row) in self._cell_to_node:
            raise ValueError(f"cell {(column, row)} already has a node")
        node = len(self.nodes)
        self.nodes.append((column, row, player, node))
        self._cell_to_node[(column, row)] = node
        for dx, dy in _NEIGHBOURS:
            other = self._cell_to_node.get((column + dx, row + dy))
            if other is None:
                continue
            direction = direction_of(dx, dy)
            same = int(self.nodes[other][2] == player)
            self.edges.append((other, node, direction, same))
            self.edges.append((node, other, direction, same))
        return node

    def pop(self) -> tuple[int, int, int, int]:
        """Remove the last node and its edges (undo the last move); return the removed node.

        Side effects: mutates ``nodes`` and ``edges``.
        """

        node = len(self.nodes) - 1
        column, row, player, index = self.nodes.pop()
        del self._cell_to_node[(column, row)]
        self.edges = [e for e in self.edges if e[0] != node and e[1] != node]
        return column, row, player, index

    # ------------------------------------------------------------ GNN tensors
    @property
    def x(self) -> np.ndarray:
        """Node feature matrix ``(num_nodes, 4)``.  Side effects: none."""

        return np.asarray(self.nodes, dtype=np.int64).reshape(-1, 4)

    @property
    def edge_index(self) -> np.ndarray:
        """PyG-style ``(2, E)`` source/target indices (both directions).  Side effects: none."""

        return np.asarray([e[:2] for e in self.edges], dtype=np.int64).reshape(-1, 2).T

    @property
    def edge_attr(self) -> np.ndarray:
        """Edge features ``(E, 2)`` = ``[direction, same_color]``.  Side effects: none."""

        return np.asarray([e[2:] for e in self.edges], dtype=np.int64).reshape(-1, 2)

    @property
    def num_nodes(self) -> int:
        """Number of nodes (= pieces on the board).  Side effects: none."""

        return len(self.nodes)

    @property
    def num_undirected_edges(self) -> int:
        """Number of undirected edges (half of ``edges``).  Side effects: none."""

        return len(self.edges) // 2

    def undirected_edges(self) -> list[tuple[int, int, int, int]]:
        """Return each undirected edge once, as ``(u, v, direction, same_color)`` with ``u < v``.

        Side effects: none.
        """

        return [e for e in self.edges if e[0] < e[1]]

    # ------------------------------------------------------------------ text
    def describe(self) -> str:
        """Return a human-readable dump of nodes, edges and ``edge_index`` (shown by the visualiser).

        Side effects: none.
        """

        lines = [f"Graph: {self.num_nodes} nodes, {self.num_undirected_edges} undirected edges "
                 f"({len(self.edges)} directed)", "",
                 "Nodes  x = [column, row, player, move_index]"]
        for i, (column, row, player, index) in enumerate(self.nodes):
            color = "red" if player == 1 else "yellow"
            lines.append(f"  n{i:<2} [{column}, {row}, {player:+d}, {index:2d}]   {color}")
        lines.append("")
        lines.append("Edges (undirected)  [u, v, direction, same_color]")
        for u, v, direction, same in self.undirected_edges():
            lines.append(f"  n{u:<2} -- n{v:<2}  dir={direction}({DIRECTION_NAMES[direction]})  same_color={same}")
        lines.append("")
        lines.append(f"edge_index (2 x {len(self.edges)}):")
        index = self.edge_index
        lines.append("  " + np.array2string(index[0], max_line_width=10_000))
        lines.append("  " + np.array2string(index[1], max_line_width=10_000))
        return "\n".join(lines)


def board_to_graph(state: np.ndarray, move_order: Sequence[tuple[int, int]] | None = None) -> BoardGraph:
    """Build the graph from a board matrix ``state[column, row]``.

    A matrix alone does not store the move order, so pass ``move_order`` (list of
    ``(column, row)``) to reproduce the incremental node ids; otherwise pieces are
    numbered bottom-up, left to right.  Raises ``ValueError`` if an ordered cell is empty.
    Side effects: none.
    """

    state = np.asarray(state)
    if move_order is None:
        move_order = [(c, r) for r in range(ROWS) for c in range(COLUMNS) if state[c, r] != 0]
    graph = BoardGraph()
    for column, row in move_order:
        player = int(state[column, row])
        if player == 0:
            raise ValueError(f"cell {(column, row)} is empty")
        graph.add_move(column, row, player)
    return graph


def from_moves(moves: Iterable[tuple[int, int, int]]) -> BoardGraph:
    """Build the graph from ``(column, row, player)`` moves in play order.  Side effects: none."""

    graph = BoardGraph()
    for column, row, player in moves:
        graph.add_move(column, row, player)
    return graph


def graph_to_board(graph: BoardGraph) -> tuple[np.ndarray, str]:
    """Decode the graph back to ``(state[column, row], turn)``; the turn follows from the node count.

    Side effects: none.
    """

    state = np.zeros(BOARD_SHAPE, dtype=np.int64)
    for column, row, player, _ in graph.nodes:
        state[column, row] = player
    turn = "red" if graph.num_nodes % 2 == 0 else "yellow"
    return state, turn
