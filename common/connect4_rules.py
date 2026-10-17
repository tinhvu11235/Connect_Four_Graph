"""Connect Four rules, same conventions as ``ConnectFourEnv`` in AlphaZeroConnect4Train.

* board ``state[column, row]``, shape ``(7, 6)``, row 0 at the bottom;
* actions are columns ``1..7``;
* red pieces +1, yellow -1; ``reward`` is +1 red win, -1 yellow win, 0 draw;
* after a win or a draw ``turn`` is not switched.

Used by the GUIs (``method1.visualize_graph``, ``demo.app_networks``), the
benchmark's random positions and the tests.  Training itself uses the AlphaZero
repo's own bitboard rules.
"""

from __future__ import annotations

import numpy as np

COLUMNS = 7
ROWS = 6
BOARD_SHAPE = (COLUMNS, ROWS)
_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


def piece_of(turn: str) -> int:
    """Return the board value of a player: ``"red"`` -> +1, ``"yellow"`` -> -1.

    Side effects: none.
    """

    return 1 if turn == "red" else -1


class ConnectFourEnv:
    """One game of Connect Four with gravity, legal moves, win and draw detection."""

    COLUMNS = COLUMNS
    ROWS = ROWS

    def __init__(self) -> None:
        """Start an empty board with red to move.

        Side effects: sets every game attribute (``turn``, ``validinputs``, ``heights``,
        ``state``, ``done``, ``reward``, ``winning_cells``) on this instance.
        """

        self.turn = "red"
        self.validinputs = list(range(1, COLUMNS + 1))
        self.heights = [0] * COLUMNS
        self.state = np.zeros(BOARD_SHAPE, dtype=np.int64)
        self.done = False
        self.reward = 0
        self.winning_cells: list[tuple[int, int]] = []

    def step(self, action: int) -> tuple[int, int, int]:
        """Drop a piece into column ``action`` (1..7) and return ``(column, row, piece)``.

        Raises ``RuntimeError`` if the game is over and ``ValueError`` for an illegal column.
        Side effects: mutates ``state``, ``heights`` and ``validinputs``; on a win or a draw
        sets ``done``, ``reward`` (and ``winning_cells`` for a win), otherwise switches ``turn``.
        """

        if self.done:
            raise RuntimeError("cannot play a move: the game is finished")
        if action not in self.validinputs:
            raise ValueError(f"illegal action {action!r}; legal={self.validinputs}")
        column = int(action) - 1
        row = self.heights[column]
        piece = piece_of(self.turn)
        self.state[column, row] = piece
        self.heights[column] += 1
        if self.heights[column] == ROWS:
            self.validinputs.remove(action)
        line = self._four_through(column, row)
        if line:
            self.done = True
            self.reward = piece
            self.validinputs = []
            self.winning_cells = line
        elif not self.validinputs:
            self.done = True
            self.reward = 0
        else:
            self.turn = "yellow" if self.turn == "red" else "red"
        return column, row, piece

    def _four_through(self, column: int, row: int) -> list[tuple[int, int]]:
        """Return the cells of a line of 4+ same-colour pieces through ``(column, row)``, or ``[]``.

        Side effects: none (reads ``state`` only).
        """

        piece = self.state[column, row]
        for dx, dy in _DIRECTIONS:
            cells = [(column, row)]
            for sign in (-1, 1):
                x, y = column + sign * dx, row + sign * dy
                while 0 <= x < COLUMNS and 0 <= y < ROWS and self.state[x, y] == piece:
                    cells.append((x, y))
                    x, y = x + sign * dx, y + sign * dy
            if len(cells) >= 4:
                return cells
        return []
