"""Tkinter widget that draws a Connect Four board; shared by both GUIs.

Used by ``method1.visualize_graph`` (real board + board decoded from the graph)
and ``demo.app_networks`` (board next to each network's graph).
"""

from __future__ import annotations

import tkinter as tk

import numpy as np

from common.connect4_rules import COLUMNS, ROWS

CELL = 64
PAD = 10
BOARD_W = COLUMNS * CELL + 2 * PAD
BOARD_H = ROWS * CELL + 2 * PAD
PIECE_COLORS = {1: "#e03131", -1: "#f5c518", 0: "#ffffff"}


class BoardCanvas(tk.Canvas):
    """A board canvas; clickable (column -> ``on_click(column)``) when ``on_click`` is given."""

    def __init__(self, master, title: str, on_click=None) -> None:
        """Create the canvas inside ``master`` and, if ``on_click`` is set, bind mouse events.

        Side effects: creates a Tk widget (not yet packed); binds click/motion/leave events.
        """

        super().__init__(master, width=BOARD_W, height=BOARD_H, bg="#1864ab", highlightthickness=0)
        self.title = title
        self.on_click = on_click
        self.hover_column: int | None = None
        if on_click is not None:
            self.bind("<Button-1>", self._click)
            self.bind("<Motion>", self._motion)
            self.bind("<Leave>", lambda _: self._set_hover(None))
            self.configure(cursor="hand2")
        self._last = None

    def _column_at(self, x: int) -> int | None:
        """Return the 0-based column under canvas x-coordinate ``x``, or ``None`` outside the board.

        Side effects: none.
        """

        column = (x - PAD) // CELL
        return column if 0 <= column < COLUMNS else None

    def _click(self, event) -> None:
        """Mouse click handler: forwards the clicked column to ``on_click``.

        Side effects: whatever ``on_click`` does (usually plays a move and redraws the GUI).
        """

        column = self._column_at(event.x)
        if column is not None:
            self.on_click(column)

    def _motion(self, event) -> None:
        """Mouse move handler: highlights the column under the cursor.

        Side effects: may redraw this canvas (see ``_set_hover``).
        """

        self._set_hover(self._column_at(event.x))

    def _set_hover(self, column) -> None:
        """Set the highlighted column (``None`` for none) and redraw the last board if it changed.

        Side effects: updates ``hover_column``; redraws this canvas.
        """

        if column != self.hover_column:
            self.hover_column = column
            if self._last is not None:
                self.draw(*self._last)

    def draw(self, state: np.ndarray, highlight=(), last=None) -> None:
        """Draw ``state[column, row]``; ``highlight`` cells get a green ring, ``last`` a black one.

        Side effects: clears and redraws this canvas; remembers the arguments for hover redraws.
        """

        self._last = (state, highlight, last)
        self.delete("all")
        if self.hover_column is not None:
            x0 = PAD + self.hover_column * CELL
            self.create_rectangle(x0, PAD, x0 + CELL, BOARD_H - PAD, fill="#1971c2", outline="")
        highlight = set(highlight)
        for column in range(COLUMNS):
            for row in range(ROWS):
                x0 = PAD + column * CELL + 6
                y0 = PAD + (ROWS - 1 - row) * CELL + 6
                value = int(state[column, row])
                width, outline = 1, "#0b3d70"
                if (column, row) in highlight:
                    width, outline = 5, "#2b8a3e"
                elif last == (column, row):
                    width, outline = 4, "#000000"
                self.create_oval(x0, y0, x0 + CELL - 12, y0 + CELL - 12,
                                 fill=PIECE_COLORS[value], outline=outline, width=width)
        for column in range(COLUMNS):
            self.create_text(PAD + column * CELL + CELL / 2, BOARD_H - 3, text=str(column + 1),
                             fill="white", font=("Segoe UI", 8))
