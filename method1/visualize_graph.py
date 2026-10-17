"""Method 1 visualiser: Connect Four -> piece graph -> Connect Four, move by move.

Left:   the real board (click a column to play, rules enforced).
Middle: the dynamic graph built from the moves (one node per piece).
Right:  a board decoded *only* from the graph.
Bottom: node features / edge_index and both board matrices, plus a match check.

    python -m method1.visualize_graph
"""

from __future__ import annotations

import tkinter as tk
from tkinter import scrolledtext

import matplotlib

matplotlib.use("TkAgg")
import networkx as nx  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from common.board_canvas import PIECE_COLORS, BoardCanvas  # noqa: E402
from common.connect4_rules import COLUMNS, ROWS, ConnectFourEnv  # noqa: E402
from method1.board_graph import BoardGraph, graph_to_board  # noqa: E402

EDGE_COLORS = {0: "#1c7ed6", 1: "#2f9e44", 2: "#9c36b5", 3: "#e8590c"}
EDGE_LABELS = {0: "vertical |", 1: "horizontal -", 2: "diagonal /", 3: "diagonal \\"}


class App:
    """The visualiser window: real board, graph, decoded board and text dumps."""

    def __init__(self, root: tk.Tk) -> None:
        """Build every widget inside ``root`` and draw the empty position.

        Side effects: creates the Tk widgets; prints the first graph dump to the console.
        """

        self.root = root
        root.title("Connect 4  ⇄  Graph (phương pháp 1)")
        self.env = ConnectFourEnv()
        self.graph = BoardGraph()
        self.history: list[int] = []  # actions 1..7

        top = tk.Frame(root)
        top.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=6)

        left = tk.LabelFrame(top, text="① Bàn cờ thật (click để đánh)", font=("Segoe UI", 10, "bold"))
        left.grid(row=0, column=0, sticky="n", padx=4)
        self.left_board = BoardCanvas(left, "real", on_click=self.play)
        self.left_board.pack(padx=6, pady=6)
        self.turn_label = tk.Label(left, font=("Segoe UI", 11, "bold"))
        self.turn_label.pack(pady=(0, 6))

        middle = tk.LabelFrame(top, text="② Graph (board → graph)", font=("Segoe UI", 10, "bold"))
        middle.grid(row=0, column=1, sticky="nsew", padx=4)
        self.figure = Figure(figsize=(5.4, 4.6), dpi=90)
        self.axes = self.figure.add_subplot(111)
        self.fig_canvas = FigureCanvasTkAgg(self.figure, master=middle)
        self.fig_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.spring = tk.BooleanVar(value=False)
        tk.Checkbutton(middle, text="Spring layout (bỏ toạ độ lưới, xem như graph thuần)",
                       variable=self.spring, command=self.refresh).pack(pady=(0, 4))

        right = tk.LabelFrame(top, text="③ Bàn cờ dựng lại từ graph (graph → board)",
                              font=("Segoe UI", 10, "bold"))
        right.grid(row=0, column=2, sticky="n", padx=4)
        self.right_board = BoardCanvas(right, "decoded")
        self.right_board.pack(padx=6, pady=6)
        self.decoded_label = tk.Label(right, font=("Segoe UI", 11, "bold"))
        self.decoded_label.pack(pady=(0, 6))
        top.columnconfigure(1, weight=1)
        top.rowconfigure(0, weight=1)

        bar = tk.Frame(root)
        bar.pack(side=tk.TOP, fill=tk.X, padx=12)
        tk.Button(bar, text="↶ Undo", width=10, command=self.undo).pack(side=tk.LEFT, padx=4)
        tk.Button(bar, text="⟲ Reset", width=10, command=self.reset).pack(side=tk.LEFT, padx=4)
        self.match_label = tk.Label(bar, font=("Segoe UI", 12, "bold"))
        self.match_label.pack(side=tk.LEFT, padx=16)
        self.status = tk.Label(bar, fg="#c92a2a", font=("Segoe UI", 10))
        self.status.pack(side=tk.LEFT, padx=8)

        bottom = tk.Frame(root)
        bottom.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=6)
        self.graph_text = scrolledtext.ScrolledText(bottom, height=14, font=("Consolas", 9))
        self.graph_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
        self.matrix_text = scrolledtext.ScrolledText(bottom, height=14, width=58, font=("Consolas", 9))
        self.matrix_text.pack(side=tk.LEFT, fill=tk.BOTH, padx=4)

        self.refresh()

    # ----------------------------------------------------------------- actions
    def play(self, column: int) -> None:
        """Play 0-based ``column`` if legal, add the piece to the graph and redraw.

        Side effects: mutates ``env``, ``graph`` and ``history``; updates the status label; redraws.
        """

        action = column + 1
        if self.env.done:
            self.status.config(text="Ván đã kết thúc — bấm Reset hoặc Undo.")
            return
        if action not in self.env.validinputs:
            self.status.config(text=f"Cột {action} đã đầy — nước đi không hợp lệ.")
            return
        col, row, piece = self.env.step(action)
        self.graph.add_move(col, row, piece)
        self.history.append(action)
        self.status.config(text="")
        self.refresh()

    def undo(self) -> None:
        """Undo the last move: pop the graph node and replay the remaining moves on a fresh board.

        Side effects: replaces ``env``; mutates ``graph`` and ``history``; redraws.
        """

        if not self.history:
            return
        self.history.pop()
        self.graph.pop()
        self.env = ConnectFourEnv()
        for action in self.history:
            self.env.step(action)
        self.status.config(text="")
        self.refresh()

    def reset(self) -> None:
        """Start a new game.  Side effects: replaces ``env`` and ``graph``; clears ``history``; redraws."""

        self.env = ConnectFourEnv()
        self.graph = BoardGraph()
        self.history.clear()
        self.status.config(text="")
        self.refresh()

    # ----------------------------------------------------------------- drawing
    def refresh(self) -> None:
        """Redraw both boards, the graph, the labels and the text panes from the current state.

        Side effects: updates every widget; prints the graph dump and the match check to the console.
        """

        last = tuple(self.graph.nodes[-1][:2]) if self.graph.nodes else None
        decoded, decoded_turn = graph_to_board(self.graph)

        self.left_board.draw(self.env.state, self.env.winning_cells, last)
        self.right_board.draw(decoded, (), last)
        self.turn_label.config(text=self._turn_text(), fg=self._turn_color())
        self.decoded_label.config(
            text=f"{self.graph.num_nodes} node → lượt kế: {decoded_turn}",
            fg=PIECE_COLORS[1] if decoded_turn == "red" else "#b08900")

        matches = np.array_equal(decoded, self.env.state)
        self.match_label.config(text="Khớp ✓" if matches else "Không khớp ✗",
                                fg="#2b8a3e" if matches else "#c92a2a")

        self._draw_graph()
        dump = self.graph.describe()
        self._set_text(self.graph_text, dump)
        self._set_text(self.matrix_text, self._matrices_text(decoded, matches))
        print(f"\n===== Move {len(self.history)}  history={self.history} =====")
        print(dump)
        print("board == graph_to_board(graph):", matches)

    def _turn_text(self) -> str:
        """Return the turn / result text under the real board.  Side effects: none."""

        if self.env.done:
            if self.env.reward == 0:
                return "Hoà!"
            return ("Đỏ" if self.env.reward == 1 else "Vàng") + " thắng!"
        return "Lượt: " + ("Đỏ (+1)" if self.env.turn == "red" else "Vàng (−1)")

    def _turn_color(self) -> str:
        """Return the colour of the turn label (side to move, winner, or black for a draw).  Side effects: none."""

        if self.env.done and self.env.reward == 0:
            return "black"
        side = self.env.reward if self.env.done else (1 if self.env.turn == "red" else -1)
        return PIECE_COLORS[1] if side == 1 else "#b08900"

    def _draw_graph(self) -> None:
        """Draw the piece graph with networkx: grid coordinates or a spring layout, edges by direction.

        Side effects: clears and redraws the middle matplotlib figure.
        """

        ax = self.axes
        ax.clear()
        g = nx.Graph()
        for i, (column, row, player, _) in enumerate(self.graph.nodes):
            g.add_node(i, column=column, row=row, player=player)
        for u, v, direction, same in self.graph.undirected_edges():
            g.add_edge(u, v, direction=direction, same=same)

        if self.spring.get() and g.number_of_nodes() > 0:
            pos = nx.spring_layout(g, seed=1)
            ax.set_aspect("equal", adjustable="datalim")
        else:
            pos = {i: (d["column"], d["row"]) for i, d in g.nodes(data=True)}
            ax.set_xlim(-0.6, COLUMNS - 0.4)
            ax.set_ylim(-0.6, ROWS - 0.4)
            ax.set_xticks(range(COLUMNS), [str(c) for c in range(COLUMNS)])
            ax.set_yticks(range(ROWS))
            ax.set_xlabel("column")
            ax.set_ylabel("row")
            ax.grid(True, alpha=0.15)
            ax.set_aspect("equal")

        for direction, color in EDGE_COLORS.items():
            for same, style in ((1, "solid"), (0, "dashed")):
                edges = [(u, v) for u, v, d in g.edges(data=True)
                         if d["direction"] == direction and d["same"] == same]
                if edges:
                    nx.draw_networkx_edges(g, pos, edgelist=edges, ax=ax, edge_color=color,
                                           style=style, width=2.2 if same else 1.2)
        if g.number_of_nodes():
            nodes = list(g.nodes)
            newest = nodes[-1]
            nx.draw_networkx_nodes(g, pos, nodelist=nodes, ax=ax, node_size=420,
                                   node_color=[PIECE_COLORS[g.nodes[n]["player"]] for n in nodes],
                                   edgecolors=["black" if n == newest else "#555" for n in nodes],
                                   linewidths=[3.0 if n == newest else 1.0 for n in nodes])
            nx.draw_networkx_labels(g, pos, ax=ax, labels={n: str(n) for n in nodes}, font_size=8)
        if self.spring.get():
            ax.set_axis_off()

        handles = [Line2D([], [], color=c, lw=2, label=EDGE_LABELS[d]) for d, c in EDGE_COLORS.items()]
        handles += [Line2D([], [], color="gray", lw=2, label="cùng màu"),
                    Line2D([], [], color="gray", lw=1, ls="--", label="khác màu")]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7, frameon=False)
        ax.set_title(f"{self.graph.num_nodes} nodes · {self.graph.num_undirected_edges} edges", fontsize=10)
        self.figure.tight_layout()
        self.fig_canvas.draw_idle()

    def _matrices_text(self, decoded: np.ndarray, matches: bool) -> str:
        """Return the text of both board matrices side by side and the equality check.  Side effects: none."""

        def as_rows(state: np.ndarray) -> str:
            """Return ``state[column, row]`` as the board looks (top row first).  Side effects: none."""

            return "\n".join(" ".join(f"{int(v):+d}" if v else " 0" for v in state[:, row])
                             for row in reversed(range(ROWS)))

        return "\n".join([
            "Bàn cờ thật  env.state[column, row]  (7 x 6):",
            np.array2string(self.env.state, separator=" "),
            "",
            "graph_to_board(graph)  (7 x 6):",
            np.array2string(decoded, separator=" "),
            "",
            "Nhìn như bàn cờ (hàng trên cùng trước)   thật | từ graph:",
            *(f"  {a}   |   {b}" for a, b in zip(as_rows(self.env.state).splitlines(),
                                               as_rows(decoded).splitlines())),
            "",
            f"np.array_equal(board, graph_to_board(graph))  ->  {matches}",
        ])

    @staticmethod
    def _set_text(widget: scrolledtext.ScrolledText, text: str) -> None:
        """Replace the content of a read-only text widget.  Side effects: rewrites ``widget``."""

        widget.config(state=tk.NORMAL)
        widget.delete("1.0", tk.END)
        widget.insert(tk.END, text)
        widget.config(state=tk.DISABLED)


def main() -> None:
    """Open the visualiser window and run the Tk event loop until it is closed.

    Side effects: opens a window; blocks until it closes.
    """

    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
