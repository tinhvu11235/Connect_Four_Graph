"""See how each trained network represents the current Connect 4 position.

Left: the board (click to play, or let a network move).
Right, one column per network (baseline CNN, method 1, method 2):
  * the graph the network computes on - which cells are nodes, which edges carry messages
    (method 1: pieces + frontier cells only, line-of-sight arcs across empty cells);
  * node colouring: input / embedding norm / PCA of embeddings as RGB / saliency for the value;
  * the move probabilities (bars and on the frontier nodes) and the value for the side to move.

    python -m demo.app_networks [--baseline-cnn PATH] [--method1 PATH] [--method2 PATH]

A network without a checkpoint is shown untrained (its title says so).
"""

from __future__ import annotations

import argparse
import tkinter as tk
from tkinter import scrolledtext, ttk

import matplotlib

matplotlib.use("TkAgg")
import numpy as np  # noqa: E402
from matplotlib import colormaps  # noqa: E402
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import FancyArrowPatch  # noqa: E402

from common.board_canvas import PIECE_COLORS, BoardCanvas  # noqa: E402
from common.connect4_rules import COLUMNS, ROWS, ConnectFourEnv  # noqa: E402
from demo.representation import METHODS, TITLES, analyse, load_network  # noqa: E402

MODES = {
    "input": "Đầu vào (loại node)",
    "norm": "Độ lớn embedding ‖h‖",
    "pca": "PCA embedding → màu RGB",
    "saliency": "Saliency: node ảnh hưởng value",
}
DISTANCE_STYLE = {1: ("#868e96", "solid", 0.0), 2: ("#1c7ed6", "dashed", 0.35), 3: ("#9c36b5", "dotted", 0.45)}
FRONTIER = "#2b8a3e"


class NetworkDemo:
    """The demo window: the board on the left, one graph + policy panel per network on the right."""

    def __init__(self, root: tk.Tk, networks: list) -> None:
        """Build the widgets inside ``root`` for ``networks`` (``LoadedNetwork`` list) and draw the empty board.

        Side effects: creates the Tk widgets and the matplotlib figure; runs every network once.
        """

        self.root = root
        self.networks = networks
        root.title("Connect 4 · cách mỗi mạng biểu diễn trạng thái")
        self.env = ConnectFourEnv()
        self.history: list[int] = []

        left = tk.Frame(root)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=8, pady=8)
        board_box = tk.LabelFrame(left, text="Bàn cờ (click để đánh)", font=("Segoe UI", 10, "bold"))
        board_box.pack()
        self.board = BoardCanvas(board_box, "real", on_click=self.play)
        self.board.pack(padx=6, pady=6)
        self.turn_label = tk.Label(board_box, font=("Segoe UI", 11, "bold"))
        self.turn_label.pack(pady=(0, 6))

        controls = tk.Frame(left)
        controls.pack(fill=tk.X, pady=4)
        tk.Button(controls, text="↶ Undo", width=9, command=self.undo).pack(side=tk.LEFT, padx=2)
        tk.Button(controls, text="⟲ Reset", width=9, command=self.reset).pack(side=tk.LEFT, padx=2)

        ai = tk.Frame(left)
        ai.pack(fill=tk.X, pady=4)
        tk.Label(ai, text="Máy đánh bằng:").pack(side=tk.LEFT)
        self.ai_choice = ttk.Combobox(ai, state="readonly", width=26,
                                      values=[TITLES[n.kind] for n in networks])
        self.ai_choice.current(0)
        self.ai_choice.pack(side=tk.LEFT, padx=4)
        tk.Button(left, text="▶ Máy đánh 1 nước (policy lớn nhất)", command=self.network_move).pack(fill=tk.X)

        mode_box = tk.LabelFrame(left, text="Tô màu node theo", font=("Segoe UI", 10, "bold"))
        mode_box.pack(fill=tk.X, pady=8)
        self.mode = tk.StringVar(value="input")
        for key, text in MODES.items():
            tk.Radiobutton(mode_box, text=text, value=key, variable=self.mode, command=self.refresh,
                           anchor="w").pack(fill=tk.X)
        self.show_edges = tk.BooleanVar(value=True)
        tk.Checkbutton(mode_box, text="Hiện cạnh", variable=self.show_edges, command=self.refresh,
                       anchor="w").pack(fill=tk.X)

        self.status = tk.Label(left, fg="#c92a2a", wraplength=440, justify=tk.LEFT)
        self.status.pack(fill=tk.X)
        self.summary = scrolledtext.ScrolledText(left, width=60, height=16, font=("Consolas", 9))
        self.summary.pack(fill=tk.BOTH, expand=True, pady=4)

        right = tk.Frame(root)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.figure = Figure(figsize=(4.6 * len(networks), 7.4), dpi=90)
        grid = self.figure.add_gridspec(2, len(networks), height_ratios=[3.2, 1])
        self.graph_axes = [self.figure.add_subplot(grid[0, i]) for i in range(len(networks))]
        self.policy_axes = [self.figure.add_subplot(grid[1, i]) for i in range(len(networks))]
        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.refresh()

    # ------------------------------------------------------------------ game
    def play(self, column: int) -> None:
        """Play 0-based ``column`` if legal, then re-run the networks and redraw.

        Side effects: mutates ``env`` and ``history``; updates the status label; redraws.
        """

        action = column + 1
        if self.env.done:
            self.status.config(text="Ván đã kết thúc — Undo hoặc Reset.")
            return
        if action not in self.env.validinputs:
            self.status.config(text=f"Cột {action} đã đầy.")
            return
        self.env.step(action)
        self.history.append(action)
        self.status.config(text="")
        self.refresh()

    def network_move(self) -> None:
        """Play the column with the highest policy of the network chosen in the combobox.

        Side effects: same as ``play``.
        """

        if self.env.done:
            return
        rep = self.representations[self.ai_choice.current()]
        self.play(int(np.argmax(rep.policy)))

    def undo(self) -> None:
        """Undo the last move by replaying the others on a fresh board.

        Side effects: replaces ``env``; mutates ``history``; redraws.
        """

        if self.history:
            self.history.pop()
            self.env = ConnectFourEnv()
            for action in self.history:
                self.env.step(action)
            self.refresh()

    def reset(self) -> None:
        """Start a new game.  Side effects: replaces ``env``; clears ``history``; redraws."""

        self.env = ConnectFourEnv()
        self.history.clear()
        self.refresh()

    # --------------------------------------------------------------- drawing
    def refresh(self) -> None:
        """Run every network on the current position and redraw the board, graphs, policies and summary.

        Side effects: sets ``representations``; updates every widget.
        """

        sign = 1 if self.env.turn == "red" else -1
        canonical = self.env.state * sign
        legal = np.zeros(COLUMNS, dtype=bool)
        legal[np.asarray(self.env.validinputs, dtype=int) - 1] = True
        self.representations = [analyse(n, canonical, legal) for n in self.networks]

        self.board.draw(self.env.state, self.env.winning_cells, self._last_cell())
        if self.env.done:
            text = "Hoà!" if self.env.reward == 0 else ("Đỏ" if self.env.reward == 1 else "Vàng") + " thắng!"
        else:
            text = "Lượt: " + ("Đỏ" if self.env.turn == "red" else "Vàng")
        self.turn_label.config(text=text)

        heights = (self.env.state != 0).sum(1)
        for network, rep, ax, pax in zip(self.networks, self.representations, self.graph_axes, self.policy_axes):
            self._draw_graph(ax, network, rep, heights)
            self._draw_policy(pax, rep, legal)
        self.figure.tight_layout()
        self.canvas.draw_idle()
        self._write_summary(heights)

    def _last_cell(self):
        """Return ``(column, row)`` of the last piece played, or ``None``.  Side effects: none."""

        if not self.history:
            return None
        column = self.history[-1] - 1
        return column, int((self.env.state[column] != 0).sum()) - 1

    def _node_colours(self, rep, heights) -> list:
        """Return one matplotlib colour per cell for the selected colouring mode.  Side effects: none."""

        mode = self.mode.get()
        colours = []
        if mode in ("norm", "saliency"):
            values = rep.norm if mode == "norm" else rep.saliency
            top = values[rep.exists].max() if rep.exists.any() else 1.0
            scaled = values / max(top, 1e-9)
            cmap = colormaps["viridis" if mode == "norm" else "magma_r"]
        for i in range(COLUMNS * ROWS):
            column, row = divmod(i, ROWS)
            if not rep.exists[i]:
                colours.append("#f1f3f5")
            elif mode == "input":
                piece = int(self.env.state[column, row])
                colours.append(PIECE_COLORS[piece] if piece else ("#d3f9d8" if row == heights[column] else "#ffffff"))
            elif mode == "pca":
                colours.append(tuple(rep.pca_rgb[i]))
            else:
                colours.append(cmap(0.15 + 0.85 * scaled[i]))
        return colours

    def _draw_graph(self, ax, network, rep, heights) -> None:
        """Draw one network's graph (nodes, edges by distance, policy on the frontier) on ``ax``.

        Side effects: clears and redraws ``ax``.
        """

        ax.clear()
        ax.set_xlim(-0.7, COLUMNS - 0.3)
        ax.set_ylim(-0.7, ROWS - 0.2)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        if self.show_edges.get():
            if rep.kind == "baseline_cnn":
                # No explicit graph: a 3x3 convolution mixes each cell with its 8 neighbours.
                for x in range(COLUMNS):
                    ax.plot([x, x], [0, ROWS - 1], color="#dee2e6", lw=0.8, zorder=0)
                for y in range(ROWS):
                    ax.plot([0, COLUMNS - 1], [y, y], color="#dee2e6", lw=0.8, zorder=0)
            else:
                for i, j, distance, _ in rep.edges:
                    (ci, ri), (cj, rj) = divmod(i, ROWS), divmod(j, ROWS)
                    colour, style, rad = DISTANCE_STYLE[distance]
                    ax.add_patch(FancyArrowPatch((ci, ri), (cj, rj), arrowstyle="-", color=colour,
                                                 linestyle=style, lw=0.9 if distance == 1 else 1.1,
                                                 connectionstyle=f"arc3,rad={rad}", zorder=1, alpha=0.8))

        colours = self._node_colours(rep, heights)
        xs, ys, sizes, rings, widths = [], [], [], [], []
        for i in range(COLUMNS * ROWS):
            column, row = divmod(i, ROWS)
            piece = int(self.env.state[column, row])
            frontier = row == heights[column] and piece == 0
            xs.append(column)
            ys.append(row)
            if not rep.exists[i]:
                sizes.append(40)
                rings.append("#ced4da")
                widths.append(0.6)
            else:
                sizes.append(330)
                rings.append(PIECE_COLORS[piece] if piece else (FRONTIER if frontier else "#adb5bd"))
                widths.append(2.6 if piece or frontier else 0.8)
        ax.scatter(xs, ys, s=sizes, c=colours, edgecolors=rings, linewidths=widths, zorder=2)

        for column in range(COLUMNS):
            if heights[column] < ROWS and rep.policy[column] > 0:
                ax.text(column, heights[column] + 0.42, f"{rep.policy[column]:.0%}", ha="center",
                        va="center", fontsize=7, color=FRONTIER, zorder=3)
        nodes = int(rep.exists.sum())
        if rep.kind == "baseline_cnn":
            info = "42 ô · conv 3×3 (không có graph tường minh)"
        else:
            counts = rep.edge_counts
            detail = ", ".join(f"d{d}: {counts[d]}" for d in sorted(counts))
            info = f"{nodes} node · {len(rep.edges)} cạnh ({detail})"
        ax.set_title(f"{network.label}\n{info}", fontsize=8.5)

    def _draw_policy(self, ax, rep, legal) -> None:
        """Draw the policy bars and the value of one network on ``ax``.  Side effects: clears and redraws ``ax``."""

        ax.clear()
        colours = [FRONTIER if legal[c] else "#dee2e6" for c in range(COLUMNS)]
        ax.bar(range(1, COLUMNS + 1), rep.policy, color=colours)
        ax.set_ylim(0, 1)
        ax.set_xticks(range(1, COLUMNS + 1))
        ax.tick_params(labelsize=7)
        win = (rep.value + 1) / 2
        side = "Đỏ" if self.env.turn == "red" else "Vàng"
        ax.set_title(f"policy · value {rep.value:+.2f} (≈{win:.0%} cho {side})", fontsize=8.5)

    def _write_summary(self, heights) -> None:
        """Write the per-network text summary (checkpoint, size, nodes, value, best move, salient cells).

        Side effects: rewrites the summary text widget.
        """

        lines = [f"Nước đã đi: {self.history}", ""]
        for network, rep in zip(self.networks, self.representations):
            best = int(np.argmax(rep.policy)) + 1 if rep.policy.any() else "-"
            top_saliency = np.argsort(-rep.saliency)[:3]
            cells = ", ".join(f"({i // ROWS + 1},{i % ROWS})" for i in top_saliency if rep.exists[i])
            lines += [
                f"■ {network.label}",
                f"  checkpoint: {network.path or 'không có (mạng chưa train)'}",
                f"  {network.parameters / 1e6:.2f} M tham số · embedding {rep.nodes.shape[1]} chiều / node",
                f"  node trong graph: {int(rep.exists.sum())}/42 · cạnh: {len(rep.edges) if rep.kind != 'baseline_cnn' else 'ngầm (conv 3×3)'}",
                f"  value {rep.value:+.3f} · nước tốt nhất: cột {best}",
                f"  ô ảnh hưởng value nhiều nhất (cột,hàng): {cells}",
                "",
            ]
        self.summary.config(state=tk.NORMAL)
        self.summary.delete("1.0", tk.END)
        self.summary.insert(tk.END, "\n".join(lines))
        self.summary.config(state=tk.DISABLED)


def main(argv: list[str] | None = None) -> None:
    """Parse the checkpoint flags, load the three networks and run the demo window.

    Side effects: reads checkpoints; opens a window and blocks until it closes.
    """

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for kind in METHODS:
        parser.add_argument(f"--{kind.replace('_', '-')}", dest=kind, metavar="PATH",
                            help=f"{kind} checkpoint (default: runs/{kind}_full, else runs/trials_{kind}, else untrained)")
    args = parser.parse_args(argv)
    networks = [load_network(kind, getattr(args, kind)) for kind in METHODS]
    root = tk.Tk()
    NetworkDemo(root, networks)
    root.mainloop()


if __name__ == "__main__":
    main()
