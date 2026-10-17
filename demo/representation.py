"""What each network "sees" and builds for one position (the data behind ``demo.app_networks``).

For every network the result has one entry per cell (node ``column * 6 + row``):
whether that node exists in the network's graph, its final embedding (the
baseline CNN's last feature map is read the same way, one vector per cell),
the embedding norm, a 3-component PCA (shown as RGB) and a saliency
``|<h_i, dV/dh_i>|`` (how much the value head relies on that node), plus the
edges the network passes messages along, the move probabilities and the value.

Networks are identified by method: ``"baseline_cnn"``, ``"method1"``, ``"method2"``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from common import repo_bridge
from common.graph_layers import COLUMNS, PAD, ROWS, neighbour_table

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("baseline_cnn", "method1", "method2")
TITLES = {
    "baseline_cnn": "Baseline · ResNet (CNN)",
    "method1": "Phương pháp 1 · đồ thị động",
    "method2": "Phương pháp 2 · lưới cố định",
}
# Checkpoints tried in order: the full 90k-game run first, then the 3-iteration trial.
DEFAULT_CHECKPOINTS = {
    kind: [ROOT / "runs" / f"{kind}_full" / "models" / "latest.pt",
           ROOT / "runs" / f"trials_{kind}" / "models" / "latest.pt"]
    for kind in METHODS
}


def method_of(architecture: str) -> str | None:
    """Return the method (``baseline_cnn``/``method1``/``method2``) of a checkpoint's architecture tag, or ``None``.

    Side effects: none.
    """

    from method1 import model as method1
    from method2 import model as method2

    if architecture.startswith("residual"):
        return "baseline_cnn"
    if architecture in (method1.ARCHITECTURE, *method1.LEGACY_ARCHITECTURES):
        return "method1"
    if architecture in (method2.ARCHITECTURE, *method2.LEGACY_ARCHITECTURES):
        return "method2"
    return None


def untrained_model(kind: str) -> torch.nn.Module:
    """Return a randomly initialised network of ``kind`` at the sizes used in the trials.

    Side effects: uses the torch RNG.
    """

    from method1.model import Method1Net
    from method2.model import Method2Net
    from train_alphazero.build_network import residual_model_config

    if kind == "baseline_cnn":
        return repo_bridge.build_any(residual_model_config(256, 6))
    return Method1Net(width=160, blocks=6) if kind == "method1" else Method2Net(width=256, blocks=6)


@dataclass
class LoadedNetwork:
    """A network ready for the demo, with where it came from."""

    kind: str
    model: torch.nn.Module
    path: Path | None          # None = untrained (no checkpoint found)
    games_total: int | None
    parameters: int

    @property
    def label(self) -> str:
        """Title shown above the network's graph: method, size and training games.  Side effects: none."""

        config = self.model.model_config
        size = f"{config['channels']}×{config['residual_blocks']}"
        if self.path is None:
            trained = "CHƯA TRAIN (không có checkpoint)"
        else:
            trained = f"đã train {self.games_total:,} ván" if self.games_total is not None else "đã train ? ván"
        return f"{TITLES[self.kind]} {size} · {trained}"


@dataclass
class Representation:
    """Everything the demo draws for one network on one position."""

    kind: str
    exists: np.ndarray        # (42,) bool: node is part of the graph
    nodes: np.ndarray         # (42, F) final embeddings (zero where absent)
    norm: np.ndarray          # (42,)
    saliency: np.ndarray      # (42,)
    pca_rgb: np.ndarray       # (42, 3) in [0, 1]
    policy: np.ndarray        # (7,) probabilities over legal columns
    value: float              # side to move, in [-1, 1]
    edges: list[tuple[int, int, int, float]] = field(default_factory=list)  # (i, j, distance, empty cells between)

    @property
    def edge_counts(self) -> dict[int, int]:
        """Number of undirected edges per distance (1..3).  Side effects: none."""

        counts: dict[int, int] = {}
        for _, _, distance, _ in self.edges:
            counts[distance] = counts.get(distance, 0) + 1
        return counts


def load_network(kind: str, path: str | Path | None = None, *, allow_untrained: bool = True) -> LoadedNetwork:
    """Load the network of method ``kind`` from ``path`` or the first existing default checkpoint.

    With no checkpoint at all, returns an untrained network when ``allow_untrained`` (so the
    demo still opens on a fresh clone), else raises ``FileNotFoundError``.  Raises
    ``ValueError`` if the checkpoint holds another method.
    Side effects: reads the checkpoint file; registers our builders in the repo (see ``repo_bridge``).
    """

    from train_alphazero.load_checkpoint import load_policy_value_checkpoint

    repo_bridge.register_all()
    candidates = [Path(path)] if path else DEFAULT_CHECKPOINTS[kind]
    for candidate in candidates:
        if candidate.exists():
            model, checkpoint = load_policy_value_checkpoint(candidate, device="cpu")
            found = method_of(model.model_config["architecture"])
            if found != kind:
                raise ValueError(f"{candidate} holds a {found} network, expected {kind}")
            games = (checkpoint.get("metadata") or {}).get("games_total")
            return LoadedNetwork(kind, model.eval(), candidate, games, sum(p.numel() for p in model.parameters()))
    if path or not allow_untrained:
        raise FileNotFoundError(f"no checkpoint for {kind}; tried {', '.join(map(str, candidates))}")
    model = untrained_model(kind).eval()
    return LoadedNetwork(kind, model, None, 0, sum(p.numel() for p in model.parameters()))


def _pca_rgb(nodes: np.ndarray, exists: np.ndarray) -> np.ndarray:
    """Project the existing nodes' embeddings on their first 3 principal components, scaled to RGB in [0, 1].

    Absent nodes get light grey.  Side effects: none.
    """

    rgb = np.full((len(nodes), 3), 0.85)
    present = nodes[exists]
    if len(present) < 2:
        return rgb
    centred = present - present.mean(0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    components = vt[:3]
    components *= np.sign(components.sum(1, keepdims=True) + 1e-12)  # stable sign between moves
    projected = centred @ components.T
    if projected.shape[1] < 3:
        projected = np.pad(projected, ((0, 0), (0, 3 - projected.shape[1])))
    low, high = projected.min(0), projected.max(0)
    rgb[exists] = (projected - low) / np.maximum(high - low, 1e-9)
    return rgb


def _edges(exists: np.ndarray, edge: torch.Tensor | None, max_distance: int) -> list:
    """List the undirected edges ``(i, j, distance, empty cells between)`` between existing nodes.

    Side effects: none.
    """

    nbr, _ = neighbour_table(max_distance)
    edges = []
    for i in np.flatnonzero(exists):
        for r, j in enumerate(nbr[i].tolist()):
            if j == PAD or j <= i or not exists[j]:
                continue
            distance = r % max_distance + 1
            gaps = float(edge[0, i, r] * 2) if edge is not None else 0.0
            edges.append((int(i), int(j), distance, gaps))
    return edges


def analyse(network: LoadedNetwork, canonical: np.ndarray, legal: np.ndarray) -> Representation:
    """Run ``network`` on one position and collect what the demo shows.

    ``canonical`` is ``(7, 6)`` with +1 for the side to move; ``legal`` a ``(7,)`` bool mask.
    Side effects: runs one backward pass for the saliency, then clears the model's gradients.
    """

    model = network.model
    states = torch.tensor(canonical, dtype=torch.float32).reshape(1, 1, COLUMNS, ROWS)
    with torch.enable_grad():
        if network.kind == "baseline_cnn":
            features = model.residual_tower(model.stem(states))
            features.retain_grad()
            logits = model.policy_head(features)
            value = torch.tanh(model.value_head(F.relu(model.value_hidden(model.value_conv(features))))).squeeze(-1)
            value.sum().backward()
            nodes = features[0].flatten(1).T
            grad = features.grad[0].flatten(1).T
            exists = np.ones(COLUMNS * ROWS, dtype=bool)
            edge, max_distance = None, 1
        else:
            encoded = model.encode(states)
            hidden = encoded["nodes"]
            hidden.retain_grad()
            logits, value = model.heads(encoded)
            value.sum().backward()
            nodes, grad = hidden[0], hidden.grad[0]
            mask = encoded["node_mask"]
            exists = np.ones(COLUMNS * ROWS, dtype=bool) if mask is None else mask[0, :, 0].numpy() > 0
            edge, max_distance = encoded["edge"], model.max_distance
    model.zero_grad(set_to_none=True)

    nodes_np = nodes.detach().numpy().copy()
    nodes_np[~exists] = 0.0
    saliency = (nodes.detach() * grad).sum(-1).abs().numpy()
    saliency[~exists] = 0.0
    logits = logits.detach()[0].masked_fill(~torch.from_numpy(legal), float("-inf"))
    policy = torch.softmax(logits, dim=0).numpy() if legal.any() else np.zeros(COLUMNS)
    edges = [] if network.kind == "baseline_cnn" else _edges(exists, edge, max_distance)
    return Representation(
        kind=network.kind,
        exists=exists,
        nodes=nodes_np,
        norm=np.linalg.norm(nodes_np, axis=1),
        saliency=saliency,
        pca_rgb=_pca_rgb(nodes_np, exists),
        policy=policy,
        value=float(value.detach()[0]),
        edges=edges,
    )
