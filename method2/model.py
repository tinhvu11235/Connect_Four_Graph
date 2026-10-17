"""Method 2 network: every cell is a node, empty cells included.

The structure (``edge_index``) never changes; only node features do.  Each node is
linked to its (up to) 8 neighbours, one weight per direction.
"""

from __future__ import annotations

from torch import Tensor

from common.graph_layers import GraphPolicyValueNet, base_model_config

ARCHITECTURE = "method2_v1"
LEGACY_ARCHITECTURES = ("gnn_grid_v1",)  # tag of checkpoints saved before the rename


class Method2Net(GraphPolicyValueNet):
    """Method 2: the fixed 42-node grid graph, distance-1 relations, no edge features."""

    def __init__(self, *, width: int = 256, blocks: int = 6) -> None:
        """Create the network.  Side effects: none outside this module."""

        super().__init__(width=width, blocks=blocks, max_distance=1, edge_features=False)
        self.model_config = model_config(width, blocks)

    def node_mask(self, board: dict[str, Tensor]) -> Tensor | None:
        """Return ``None``: every cell exists, empty or not.  Side effects: none."""

        return None


def model_config(width: int, blocks: int) -> dict:
    """Return the ``model_config`` dict stored in checkpoints for a method 2 network.

    Side effects: none.
    """

    return base_model_config(ARCHITECTURE, width, blocks)


def build(config: dict) -> Method2Net:
    """Build a fresh ``Method2Net`` from a ``model_config`` (current or legacy tag).

    Raises ``ValueError`` for another architecture.  Side effects: none (random init).
    """

    if config.get("architecture") not in (ARCHITECTURE, *LEGACY_ARCHITECTURES):
        raise ValueError(f"not a {ARCHITECTURE} config: {config!r}")
    return Method2Net(width=config["channels"], blocks=config["residual_blocks"])
