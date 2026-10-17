"""GNN building blocks shared by method 1 and method 2.

Both networks keep the AlphaZero repo's interface: canonical boards ``(N, 1, 7, 6)``
(+1 side to move, -1 opponent, ``state[column, row]``, row 0 at the bottom) in,
raw policy logits ``(N, 7)`` and a tanh value ``(N,)`` out.

Node ``i`` is the cell ``column * ROWS + row`` (the order of ``states.reshape(N, 42)``).
A *mask* says which nodes exist in a position; method 2 has no mask (every cell exists),
method 1 keeps only pieces and frontier cells.

Message passing (R-GCN, sum aggregation, one weight per directed relation)::

    out_i = W0 h_i + sum_r W_r h_nbr(i,r) + Wg mean_{existing j} h_j + sum_r e_ir U_r

A relation is a direction (8) and a distance (1..K).  On a board every relation is a
fixed offset, so the sum over relations at distance ``k`` is exactly a 3x3 convolution
with dilation ``k`` whose 8 off-centre taps are the ``W_r``.  ``RelationalLayer`` runs it
that way (cuDNN, no gathered copies); ``reference_relational`` is the literal gather
form, and the tests check both give the same numbers.  Nodes that do not exist are zero,
so they send nothing; their outputs are masked, so they receive nothing.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from common.connect4_rules import COLUMNS, ROWS

CELLS = COLUMNS * ROWS
PAD = CELLS  # index of the all-zero padding node in gathered tensors

# Directed relations: N, S, E, W, NE, SW, SE, NW  (dx = column step, dy = row step).
DIRECTIONS = ((0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (-1, -1), (1, -1), (-1, 1))
DIRECTION_NAMES = ("N", "S", "E", "W", "NE", "SW", "SE", "NW")
NODE_FEATURES = 7  # own, opp, empty, playable, row/5, row parity, distance to centre


def cell(column: int, row: int) -> int:
    """Return the node index of a cell: ``column * ROWS + row``.

    Side effects: none.
    """

    return column * ROWS + row


def relation_index(direction: int, distance: int, max_distance: int) -> int:
    """Return the relation index of ``(direction, distance)``: ``direction * max_distance + distance - 1``.

    Side effects: none.
    """

    return direction * max_distance + distance - 1


def neighbour_table(max_distance: int = 1) -> tuple[Tensor, Tensor]:
    """Build ``(nbr, between)`` for relations ``direction * max_distance + (distance - 1)``.

    ``nbr[i, r]`` is the node reached from ``i`` along relation ``r`` (``PAD`` when off
    the board).  ``between[i, r, :]`` lists the cells strictly in between (``PAD``-filled);
    only method 1's line-of-sight edges use it, to count empty cells.
    Side effects: none (returns new tensors).
    """

    relations = len(DIRECTIONS) * max_distance
    nbr = torch.full((CELLS, relations), PAD, dtype=torch.long)
    between = torch.full((CELLS, relations, max(max_distance - 1, 1)), PAD, dtype=torch.long)
    for column in range(COLUMNS):
        for row in range(ROWS):
            i = cell(column, row)
            for d, (dx, dy) in enumerate(DIRECTIONS):
                for k in range(1, max_distance + 1):
                    x, y = column + k * dx, row + k * dy
                    if 0 <= x < COLUMNS and 0 <= y < ROWS:
                        r = relation_index(d, k, max_distance)
                        nbr[i, r] = cell(x, y)
                        for step in range(1, k):
                            between[i, r, step - 1] = cell(column + step * dx, row + step * dy)
    return nbr, between


def _positional_features() -> Tensor:
    """Return the ``(42, 3)`` geometry features per cell: row/5, row parity, distance to the centre column.

    Side effects: none.  They depend only on geometry, so they would generalise to other board sizes.
    """

    rows = torch.arange(ROWS, dtype=torch.float32).repeat(COLUMNS)
    columns = torch.arange(COLUMNS, dtype=torch.float32).repeat_interleave(ROWS)
    return torch.stack(
        [rows / (ROWS - 1), rows.remainder(2), (columns - (COLUMNS - 1) / 2).abs() / ((COLUMNS - 1) / 2)],
        dim=-1,
    )


class BoardFeatures(nn.Module):
    """Turns ``(N, 1, 7, 6)`` boards into node features and the cell masks."""

    def __init__(self) -> None:
        """Register the constant positional features as a non-persistent buffer (not saved in checkpoints).

        Side effects: none outside this module.
        """

        super().__init__()
        self.register_buffer("positional", _positional_features(), persistent=False)

    def forward(self, states: Tensor) -> dict[str, Tensor]:
        """Return ``features (N, 42, 7)``, ``occupied``/``playable`` ``(N, 42)`` bool and ``heights (N, 7)``.

        Raises ``ValueError`` on a wrong input shape.  Side effects: none.
        """

        if states.ndim != 4 or tuple(states.shape[1:]) != (1, COLUMNS, ROWS):
            raise ValueError(f"expected shape (N, 1, 7, 6); got {tuple(states.shape)}")
        board = states.reshape(states.shape[0], COLUMNS, ROWS).float()
        occupied = board != 0
        heights = occupied.sum(-1)  # (N, 7); gravity keeps pieces contiguous
        rows = torch.arange(ROWS, device=states.device)
        playable = (rows[None, None, :] == heights[:, :, None]).reshape(-1, CELLS)
        board = board.reshape(-1, CELLS)
        own = (board > 0).float()
        opp = (board < 0).float()
        features = torch.cat(
            [
                torch.stack([own, opp, 1.0 - own - opp, playable.float()], dim=-1),
                self.positional.expand(board.shape[0], -1, -1),
            ],
            dim=-1,
        )
        return {
            "features": features,                     # (N, 42, 7)
            "occupied": occupied.reshape(-1, CELLS),  # (N, 42) bool
            "playable": playable,                     # (N, 42) bool
            "heights": heights,                       # (N, 7)
        }


def to_grid(nodes: Tensor) -> Tensor:
    """Reshape node tensors ``(N, 42, C)`` to grid tensors ``(N, C, 7, 6)``.

    Side effects: none.
    """

    return nodes.transpose(1, 2).reshape(nodes.shape[0], nodes.shape[2], COLUMNS, ROWS)


def to_nodes(grid: Tensor) -> Tensor:
    """Reshape grid tensors ``(N, C, 7, 6)`` to node tensors ``(N, 42, C)``.

    Side effects: none.
    """

    return grid.flatten(2).transpose(1, 2)


def masked_mean_grid(h: Tensor, mask: Tensor | None) -> Tensor:
    """Mean over existing nodes: ``h (N, C, 7, 6)``, ``mask (N, 1, 7, 6)`` or ``None`` (all) -> ``(N, C)``.

    Side effects: none.
    """

    if mask is None:
        return h.mean((2, 3))
    return (h * mask).sum((2, 3)) / mask.sum((2, 3)).clamp_min(1.0)


def apply_mask(x: Tensor, mask: Tensor | None) -> Tensor:
    """Zero the absent nodes of ``x`` (no-op when ``mask`` is ``None``).

    Side effects: none (returns a new tensor).
    """

    return x if mask is None else x * mask


class RelationalLayer(nn.Module):
    """One relational message-passing layer, computed as dilated 3x3 convolutions."""

    def __init__(self, in_width: int, out_width: int, max_distance: int, edge_features: bool) -> None:
        """Create the self, per-relation, global and (optional) edge-feature weights.

        Weights use ``nn.Linear``'s default uniform init over the concatenated inputs.
        Side effects: none outside this module.
        """

        super().__init__()
        self.max_distance = max_distance
        relations = len(DIRECTIONS) * max_distance
        bound = 1.0 / math.sqrt(in_width * (relations + 2))

        def uniform(*shape: int) -> nn.Parameter:
            """Return a new parameter of ``shape`` drawn from U(-bound, bound).  Side effects: uses the torch RNG."""

            return nn.Parameter(torch.empty(*shape).uniform_(-bound, bound))

        self.self_weight = uniform(out_width, in_width)
        self.relation_weight = uniform(relations, out_width, in_width)
        self.global_weight = uniform(out_width, in_width)
        self.edge_weight = uniform(relations, out_width) if edge_features else None
        taps = torch.tensor([(dx + 1) * 3 + (dy + 1) for dx, dy in DIRECTIONS])
        self.register_buffer("taps", taps, persistent=False)
        self._kernel_cache: tuple[tuple, list[Tensor]] | None = None

    def kernels(self) -> list[Tensor]:
        """Return the 3x3 kernel of every distance; cached for inference until a weight changes.

        The cache key includes the weights' version counters, so optimizer steps invalidate it.
        Side effects: may update ``_kernel_cache`` (only in eval mode without autograd).
        """

        if self.training or torch.is_grad_enabled():
            return [self.kernel(d) for d in range(1, self.max_distance + 1)]
        key = (self.self_weight._version, self.relation_weight._version, self.self_weight.data_ptr(),
               torch.is_inference_mode_enabled(),
               self.self_weight.device, torch.get_autocast_dtype("cuda") if torch.is_autocast_enabled("cuda") else None)
        if self._kernel_cache is None or self._kernel_cache[0] != key:
            kernels = [self.kernel(d) for d in range(1, self.max_distance + 1)]
            if key[-1] is not None:
                kernels = [k.to(key[-1]) for k in kernels]
            self._kernel_cache = (key, kernels)
        return self._kernel_cache[1]

    def kernel(self, distance: int) -> Tensor:
        """Assemble the ``(out, in, 3, 3)`` kernel (dilation ``distance``) from the relations of that distance.

        The centre tap holds the self weight at distance 1 only.  Side effects: none.
        """

        out_width, in_width = self.self_weight.shape
        weights = self.relation_weight[distance - 1 :: self.max_distance]  # (8, out, in), direction order
        taps = self.self_weight.new_zeros(9, out_width, in_width)
        taps = taps.index_copy(0, self.taps, weights)
        if distance == 1:
            taps = taps.index_copy(0, self.taps.new_tensor([4]), self.self_weight[None])
        return taps.permute(1, 2, 0).reshape(out_width, in_width, 3, 3)

    def forward(self, h: Tensor, mask: Tensor | None, edge: Tensor | None = None) -> Tensor:
        """Apply the layer: ``h (N, in, 7, 6)`` (zero where absent), ``mask (N, 1, 7, 6)``/``None``, ``edge (N, R, 7, 6)``.

        Returns ``(N, out, 7, 6)``; the caller masks the output.  Side effects: see ``kernels`` (cache).
        """

        out = None
        for distance, kernel in enumerate(self.kernels(), start=1):
            term = F.conv2d(h, kernel, padding=distance, dilation=distance)
            out = term if out is None else out + term
        out = out + F.linear(masked_mean_grid(h, mask), self.global_weight)[:, :, None, None]
        if self.edge_weight is not None:
            out = out + torch.einsum("nrxy,ro->noxy", edge.to(out.dtype), self.edge_weight.to(out.dtype))
        return out


def reference_relational(layer: RelationalLayer, h: Tensor, mask: Tensor, edge: Tensor | None, nbr: Tensor) -> Tensor:
    """Compute ``RelationalLayer.forward`` in the literal gather form on nodes ``(N, 42, in)`` (test oracle).

    Side effects: none.
    """

    batch = h.shape[0]
    padded = torch.cat([h, h.new_zeros(batch, 1, h.shape[2])], dim=1)
    gathered = padded[:, nbr]  # (N, 42, R, in)
    out = h @ layer.self_weight.T + torch.einsum("nkri,roi->nko", gathered, layer.relation_weight)
    pooled = (h * mask).sum(1) / mask.sum(1).clamp_min(1.0)
    out = out + (pooled @ layer.global_weight.T)[:, None, :]
    if layer.edge_weight is not None:
        out = out + torch.einsum("nkr,ro->nko", edge, layer.edge_weight)
    return out


def channel_norm(norm: nn.LayerNorm, x: Tensor) -> Tensor:
    """Apply ``norm`` (LayerNorm) over the channels of ``x (N, C, 7, 6)``, independently for every node.

    Side effects: none.
    """

    return norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ResidualGraphBlock(nn.Module):
    """Two relational layers with LayerNorm and a residual connection."""

    def __init__(self, width: int, max_distance: int, edge_features: bool) -> None:
        """Create the two layers and their norms.  Side effects: none outside this module."""

        super().__init__()
        self.layer1 = RelationalLayer(width, width, max_distance, edge_features)
        self.norm1 = nn.LayerNorm(width)
        self.layer2 = RelationalLayer(width, width, max_distance, edge_features)
        self.norm2 = nn.LayerNorm(width)

    def forward(self, h: Tensor, mask: Tensor | None, edge: Tensor | None) -> Tensor:
        """Return ``relu(h + norm(layer2(relu(norm(layer1(h))))))``, absent nodes kept at zero.

        Side effects: none.
        """

        x = apply_mask(F.relu(channel_norm(self.norm1, self.layer1(h, mask, edge))), mask)
        x = channel_norm(self.norm2, self.layer2(x, mask, edge))
        return apply_mask(F.relu(h + x), mask)


class GraphPolicyValueNet(nn.Module):
    """Relational GNN with a global readout, a per-column policy and a pooled value.

    Subclasses (``method1.model.Method1Net``, ``method2.model.Method2Net``) decide the
    graph: which nodes exist (``node_mask``), the relation distances and the edge features.
    """

    def __init__(self, *, width: int, blocks: int, max_distance: int, edge_features: bool) -> None:
        """Create the stem, ``blocks`` residual blocks, the policy head and the value head.

        Raises ``ValueError`` if ``width < 16`` or ``blocks < 1``.
        Side effects: registers the ``nbr`` neighbour table as a non-persistent buffer.
        """

        super().__init__()
        if width < 16 or blocks < 1:
            raise ValueError("width must be >= 16 and blocks positive")
        nbr, _ = neighbour_table(max_distance)
        self.register_buffer("nbr", nbr, persistent=False)
        self.max_distance = max_distance
        self.board_features = BoardFeatures()
        self.stem = RelationalLayer(NODE_FEATURES, width, max_distance, edge_features)
        self.stem_norm = nn.LayerNorm(width)
        self.tower = nn.ModuleList(ResidualGraphBlock(width, max_distance, edge_features) for _ in range(blocks))
        self.policy_head = nn.Sequential(nn.Linear(2 * width, 64), nn.ReLU(inplace=True), nn.Linear(64, 1))
        self.value_hidden = nn.Linear(2 * width, 256)
        self.value_head = nn.Linear(256, 1)

    # Hooks the two methods override ---------------------------------------------
    def node_mask(self, board: dict[str, Tensor]) -> Tensor | None:
        """Return ``(N, 42)`` bool: which nodes exist; ``None`` means every node (static grid).

        Must be overridden.  Side effects: none.
        """

        raise NotImplementedError

    def edge_features(self, board: dict[str, Tensor], mask: Tensor | None) -> Tensor | None:
        """Return ``(N, 42, R)`` per-edge scalars, or ``None`` for no edge features; ``mask`` is ``(N, 42, 1)``.

        Default: no edge features.  Side effects: none.
        """

        return None

    # -------------------------------------------------------------------------------
    def forward(self, states: Tensor) -> tuple[Tensor, Tensor]:
        """Return ``(policy_logits (N, 7), value (N,))`` for canonical boards ``(N, 1, 7, 6)``.

        This is the interface the AlphaZero repo calls.  Side effects: none.
        """

        return self.heads(self.encode(states))

    def encode(self, states: Tensor) -> dict[str, Tensor | None]:
        """Run the GNN and return the graph it used and its final node embeddings.

        Keys: ``board`` (see ``BoardFeatures``), ``node_mask`` ``(N, 42, 1)`` or ``None``,
        ``edge`` ``(N, 42, R)`` or ``None``, ``nodes`` ``(N, 42, width)``.  The demo reads it
        directly.  Side effects: none.
        """

        board = self.board_features(states)
        exists = self.node_mask(board)
        node_mask = exists.float()[..., None] if exists is not None else None  # (N, 42, 1)
        edge_nodes = self.edge_features(board, node_mask)
        mask = to_grid(node_mask) if node_mask is not None else None
        edge = to_grid(edge_nodes) if edge_nodes is not None else None
        x = apply_mask(to_grid(board["features"]), mask)
        if states.is_cuda:
            x = x.contiguous(memory_format=torch.channels_last)
        h = apply_mask(F.relu(channel_norm(self.stem_norm, self.stem(x, mask, edge))), mask)
        for block in self.tower:
            h = block(h, mask, edge)
        return {
            "board": board,
            "node_mask": node_mask,
            "edge": edge_nodes,
            "nodes": to_nodes(h),
        }

    def heads(self, encoded: dict) -> tuple[Tensor, Tensor]:
        """Turn ``encode``'s output into ``(policy_logits (N, 7), value (N,))``.

        Policy: per column, the node where its next piece lands (top cell for a full column;
        full columns are masked outside the model, as for the CNN) plus the mean embedding.
        Value: tanh of an MLP over the mean and max of the existing nodes.  Side effects: none.
        """

        board, node_mask, nodes = encoded["board"], encoded["node_mask"], encoded["nodes"]
        if node_mask is None:
            pooled_mean = nodes.mean(1)
        else:
            pooled_mean = (nodes * node_mask).sum(1) / node_mask.sum(1).clamp_min(1.0)
        pooled_max = nodes.amax(dim=1)  # h >= 0 after ReLU and absent nodes are 0
        heights = board["heights"].clamp_max(ROWS - 1)
        target = torch.arange(COLUMNS, device=nodes.device) * ROWS + heights  # (N, 7)
        column_nodes = torch.gather(nodes, 1, target[..., None].expand(-1, -1, nodes.shape[-1]))
        policy_input = torch.cat([column_nodes, pooled_mean[:, None, :].expand(-1, COLUMNS, -1)], dim=-1)
        policy_logits = self.policy_head(policy_input).squeeze(-1)
        value_features = F.relu(self.value_hidden(torch.cat([pooled_mean, pooled_max], dim=-1)))
        value = torch.tanh(self.value_head(value_features)).squeeze(-1)
        return policy_logits, value


def base_model_config(architecture: str, width: int, blocks: int) -> dict:
    """Return the ``model_config`` dict the repo stores in checkpoints (shared fields of both methods).

    Side effects: none.
    """

    return {
        "board_shape": [COLUMNS, ROWS],
        "input_channels": 1,
        "policy_actions": COLUMNS,
        "value_output": "scalar_tanh",
        "architecture": architecture,
        "channels": int(width),
        "residual_blocks": int(blocks),
    }
