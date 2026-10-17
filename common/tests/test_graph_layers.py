"""Tests of ``common.graph_layers``: the convolution form equals literal message passing; the kernel cache."""

import pytest
import torch

from common.graph_layers import RelationalLayer, neighbour_table, reference_relational, to_grid, to_nodes


@pytest.mark.parametrize("max_distance,edge", [(1, False), (3, False), (3, True)])
def test_convolution_equals_gather_message_passing(max_distance, edge):
    """``RelationalLayer`` (dilated convs) gives the same numbers as the gather form, with masks and edges."""

    torch.manual_seed(0)
    layer = RelationalLayer(5, 4, max_distance, edge).double()
    nbr, _ = neighbour_table(max_distance)
    mask = (torch.rand(3, 42, 1) > 0.4).double()
    h = torch.randn(3, 42, 5, dtype=torch.float64) * mask
    edges = torch.rand(3, 42, nbr.shape[1], dtype=torch.float64) if edge else None
    fast = to_nodes(layer(to_grid(h), to_grid(mask), to_grid(edges) if edge else None))
    slow = reference_relational(layer, h, mask, edges, nbr)
    assert torch.allclose(fast, slow, atol=1e-10)


def test_inference_kernel_cache_follows_optimizer_updates():
    """The cached inference kernels are rebuilt after every optimizer step."""

    from method1.model import Method1Net

    torch.manual_seed(1)
    model = Method1Net(width=16, blocks=1)
    states = torch.zeros(2, 1, 7, 6)
    states[0, 0, 3, 0] = 1
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    for _ in range(2):
        model.eval()
        with torch.inference_mode():
            cached = model(states)[0]
        with torch.no_grad():
            model.train()  # training path never uses the cache
            fresh = model(states)[0]
        assert torch.allclose(cached, fresh, atol=1e-6)
        model.train()
        logits, value = model(states)
        optimizer.zero_grad()
        (logits.square().sum() + value.sum()).backward()
        optimizer.step()
