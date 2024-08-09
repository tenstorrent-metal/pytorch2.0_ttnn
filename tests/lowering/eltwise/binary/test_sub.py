import torch
import torch_ttnn
import pytest
import ttnn

from tests.utils import assert_with_pcc


class SubModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x - y


class RSubModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.rsub(x, y)


class RSubScalarModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, scalar):
        return torch.rsub(x, scalar)


@pytest.mark.parametrize(
    "input_shapes",
    [[(4, 4), (4, 4)]],
)
def test_sub(device, input_shapes):
    m = SubModule()
    inputs = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
    result_before = m.forward(*inputs)
    option = torch_ttnn.TorchTtnnOption(device=device)
    option.gen_graphviz = True
    # The compilation is lazy, so we need to run forward once to trigger the compilation
    m = torch.compile(m, backend=torch_ttnn.backend, options=option)
    result_after = m.forward(*inputs)
    option._out_fx_graphs[0].print_tabular()

    # Check the graph has be rewritten and contain ttnn ops
    nodes = list(option._out_fx_graphs[0].nodes)
    assert [node.target for node in nodes].count(ttnn.sub) == 1
    # Check inference result
    assert torch.allclose(result_before, result_after)


@pytest.mark.parametrize(
    "input_shapes",
    [[(4, 4), (4, 4)]],
)
def test_rsub(device, input_shapes):
    m = RSubModule()
    inputs = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
    result_before = m.forward(*inputs)
    option = torch_ttnn.TorchTtnnOption(device=device)
    option.gen_graphviz = True
    # The compilation is lazy, so we need to run forward once to trigger the compilation
    m = torch.compile(m, backend=torch_ttnn.backend, options=option)
    result_after = m.forward(*inputs)
    option._out_fx_graphs[0].print_tabular()

    # Check the graph has be rewritten and contain ttnn ops
    nodes = list(option._out_fx_graphs[0].nodes)
    assert [node.target for node in nodes].count(ttnn.sub) == 1
    # Intermediate node meta check if preserved
    for node in nodes:
        if node.target == ttnn.full:
            assert node.meta["val"].size() == input_shapes[0]
    # Check inference result
    assert torch.allclose(result_before, result_after)


@pytest.mark.parametrize(
    "input_shapes",
    [[(4, 4)]],
)
def test_rsub_scalar(device, input_shapes):
    m = RSubScalarModule()
    inputs = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
    result_before = m.forward(*inputs, 5)
    option = torch_ttnn.TorchTtnnOption(device=device)
    option.gen_graphviz = True
    # The compilation is lazy, so we need to run forward once to trigger the compilation
    m = torch.compile(m, backend=torch_ttnn.backend, options=option)
    result_after = m.forward(*inputs, 5)
    option._out_fx_graphs[0].print_tabular()

    # Check the graph has be rewritten and contain ttnn ops
    nodes = list(option._out_fx_graphs[0].nodes)
    target = [node.target for node in nodes]
    assert target.count(ttnn.full) == 1
    assert target.count(ttnn.sub) == 1
    assert target.index(ttnn.full) < target.index(ttnn.sub)
    # Intermediate node meta check if preserved
    for node in nodes:
        if node.target == ttnn.full:
            assert node.meta["val"].size() == input_shapes[0]
    # Check inference result
    assert_with_pcc(result_before, result_after, 0.999)
