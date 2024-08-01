import torch
import torch_ttnn
import pytest
import ttnn

from tests.utils import assert_with_pcc


class AddModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x + y


class MatmulModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.matmul(x, y)


class BatchMatmulModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.bmm(x, y)


class AddMatmulModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mm = MatmulModule()

    def forward(self, x, y):
        m = torch.add(x, y)
        return self.mm(m, m)


@pytest.mark.parametrize(
    "input_shapes",
    [
        [(4, 4), (4, 4)],
    ],
)
def test_add(device, input_shapes):
    m = AddModule()
    inputs = [torch.randint(0, 3, shape).type(torch.bfloat16) for shape in input_shapes]
    result_before = m.forward(*inputs)
    option = torch_ttnn.TorchTtnnOption(device=device)
    # The compilation is lazy, so we need to run forward once to trigger the compilation
    m = torch.compile(m, backend=torch_ttnn.backend, options=option)
    result_after = m.forward(*inputs)
    assert 1 == len(option._out_fx_graphs)
    option._out_fx_graphs[0].print_tabular()

    # Check the graph has be rewritten and contain ttnn ops
    nodes = list(option._out_fx_graphs[0].nodes)
    assert [node.target for node in nodes].count(ttnn.add) == 1
    # Check inference result
    assert torch.allclose(result_before, result_after)


@pytest.mark.parametrize(
    "input_shapes",
    [
        [(32, 32), (32, 32)],
    ],
)
def test_matmul(device, input_shapes):
    m = MatmulModule()
    inputs = [torch.randint(0, 3, shape).type(torch.bfloat16) for shape in input_shapes]
    result_before = m.forward(*inputs)
    option = torch_ttnn.TorchTtnnOption(device=device)
    option.gen_graphviz = True
    # The compilation is lazy, so we need to run forward once to trigger the compilation
    m = torch.compile(m, backend=torch_ttnn.backend, options=option)
    result_after = m.forward(*inputs)
    assert 1 == len(option._out_fx_graphs)
    option._out_fx_graphs[0].print_tabular()

    # Check the graph has be rewritten and contain ttnn ops
    nodes = list(option._out_fx_graphs[0].nodes)
    assert [node.target for node in nodes].count(ttnn.matmul) == 1
    # Check inference result
    assert torch.allclose(result_before, result_after)


@pytest.mark.parametrize(
    "input_shapes",
    [
        [(10, 64, 32), (10, 32, 128)],
    ],
)
def test_batchmatmul(device, input_shapes):
    m = BatchMatmulModule()
    inputs = [torch.rand(shape).type(torch.bfloat16) for shape in input_shapes]
    result_before = m.forward(*inputs)
    option = torch_ttnn.TorchTtnnOption(device=device)
    option.gen_graphviz = True
    # The compilation is lazy, so we need to run forward once to trigger the compilation
    m = torch.compile(m, backend=torch_ttnn.backend, options=option)
    result_after = m.forward(*inputs)
    assert 1 == len(option._out_fx_graphs)
    option._out_fx_graphs[0].print_tabular()

    # Check the graph has be rewritten and contain ttnn ops
    nodes = list(option._out_fx_graphs[0].nodes)
    assert [node.target for node in nodes].count(ttnn.matmul) == 1
    # Check inference result
    assert_with_pcc(result_before, result_after, 0.999)


@pytest.mark.parametrize(
    "input_shapes",
    [
        [(4, 4), (4, 4)],
    ],
)
def test_add_and_matmul(device, input_shapes):
    m = AddMatmulModule()
    inputs = [torch.randint(0, 3, shape).type(torch.bfloat16) for shape in input_shapes]
    result_before = m.forward(*inputs)
    option = torch_ttnn.TorchTtnnOption(device=device)
    option.gen_graphviz = True
    # The compilation is lazy, so we need to run forward once to trigger the compilation
    m = torch.compile(m, backend=torch_ttnn.backend, options=option)
    result_after = m.forward(*inputs)
    assert 1 == len(option._out_fx_graphs)
    option._out_fx_graphs[0].print_tabular()

    # Check the graph has be rewritten and contain ttnn ops
    nodes = list(option._out_fx_graphs[0].nodes)
    target = [node.target for node in nodes]
    assert target.count(ttnn.add) == 1
    assert target.count(ttnn.matmul) == 1
    assert target.index(ttnn.add) < target.index(ttnn.matmul)
    assert nodes[target.index(ttnn.matmul)].args[0].target == ttnn.add
    assert nodes[target.index(ttnn.matmul)].args[1].target == ttnn.add
    # Check inference result
    assert torch.allclose(result_before, result_after, 0.999)
