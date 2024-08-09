import torch
import torch_ttnn
import pytest
import ttnn

from tests.utils import assert_with_pcc


class CloneFromNodeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        a = input + input
        return torch.clone(a)


class CloneFromArgModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.clone(input)


@pytest.mark.parametrize(
    "input_shapes",
    [[(4, 4)]],
)
def test_clone_from_arg(device, input_shapes):
    m = CloneFromArgModule()
    inputs = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
    result_before = m.forward(*inputs)
    option = torch_ttnn.TorchTtnnOption(device=device)
    option.gen_graphviz = False
    # The compilation is lazy, so we need to run forward once to trigger the compilation
    m = torch.compile(m, backend=torch_ttnn.backend, options=option)
    result_after = m.forward(*inputs)
    option._out_fx_graphs[0].print_tabular()

    # Check the graph has be rewritten and contain ttnn ops
    nodes = list(option._out_fx_graphs[0].nodes)
    assert [node.target for node in nodes].count(ttnn.clone) == 1
    # Check inference result
    assert torch.allclose(result_before, result_after)


@pytest.mark.parametrize(
    "input_shapes",
    [[(4, 4)]],
)
def test_clone_from_node(device, input_shapes):
    m = CloneFromNodeModule()
    inputs = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
    result_before = m.forward(*inputs)
    option = torch_ttnn.TorchTtnnOption(device=device)
    option.gen_graphviz = False
    # The compilation is lazy, so we need to run forward once to trigger the compilation
    m = torch.compile(m, backend=torch_ttnn.backend, options=option)
    result_after = m.forward(*inputs)
    option._out_fx_graphs[0].print_tabular()

    # Check the graph has be rewritten and contain ttnn ops
    nodes = list(option._out_fx_graphs[0].nodes)
    target = [node.target for node in nodes]
    assert target.count(ttnn.clone) == 1
    clone_arg_0 = nodes[target.index(ttnn.clone)].args[0].target
    assert isinstance(clone_arg_0, ttnn.decorators.FastOperation) or isinstance(
        clone_arg_0, ttnn.decorators.Operation
    )
    # Check inference result
    assert torch.allclose(result_before, result_after)
