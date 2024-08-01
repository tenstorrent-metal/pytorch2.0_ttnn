import torch
import torch_ttnn
import pytest
import ttnn

from tests.utils import assert_with_pcc


class AtenTModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.t(x)


def _t(device, input_shapes):
    m = AtenTModule()
    inputs = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
    result_before = m.forward(*inputs)
    option = torch_ttnn.TorchTtnnOption(device=device)
    option.gen_graphviz = True
    # The compilation is lazy, so we need to run forward once to trigger the compilation
    m = torch.compile(m, backend=torch_ttnn.backend, options=option)
    result_after = m.forward(*inputs)
    option._out_fx_graphs[0].print_tabular()

    # Check inference result
    assert_with_pcc(result_before, result_after)

    # Return nodes
    return list(option._out_fx_graphs[0].nodes)


@pytest.mark.parametrize(
    "input_shapes",
    [
        [(1, 32)],
    ],
)
def test_aten_t(device, input_shapes):
    # Check the graph has be rewritten and contain ttnn ops
    nodes = _t(device, input_shapes)
    assert [node.target for node in nodes].count(ttnn.permute) == 1


@pytest.mark.parametrize(
    "input_shapes",
    [
        [5],
    ],
)
def test_aten_t_0d(device, input_shapes):
    # Check the graph has be rewritten and contain ttnn ops
    nodes = _t(device, input_shapes)

    assert nodes[1].target == torch.ops.aten.t.default
    assert nodes[1].args[0].target == "arg0_1"
    assert nodes[1].args[0].op == "placeholder"
    assert nodes[2].target == "output"
    assert nodes[2].op == "output"


@pytest.mark.parametrize(
    "input_shapes",
    [
        [(5)],
    ],
)
def test_aten_t_1d(device, input_shapes):
    # Check the graph has be rewritten and contain ttnn ops
    nodes = _t(device, input_shapes)

    assert nodes[1].target == torch.ops.aten.t.default
    assert nodes[1].args[0].target == "arg0_1"
    assert nodes[1].args[0].op == "placeholder"
    assert nodes[2].target == "output"
    assert nodes[2].op == "output"
