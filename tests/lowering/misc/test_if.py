import torch
import torch_ttnn
import ttnn
import pytest

from torch.fx.passes.dialect.common.cse_pass import CSEPass


class IfModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if torch.sum(x) > 0:
            return x + x
        else:
            return torch.matmul(x, x)


@pytest.mark.parametrize("input_shapes", [(4, 4)])
def test_if(compiler_options, input_shapes):
    m = IfModule()
    inputs_then = [torch.tensor(range(1, 17)).reshape(input_shapes).to(torch.bfloat16)]
    inputs_else = [-inputs_then[0]]

    result_before_then = m.forward(*inputs_then)
    result_before_else = m.forward(*inputs_else)

    # The compilation is lazy, so we need to run forward once to trigger the compilation
    m = torch.compile(m, backend=torch_ttnn.backend, options=compiler_options)
    result_after_then = m.forward(*inputs_then)
    result_after_else = m.forward(*inputs_else)

    # After the forward & compilation, there should be a total of 3 graphs
    assert len(compiler_options._out_fx_graphs) == 3
    compiler_options._out_fx_graphs[0].print_tabular()

    # Check the graph has been rewritten and contains ttnn ops
    nodes_0 = list(compiler_options._out_fx_graphs[0].nodes)
    assert len(nodes_0) == 4
    assert nodes_0[1].target == torch.ops.aten.sum.default
    assert nodes_0[2].target == torch.ops.aten.gt.Scalar

    nodes_1 = list(compiler_options._out_fx_graphs[1].nodes)
    assert len(nodes_1) == 9
    assert nodes_1[1].target == ttnn.from_torch
    assert nodes_1[2].target == ttnn.to_layout
    assert nodes_1[3].target == ttnn.to_device
    assert nodes_1[4].target == ttnn.add
    assert nodes_1[5].target == ttnn.from_device
    assert nodes_1[6].target == ttnn.to_layout
    assert nodes_1[7].target == ttnn.to_torch

    nodes_2 = list(compiler_options._out_fx_graphs[2].nodes)
    assert len(nodes_2) == 9
    assert nodes_2[1].target == ttnn.from_torch
    assert nodes_2[2].target == ttnn.to_layout
    assert nodes_2[3].target == ttnn.to_device
    assert nodes_2[4].target == ttnn.matmul
    assert nodes_2[5].target == ttnn.from_device
    assert nodes_2[6].target == ttnn.to_layout
    assert nodes_2[7].target == ttnn.to_torch

    # Check inference results
    assert torch.allclose(result_before_then, result_after_then)
    assert torch.allclose(result_before_else, result_after_else)
