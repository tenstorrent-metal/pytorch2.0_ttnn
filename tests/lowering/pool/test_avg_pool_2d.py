import torch
import torch_ttnn
import pytest
import ttnn

from tests.utils import assert_with_pcc


class AdaptiveAvgPool2dModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch._adaptive_avg_pool2d(input, (1, 1))


@pytest.mark.parametrize(
    "input_shapes",
    [[(1, 2048, 7, 7)]],
)
def test_adaptive_avg_pool_2d(device, input_shapes):
    m = AdaptiveAvgPool2dModule()
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
    assert [node.target for node in nodes].count(ttnn.global_avg_pool2d) == 1
    # Check inference result
    assert_with_pcc(result_before, result_after)
