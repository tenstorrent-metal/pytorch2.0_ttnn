import torch
import torch_ttnn
import pytest
import ttnn

from tests.utils import assert_with_pcc


class ClampModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, min, max):
        return torch.clamp(input, min=min, max=max)


@pytest.mark.parametrize(
    "input_shapes",
    [[(4, 4)]],
)
def test_clamp(device, input_shapes):
    m = ClampModule()
    inputs = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
    min, max = -0.5, 0.5
    result_before = m.forward(inputs[0], min, max)
    option = torch_ttnn.TorchTtnnOption(device=device)
    option.gen_graphviz = True
    # The compilation is lazy, so we need to run forward once to trigger the compilation
    m = torch.compile(m, backend=torch_ttnn.backend, options=option)
    result_after = m.forward(inputs[0], min, max)
    option._out_fx_graphs[0].print_tabular()

    # Check the graph has be rewritten and contain ttnn ops
    nodes = list(option._out_fx_graphs[0].nodes)
    assert [node.target for node in nodes].count(ttnn.clip) == 1
    # Check inference result
    assert_with_pcc(result_before, result_after)
