import torch
import torch_ttnn
import pytest
import ttnn

from tests.utils import assert_with_pcc


class TransposeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, dim0, dim1):
        return torch.transpose(x, dim0, dim1)


@pytest.mark.parametrize(
    "input_shape, dim0, dim1",
    [
        # Constraint: Last dim of input should be even.
        # If not, this runtime error will be thrown:
        # RuntimeError: TT_FATAL @ ../tt_metal/impl/buffers/buffer.cpp:41: page_size % sizeof(uint32_t) == 0
        ((5, 3, 2), 0, 2),
        ((1, 192, 1344), 1, 2),
    ],
)
def test_transpose(device, input_shape, dim0, dim1):
    m = TransposeModule()
    input = torch.rand(input_shape, dtype=torch.bfloat16)
    result_before = m.forward(input, dim0, dim1)
    option = torch_ttnn.TorchTtnnOption(device=device)
    option.gen_graphviz = True
    # The compilation is lazy, so we need to run forward once to trigger the compilation
    m = torch.compile(m, backend=torch_ttnn.backend, options=option)
    result_after = m.forward(input, dim0, dim1)
    option._out_fx_graphs[0].print_tabular()

    # Check the graph has be rewritten and contain ttnn ops
    nodes = list(option._out_fx_graphs[0].nodes)
    [node.target for node in nodes].count(ttnn.permute) == 1
    # Check inference result
    print(result_before.shape, result_after.shape)
    assert_with_pcc(result_before, result_after)
