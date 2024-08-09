import torch
import torch_ttnn
import pytest
import ttnn

from tests.utils import assert_with_pcc


@pytest.mark.parametrize(
    "input_shapes",
    [[(4, 4), (4, 4)]],
)
def test_fall_back(device, input_shapes):
    class MixModule(torch.nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x, y):
            z = torch.div(x, y)
            z = torch.add(z, z)
            z = torch.matmul(z, z)
            z = torch.div(z, z)
            z = torch.div(z, z)
            return z

    m = MixModule()
    inputs = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
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
    assert target.count(ttnn.reciprocal) == 3
    assert target.count(ttnn.mul) == 3
    assert target.count(ttnn.add) == 1
    assert target.count(ttnn.matmul) == 1

    nodes = list(option._out_fx_graphs[-1].nodes)
    reciprocal_idx = [i for i, n in enumerate(target) if n == ttnn.reciprocal]
    multiply_idx = [i for i, n in enumerate(target) if n == ttnn.multiply]
    assert len(reciprocal_idx) == len(multiply_idx)
    for ri, mi in zip(reciprocal_idx, multiply_idx):
        assert ri < mi
    add_idx = target.index(ttnn.add)
    assert add_idx > multiply_idx[0]
    assert add_idx < multiply_idx[1]
    assert add_idx < multiply_idx[2]
    matmul_idx = target.index(ttnn.matmul)
    assert matmul_idx > multiply_idx[0]
    assert matmul_idx > add_idx
    assert matmul_idx < multiply_idx[1]
    assert matmul_idx < multiply_idx[2]

    # Check inference result
    assert_with_pcc(result_before, result_after)
