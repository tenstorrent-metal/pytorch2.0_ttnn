import torch
import torch_ttnn
import pytest
import ttnn

from tests.utils import assert_with_pcc


class BaddbmmModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, batch1, batch2, beta=1, alpha=1):
        if beta == 1:
            return torch.baddbmm(input, batch1, batch2, alpha=alpha)
        elif alpha == 1:
            return torch.baddbmm(input, batch1, batch2, beta=beta)
        elif beta == 1 and alpha == 1:
            return torch.baddbmm(input, batch1, batch2)
        else:
            return torch.baddbmm(input, batch1, batch2, beta=beta, alpha=alpha)


@pytest.mark.parametrize(
    "input_shapes",
    [[(10, 64, 128), (10, 64, 32), (10, 32, 128)]],
)
def test_baddbmm(device, input_shapes):
    m = BaddbmmModule()
    inputs = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
    option = torch_ttnn.TorchTtnnOption(device=device)
    option.gen_graphviz = True

    # (1) Test with default alpha and beta values
    result_before = m.forward(*inputs)
    # The compilation is lazy, so we need to run forward once to trigger the compilation
    m_ttnn = torch.compile(m, backend=torch_ttnn.backend, options=option)
    result_after = m_ttnn.forward(*inputs)
    option._out_fx_graphs[-1].print_tabular()

    # Check the graph has be rewritten and contain ttnn ops
    nodes = list(option._out_fx_graphs[-1].nodes)
    target = [node.target for node in nodes]
    assert target.count(ttnn.matmul) == 1
    assert target.count(ttnn.add) == 1
    assert target.index(ttnn.matmul) < target.index(ttnn.add)
    assert nodes[target.index(ttnn.add)].args[1].target == ttnn.matmul
    # Intermediate node meta check if preserved
    for node in nodes:
        if node.target == ttnn.matmul or node.target == ttnn.multiply:
            assert node.meta["val"].size() == input_shapes[0]
    # Check inference result
    assert_with_pcc(result_before, result_after, 0.999)

    # (2) Test with alpha and default beta value
    result_before = m.forward(*inputs, alpha=2)
    m_ttnn = torch.compile(m, backend=torch_ttnn.backend, options=option)
    result_after = m_ttnn.forward(*inputs, alpha=2)
    option._out_fx_graphs[-1].print_tabular()

    nodes = list(option._out_fx_graphs[-1].nodes)
    target = [node.target for node in nodes]
    assert target.count(ttnn.matmul) == 1
    assert target.count(ttnn.multiply) == 1
    assert target.index(ttnn.matmul) < target.index(ttnn.multiply)
    assert nodes[target.index(ttnn.multiply)].args[0].target == ttnn.matmul
    assert target.count(ttnn.add) == 1
    assert target.index(ttnn.multiply) < target.index(ttnn.add)
    assert nodes[target.index(ttnn.add)].args[1].target == ttnn.multiply
    # Intermediate node meta check if preserved
    for node in nodes:
        if node.target == ttnn.matmul or node.target == ttnn.multiply:
            assert node.meta["val"].size() == input_shapes[0]
    assert_with_pcc(result_before, result_after, 0.999)

    # (3) Test with beta and default alpha value
    result_before = m.forward(*inputs, beta=2)
    m_ttnn = torch.compile(m, backend=torch_ttnn.backend, options=option)
    result_after = m_ttnn.forward(*inputs, beta=2)
    option._out_fx_graphs[-1].print_tabular()

    nodes = list(option._out_fx_graphs[-1].nodes)
    target = [node.target for node in nodes]
    assert target.count(ttnn.matmul) == 1
    assert target.count(ttnn.multiply) == 1
    assert target.count(ttnn.add) == 1
    assert target.index(ttnn.matmul) < target.index(ttnn.add)
    assert target.index(ttnn.multiply) < target.index(ttnn.add)
    assert nodes[target.index(ttnn.add)].args[0].target == ttnn.multiply
    assert nodes[target.index(ttnn.add)].args[1].target == ttnn.matmul
    # Intermediate node meta check if preserved
    for node in nodes:
        if node.target == ttnn.matmul or node.target == ttnn.multiply:
            assert node.meta["val"].size() == input_shapes[0]
    assert_with_pcc(result_before, result_after, 0.999)

    # (4) Test with beta and alpha values
    result_before = m.forward(*inputs, beta=2, alpha=2)
    m_ttnn = torch.compile(m, backend=torch_ttnn.backend, options=option)
    result_after = m_ttnn.forward(*inputs, beta=2, alpha=2)
    option._out_fx_graphs[-1].print_tabular()

    nodes = list(option._out_fx_graphs[-1].nodes)
    target = [node.target for node in nodes]
    assert target.count(ttnn.matmul) == 1
    assert target.count(ttnn.multiply) == 2
    assert target.count(ttnn.add) == 1
    multiply_idx = [i for i, n in enumerate(target) if n == ttnn.multiply]
    assert target.index(ttnn.matmul) < multiply_idx[0]
    assert nodes[multiply_idx[0]].args[0].target == ttnn.matmul
    add_index = target.index(ttnn.add)
    assert multiply_idx[0] < add_index
    assert multiply_idx[1] < add_index
    assert nodes[add_index].args[0].target == ttnn.multiply
    assert nodes[add_index].args[1].target == ttnn.multiply
    # Intermediate node meta check if preserved
    for node in nodes:
        if node.target == ttnn.matmul or node.target == ttnn.multiply:
            assert node.meta["val"].size() == input_shapes[0]
    assert_with_pcc(result_before, result_after, 0.999)

    # (5) Test special case when beta is 0
    result_before = m.forward(*inputs, beta=0, alpha=2)
    m_ttnn = torch.compile(m, backend=torch_ttnn.backend, options=option)
    result_after = m_ttnn.forward(*inputs, beta=0, alpha=2)
    option._out_fx_graphs[-1].print_tabular()

    nodes = list(option._out_fx_graphs[-1].nodes)
    target = [node.target for node in nodes]
    assert target.count(ttnn.matmul) == 1
    assert target.count(ttnn.multiply) == 1
    assert target.index(ttnn.matmul) < target.index(ttnn.multiply)
    assert nodes[target.index(ttnn.multiply)].args[0].target == ttnn.matmul
    # Intermediate node meta check if preserved
    for node in nodes:
        if node.target == ttnn.matmul or node.target == ttnn.multiply:
            assert node.meta["val"].size() == input_shapes[0]
    assert_with_pcc(result_before, result_after, 0.999)
