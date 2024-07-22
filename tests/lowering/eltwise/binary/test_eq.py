import torch
import torch_ttnn
import pytest
import ttnn


class EqModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input1, input2):
        return torch.eq(input1, input2)


@pytest.mark.parametrize(
    "input_shapes",
    [[(4, 4), (4, 4)]],
)
def test_eq_tensor(device, input_shapes):
    m = EqModule()
    inputs = [
        torch.randint(0, 2, shape, dtype=torch.bfloat16) for shape in input_shapes
    ]
    result_before = m.forward(*inputs)
    option = torch_ttnn.TorchTtnnOption(device=device)
    option.gen_graphviz = True
    # The compilation is lazy, so we need to run forward once to trigger the compilation
    m = torch.compile(m, backend=torch_ttnn.backend, options=option)
    result_after = m.forward(*inputs)
    option._out_fx_graphs[0].print_tabular()

    # Check the graph has be rewritten and contain ttnn ops
    nodes = list(option._out_fx_graphs[0].nodes)
    assert [node.target for node in nodes].count(ttnn.eq) == 1
    # Check inference result
    assert torch.allclose(result_before, result_after.to(torch.bool))


@pytest.mark.parametrize(
    "input_shapes",
    [[(64, 128)]],
)
def test_eq_scalar(device, input_shapes):
    m = EqModule()
    inputs = [
        torch.randint(0, 2, shape, dtype=torch.bfloat16) for shape in input_shapes
    ]
    scalar = torch.randint(0, 2, (1,)).item()
    result_before = m.forward(inputs[0], scalar)
    option = torch_ttnn.TorchTtnnOption(device=device)
    option.gen_graphviz = True
    # The compilation is lazy, so we need to run forward once to trigger the compilation
    m = torch.compile(m, backend=torch_ttnn.backend, options=option)
    result_after = m.forward(inputs[0], scalar)
    option._out_fx_graphs[0].print_tabular()

    # Check the graph has be rewritten and contain ttnn ops
    nodes = list(option._out_fx_graphs[0].nodes)
    target = [node.target for node in nodes]
    assert target.count(ttnn.full) == 1
    assert target.count(ttnn.eq) == 1
    assert target.index(ttnn.full) < target.index(ttnn.eq)
    # Check inference result
    assert torch.allclose(result_before, result_after.to(torch.bool))
