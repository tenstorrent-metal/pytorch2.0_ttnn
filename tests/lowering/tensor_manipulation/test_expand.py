import torch
import torch_ttnn
import pytest
import ttnn
import tt_lib
from torch_ttnn.utils import (
    DummyTtnnRowMajorLayout,
    DummyTtnnTileLayout,
)


class ExpandModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, new_shape):
        return x.expand(new_shape)


class ExpandAfterOpModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, new_shape):
        a = torch.clone(x)
        return a.expand(new_shape)


class ExpandBeforeOpModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, new_shape):
        ex = x.expand(new_shape)
        return torch.add(ex, ex)


class ExpandBetweenOpsModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, new_shape):
        a = torch.clone(x)
        ex = a.expand(new_shape)
        return torch.add(ex, ex)


@pytest.mark.parametrize(
    "module_class",
    [
        ExpandModule,
        ExpandAfterOpModule,
        ExpandBeforeOpModule,
        ExpandBetweenOpsModule,
    ],
)
@pytest.mark.parametrize(
    "input_shapes",
    [
        [(1, 4), (4, 4)],
        [(2, 2), (4, 4)],
        [(3, 3), (6, 6)],
    ],
)
def test_expand_modules(compiler_options, module_class, input_shapes):
    m = module_class()
    tensor = torch.rand(input_shapes[0], dtype=torch.bfloat16)
    new_shape = input_shapes[1]
    inputs = [tensor, new_shape]
    result_before = m.forward(*inputs)

    compiler_options.gen_graphviz = True
    # The compilation is lazy, so we need to run forward once to trigger the compilation
    m = torch.compile(m, backend=torch_ttnn.backend, options=compiler_options)
    result_after = m.forward(*inputs)
    compiler_options._out_fx_graphs[0].print_tabular()

    # Check the graph has been rewritten and contains ttnn ops
    nodes = list(compiler_options._out_fx_graphs[0].nodes)
    if module_class == ExpandModule:
        assert nodes[4].target == ttnn.repeat
        assert nodes[4].args[1].target == ttnn.Shape
        assert nodes[5].target == ttnn.from_device
        assert nodes[6].target == ttnn.to_layout
        assert nodes[7].target == ttnn.to_torch
    elif module_class == ExpandAfterOpModule:
        assert nodes[8].target == ttnn.repeat
        assert nodes[8].args[0].target == ttnn.to_layout
        assert nodes[8].args[0].args[0].target == ttnn.clone
        assert isinstance(nodes[8].args[0].args[1], DummyTtnnRowMajorLayout)
        assert nodes[8].args[1].target == ttnn.Shape
        assert nodes[9].target == ttnn.from_device
        assert nodes[10].target == ttnn.to_layout
        assert nodes[11].target == ttnn.to_torch
    elif module_class == ExpandBeforeOpModule:
        assert nodes[4].target == ttnn.repeat
        assert nodes[4].args[1].target == ttnn.Shape
        assert nodes[5].target == ttnn.to_layout
        assert nodes[5].args[0].target == ttnn.repeat
        assert isinstance(nodes[5].args[1], DummyTtnnTileLayout)
        assert nodes[6].target == ttnn.add
        assert nodes[7].target == ttnn.from_device
        assert nodes[8].target == ttnn.to_layout
        assert nodes[9].target == ttnn.to_torch
    elif module_class == ExpandBetweenOpsModule:
        assert nodes[8].target == ttnn.repeat
        assert nodes[8].args[0].target == ttnn.to_layout
        assert nodes[8].args[0].args[0].target == ttnn.clone
        assert isinstance(nodes[8].args[0].args[1], DummyTtnnRowMajorLayout)
        assert nodes[8].args[1].target == ttnn.Shape
        assert nodes[9].target == ttnn.to_layout
        assert nodes[9].args[0].target == ttnn.repeat
        assert isinstance(nodes[9].args[1], DummyTtnnTileLayout)
        assert nodes[10].target == ttnn.add
        assert nodes[11].target == ttnn.from_device
        assert nodes[12].target == ttnn.to_layout
        assert nodes[13].target == ttnn.to_torch

    # Check inference result
    assert torch.allclose(result_before, result_after, rtol=0.2)
