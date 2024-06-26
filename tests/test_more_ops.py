import torch
import torch_ttnn
import ttnn
import unittest


class SubModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x - y

    def input_shapes(self):
        return [(4, 4), (4, 4)]


class MulModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.mul(x, y)

    def input_shapes(self):
        return [(4, 4), (4, 4)]


class SoftmaxModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, axis):
        return torch.softmax(x, axis)

    def input_shapes(self):
        return [(1, 1, 64, 32)]


class TanhModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.tanh(x)

    def input_shapes(self):
        return [(4, 4)]


class ReshapeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, new_shape):
        return torch.reshape(x, new_shape)

    def input_shapes(self):
        return [(4, 4)]


class PermuteModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, order):
        return torch.permute(x, order)

    def input_shapes(self):
        return [(4, 4)]


class TestModules(unittest.TestCase):
    def setUp(self):
        # Open device 0
        self.device: ttnn.Device = ttnn.open(0)

    def tearDown(self):
        # Close the device
        ttnn.close(self.device)

    def test_sub(self):
        m = SubModule()
        input_shapes = m.input_shapes()
        inputs = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
        result_before = m.forward(*inputs)
        option = torch_ttnn.TenstorrentBackendOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend="ttnn", options=option)
        result_after = m.forward(*inputs)
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        self.assertTrue(nodes[6].target == ttnn.sub)
        self.assertTrue(nodes[6].args[0].target == ttnn.to_device)
        self.assertTrue(nodes[6].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[7].target == ttnn.from_device)
        self.assertTrue(nodes[8].target == ttnn.to_layout)
        self.assertTrue(nodes[9].target == ttnn.to_torch)
        # Check inference result
        self.assertTrue(torch.allclose(result_before, result_after))

    def test_mul(self):
        m = MulModule()
        input_shapes = m.input_shapes()
        inputs = [
            torch.randint(1, 5, shape).type(torch.bfloat16) for shape in input_shapes
        ]
        result_before = m.forward(*inputs)
        option = torch_ttnn.TenstorrentBackendOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend="ttnn", options=option)
        result_after = m.forward(*inputs)
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        self.assertTrue(nodes[6].target == ttnn.mul)
        self.assertTrue(nodes[6].args[0].target == ttnn.to_device)
        self.assertTrue(nodes[6].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[7].target == ttnn.from_device)
        self.assertTrue(nodes[8].target == ttnn.to_layout)
        self.assertTrue(nodes[9].target == ttnn.to_torch)
        # Check inference result
        self.assertTrue(torch.allclose(result_before, result_after))

    def test_softmax(self):
        m = SoftmaxModule()
        input_shapes = m.input_shapes()
        inputs = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
        axis = -1
        result_before = m.forward(*inputs, axis)
        option = torch_ttnn.TenstorrentBackendOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend="ttnn", options=option)
        result_after = m.forward(*inputs, axis)
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        self.assertTrue(nodes[3].target == ttnn.softmax)
        self.assertTrue(nodes[3].args[0].target == ttnn.to_device)
        self.assertTrue(nodes[3].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[4].target == ttnn.from_device)
        self.assertTrue(nodes[5].target == ttnn.to_layout)
        self.assertTrue(nodes[6].target == ttnn.to_torch)
        # Check inference result
        self.assertTrue(torch.allclose(result_before, result_after, rtol=0.2))

    def test_tanh(self):
        m = TanhModule()
        input_shapes = m.input_shapes()
        inputs = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
        result_before = m.forward(*inputs)
        option = torch_ttnn.TenstorrentBackendOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend="ttnn", options=option)
        result_after = m.forward(*inputs)
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        self.assertTrue(nodes[3].target == ttnn.tanh)
        self.assertTrue(nodes[3].args[0].target == ttnn.to_device)
        self.assertTrue(nodes[3].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[4].target == ttnn.from_device)
        self.assertTrue(nodes[5].target == ttnn.to_layout)
        self.assertTrue(nodes[6].target == ttnn.to_torch)
        # Check inference result
        self.assertTrue(torch.allclose(result_before, result_after, rtol=0.2))

    def test_reshape(self):
        m = ReshapeModule()
        input_shapes = m.input_shapes()
        inputs = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
        new_shape = (2, 8)
        result_before = m.forward(*inputs, new_shape)
        option = torch_ttnn.TenstorrentBackendOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend="ttnn", options=option)
        result_after = m.forward(*inputs, new_shape)
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        self.assertTrue(nodes[3].target == ttnn.reshape)
        self.assertTrue(nodes[3].args[0].target == ttnn.to_device)
        self.assertTrue(nodes[3].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[4].target == ttnn.from_device)
        self.assertTrue(nodes[5].target == ttnn.to_layout)
        self.assertTrue(nodes[6].target == ttnn.to_torch)
        # Check inference result
        self.assertTrue(torch.allclose(result_before, result_after))

    # NOTE(yoco) This test failed because currently
    # the ttnn.permute does nothing. Seems like the ttnn.permute
    # is not implemented yet.
    def test_permute(self):
        m = PermuteModule()
        input_shapes = m.input_shapes()
        inputs = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
        order = (1, 0)
        result_before = m.forward(*inputs, order)
        option = torch_ttnn.TenstorrentBackendOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend="ttnn", options=option)
        result_after = m.forward(*inputs, order)
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        self.assertTrue(nodes[3].target == ttnn.permute)
        self.assertTrue(nodes[3].args[0].target == ttnn.to_device)
        self.assertTrue(nodes[3].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[4].target == ttnn.from_device)
        self.assertTrue(nodes[5].target == ttnn.to_layout)
        self.assertTrue(nodes[6].target == ttnn.to_torch)
        # Check inference result
        self.assertTrue(torch.allclose(result_before, result_after))


if __name__ == "__main__":
    unittest.main()
