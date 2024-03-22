import torch
import torch_ttnn
import unittest
from torch_ttnn import ttnn

from tests.ttnn.utils_for_testing import check_with_pcc

class AddModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return x + y

    def input_shapes(self):
        return [(4, 4), (4, 4)]


class MatmulModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.matmul(x, y)

    def input_shapes(self):
        return [(32, 32), (32, 32)]


class BatchMatmulModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.bmm(x, y)

    def input_shapes(self):
        return [(10, 64, 32), (10, 32, 128)]


# Nested module for demonstration, verify nested modules work
class AddMatmulModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mm = MatmulModule()

    def forward(self, x, y):
        m = torch.add(x, y)
        return self.mm(m, m)

    def input_shapes(self):
        return [(4, 4), (4, 4)]


class TestModules(unittest.TestCase):
    def setUp(self):
        # Open device 0
        self.device: ttnn.Device = ttnn.open_device(device_id = 0)

    def tearDown(self):
        # Close the device
        ttnn.close_device(self.device)

    def test_add(self):
        m = AddModule()
        input_shapes = m.input_shapes()
        inputs = [
            torch.randint(0, 3, shape).type(torch.bfloat16) for shape in input_shapes
        ]
        result_before = m.forward(*inputs)
        option = torch_ttnn.TorchTtnnOption(device=self.device)
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend=torch_ttnn.backend(option))
        result_after = m.forward(*inputs)
        self.assertEqual(1, len(option._out_fx_graphs))
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        self.assertEqual(nodes[8].target, ttnn.add)
        self.assertEqual(nodes[8].args[0].target, ttnn.to_device)
        self.assertEqual(nodes[8].args[0].args[0].target, ttnn.to_layout)
        self.assertEqual(nodes[8].args[0].args[0].args[0].target, ttnn.from_torch)
        self.assertEqual(nodes[9].target, ttnn.from_device)
        self.assertEqual(nodes[10].target, ttnn.to_layout)
        self.assertEqual(nodes[11].target, ttnn.to_torch)
        # Check inference result
        self.assertTrue(torch.allclose(result_before, result_after))

    def test_matmul(self):
        m = MatmulModule()
        input_shapes = m.input_shapes()
        inputs = [
            torch.randint(0, 3, shape).type(torch.bfloat16) for shape in input_shapes
        ]
        result_before = m.forward(*inputs)
        option = torch_ttnn.TorchTtnnOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend=torch_ttnn.backend(option))
        result_after = m.forward(*inputs)
        self.assertEqual(1, len(option._out_fx_graphs))
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        self.assertEqual(nodes[8].target, ttnn.matmul)
        self.assertEqual(nodes[8].args[0].target, ttnn.to_device)
        self.assertEqual(nodes[8].args[0].args[0].target, ttnn.to_layout)
        self.assertEqual(nodes[8].args[0].args[0].args[0].target, ttnn.from_torch)
        self.assertEqual(nodes[9].target, ttnn.from_device)
        self.assertEqual(nodes[10].target, ttnn.to_layout)
        self.assertEqual(nodes[11].target, ttnn.to_torch)
        # Check inference result
        self.assertTrue(torch.allclose(result_before, result_after))

    def test_batchmatmul(self):
        m = BatchMatmulModule()
        input_shapes = m.input_shapes()
        inputs = [
            torch.rand(shape).type(torch.bfloat16) for shape in input_shapes
        ]
        result_before = m.forward(*inputs)
        option = torch_ttnn.TorchTtnnOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend=torch_ttnn.backend(option))
        result_after = m.forward(*inputs)
        self.assertEqual(1, len(option._out_fx_graphs))
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        self.assertEqual(nodes[8].target, ttnn.matmul)
        self.assertEqual(nodes[8].args[0].target, ttnn.to_device)
        self.assertEqual(nodes[8].args[0].args[0].target, ttnn.to_layout)
        self.assertEqual(nodes[8].args[0].args[0].args[0].target, ttnn.from_torch)
        self.assertEqual(nodes[8].args[1].target, ttnn.to_device)
        self.assertEqual(nodes[8].args[1].args[0].target, ttnn.to_layout)
        self.assertEqual(nodes[8].args[1].args[0].args[0].target, ttnn.from_torch)
        self.assertEqual(nodes[9].target, ttnn.from_device)
        self.assertEqual(nodes[10].target, ttnn.to_layout)
        self.assertEqual(nodes[11].target, ttnn.to_torch)
        # Check inference result
        self.assertTrue(check_with_pcc(result_before, result_after))

    def test_add_and_matmul(self):
        m = AddMatmulModule()
        input_shapes = m.input_shapes()
        inputs = [
            torch.randint(0, 3, shape).type(torch.bfloat16) for shape in input_shapes
        ]
        result_before = m.forward(*inputs)
        option = torch_ttnn.TorchTtnnOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend=torch_ttnn.backend(option))
        result_after = m.forward(*inputs)
        self.assertEqual(1, len(option._out_fx_graphs))
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        self.assertEqual(nodes[8].target, ttnn.add)
        self.assertEqual(nodes[8].args[0].target, ttnn.to_device)
        self.assertEqual(nodes[8].args[0].args[0].target, ttnn.to_layout)
        self.assertEqual(nodes[8].args[0].args[0].args[0].target, ttnn.from_torch)
        self.assertEqual(nodes[9].target, ttnn.matmul)
        self.assertEqual(nodes[10].target, ttnn.from_device)
        self.assertEqual(nodes[11].target, ttnn.to_layout)
        self.assertEqual(nodes[12].target, ttnn.to_torch)
        # Check inference result
        self.assertTrue(torch.allclose(result_before, result_after))


if __name__ == "__main__":
    unittest.main()
