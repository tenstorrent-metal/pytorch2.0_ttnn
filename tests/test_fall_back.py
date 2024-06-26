import torch
import torch_ttnn
import ttnn
import unittest


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

    def input_shapes(self):
        return [(4, 4), (4, 4)]


class TestModules(unittest.TestCase):
    def setUp(self):
        # Open device 0
        self.device: ttnn.Device = ttnn.open(0)

    def tearDown(self):
        # Close the device
        ttnn.close(self.device)

    def test_fall_back(self):
        m = MixModule()
        input_shapes = m.input_shapes()
        inputs = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
        result_before = m.forward(*inputs)
        option = torch_ttnn.TenstorrentBackendOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend="ttnn", options=option)
        result_after = m.forward(*inputs)
        self.assertEqual(1, len(option._out_fx_graphs))
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        self.assertEqual(nodes[3].target, ttnn.from_torch)
        self.assertEqual(nodes[4].target, ttnn.to_device)
        self.assertEqual(nodes[5].target, ttnn.add)
        self.assertEqual(nodes[6].target, ttnn.matmul)
        self.assertEqual(nodes[7].target, ttnn.from_device)
        self.assertEqual(nodes[8].target, ttnn.to_layout)
        self.assertEqual(nodes[9].target, ttnn.to_torch)
        # Check inference result
        self.assertTrue(torch.allclose(result_before, result_after))
