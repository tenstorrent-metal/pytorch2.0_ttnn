import torch
import torch_ttnn
import ttnn
import unittest
from torch.fx.passes.dialect.common.cse_pass import CSEPass


class IfModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        if torch.sum(x) > 0:
            return x + x
        else:
            return torch.matmul(x, x)

    def input_shapes(self):
        return [(4, 4)]


class TestModules(unittest.TestCase):
    def setUp(self):
        # Open device 0
        self.device: ttnn.Device = ttnn.open(0)

    def tearDown(self):
        # Close the device
        ttnn.close(self.device)

    def test_if(self):
        m = IfModule()
        inputs_then = [torch.tensor(range(1, 17)).reshape(4, 4).to(torch.bfloat16)]
        inputs_else = [-inputs_then[0]]
        result_before_then = m.forward(*inputs_then)
        result_before_else = m.forward(*inputs_else)
        option = torch_ttnn.TenstorrentBackendOption(device=self.device)
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend="ttnn", options=option)
        result_after_then = m.forward(*inputs_then)
        result_after_else = m.forward(*inputs_else)

        # After the forward & compilation, there should be total 3 graphs
        self.assertEqual(3, len(option._out_fx_graphs))
        option._out_fx_graphs[0].print_tabular()
        # Check the graph has be rewritten and contain ttnn ops
        nodes_0 = list(option._out_fx_graphs[0].nodes)
        self.assertEqual(len(nodes_0), 4)
        self.assertEqual(nodes_0[1].target, torch.ops.aten.sum.default)
        self.assertEqual(nodes_0[2].target, torch.ops.aten.gt.Scalar)
        nodes_1 = list(option._out_fx_graphs[1].nodes)
        self.assertEqual(len(nodes_1), 8)
        self.assertEqual(nodes_1[1].target, ttnn.from_torch)
        self.assertEqual(nodes_1[2].target, ttnn.to_device)
        self.assertEqual(nodes_1[3].target, ttnn.add)
        self.assertEqual(nodes_1[4].target, ttnn.from_device)
        self.assertEqual(nodes_1[5].target, ttnn.to_layout)
        self.assertEqual(nodes_1[6].target, ttnn.to_torch)
        nodes_2 = list(option._out_fx_graphs[2].nodes)
        self.assertEqual(len(nodes_2), 8)
        self.assertEqual(nodes_2[1].target, ttnn.from_torch)
        self.assertEqual(nodes_2[2].target, ttnn.to_device)
        self.assertEqual(nodes_2[3].target, ttnn.matmul)
        self.assertEqual(nodes_2[4].target, ttnn.from_device)
        self.assertEqual(nodes_2[5].target, ttnn.to_layout)
        self.assertEqual(nodes_2[6].target, ttnn.to_torch)

        # Check inference result
        self.assertTrue(torch.allclose(result_before_then, result_after_then))
        self.assertTrue(torch.allclose(result_before_else, result_after_else))
