import os
import shutil
import torch
# from torch_ttnn import torch_stat
from torch_ttnn.tracer import Tracer
import unittest
from torch._dynamo.backends.common import aot_autograd

class ConvModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_channels=3,
            out_channels=5,
            kernel_size=(3, 4),
            stride=(1, 2),
            padding=(2, 1),
            padding_mode="zeros",
            dilation=(1, 3),
            groups=1,
            bias=True,
        )

    def forward(self, x):
        return self.conv(x)

    def input_shapes(self):
        return [(1, 3, 32, 32)]


class TestModules(unittest.TestCase):
    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)
        self.out_folder = os.path.join(os.getcwd(), ".test_stat")

    def setUp(self):
        shutil.rmtree(self.out_folder, ignore_errors=True)

    def tearDown(self):
        shutil.rmtree(self.out_folder, ignore_errors=True)

    @staticmethod
    def dummy_backend(gm, example_inputs):
        return gm

    def test_conv(self):
        m = ConvModule()
        input_shapes = m.input_shapes()
        inputs = [torch.rand(shape, requires_grad=True) for shape in input_shapes]

        aot_backend = aot_autograd(fw_compiler = Tracer(self.dummy_backend, "fw_conv", self.out_folder))
        m = torch.compile(m, backend=aot_backend)
        _ = m.forward(*inputs)

        fw_result_json_path = os.path.join(self.out_folder, "raw", "fw_conv_orig_0.json")
        self.assertTrue(os.path.isfile(fw_result_json_path))

    def test_conv_with_backward(self):
        m = ConvModule()
        input_shapes = m.input_shapes()
        inputs = [torch.rand(shape, requires_grad=True) for shape in input_shapes]

        aot_backend = aot_autograd(fw_compiler = Tracer(self.dummy_backend, "fw_conv", self.out_folder),
                                   bw_compiler = Tracer(self.dummy_backend, "bw_conv", self.out_folder))
        m = torch.compile(m, backend=aot_backend)
        result = m.forward(*inputs)
        result.backward(torch.ones_like(result))

        fw_result_json_path = os.path.join(self.out_folder, "raw", "fw_conv_orig_0.json")
        bw_result_json_path = os.path.join(self.out_folder, "raw", "bw_conv_orig_0.json")
        self.assertTrue(os.path.isfile(fw_result_json_path))
        self.assertTrue(os.path.isfile(bw_result_json_path))

    def test_conv_with_modi(self):
        m = ConvModule()
        input_shapes = m.input_shapes()
        inputs = [torch.rand(shape, requires_grad=True) for shape in input_shapes]

        aot_backend = aot_autograd(fw_compiler = Tracer(self.dummy_backend, "fw_conv", self.out_folder, trace_modi=True))
        m = torch.compile(m, backend=aot_backend)
        _ = m.forward(*inputs)

        fw_result_json_path = os.path.join(self.out_folder, "raw", "fw_conv_orig_0.json")
        fw_modi_result_json_path = os.path.join(self.out_folder, "raw", "fw_conv_modi_0.json")
        self.assertTrue(os.path.isfile(fw_result_json_path))
        self.assertTrue(os.path.isfile(fw_modi_result_json_path))