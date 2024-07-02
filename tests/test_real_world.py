import torch
import torchvision
import torch_ttnn
import unittest
from torch_ttnn import ttnn
import collections
import os


class TestRealWorld(unittest.TestCase):
    def setUp(self):
        # Open device 0
        self.device: ttnn.Device = ttnn.open_device(device_id=0)

    def tearDown(self):
        # Close the device
        ttnn.close_device(self.device)

    @unittest.skipUnless(
        os.getenv("RUN_INTEGRATION_TESTS", "0") == "1",
        "Skip this test because it takes 135 MB to download the ResNet18 model. Set RUN_INTEGRATION_TESTS=1 to run it.",
    )
    def test_resnet(self):
        # Download model from cloud
        model = torchvision.models.get_model("resnet18", pretrained=True)
        model.eval()
        model = model.to(torch.bfloat16)

        # Create random input tensor
        input_batch = torch.rand((1, 3, 224, 224), dtype=torch.bfloat16)

        # Run inference with the original model
        with torch.no_grad():
            output_before = model(input_batch)

        # Compile the model
        option = torch_ttnn.TorchTtnnOption(device=self.device)
        option.gen_graphviz = True
        model = torch.compile(model, backend=torch_ttnn.backend(option))

        # Run inference with the compiled model
        with torch.no_grad():
            output_after = model(input_batch)

        # Check the graph has be rewritten and contain ttnn ops
        target_count = collections.Counter()
        for n in option._out_fx_graphs[0].nodes:
            target_count[n.target] += 1
        self.assertEqual(target_count[ttnn.from_torch], 17)
        self.assertEqual(target_count[ttnn.to_device], 17)
        self.assertEqual(target_count[ttnn.from_device], 9)
        self.assertEqual(target_count[ttnn.to_torch], 9)
        self.assertEqual(target_count[ttnn.add], 8)
        self.assertEqual(target_count[ttnn.reshape], 1)
        self.assertEqual(target_count[ttnn.matmul], 0)

        # Check inference result
        self.assertTrue(torch.allclose(output_before, output_after))
