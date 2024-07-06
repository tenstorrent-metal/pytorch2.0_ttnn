import torch
import torch_ttnn
import unittest
import ttnn
import tt_lib

from torch_ttnn.utils import check_with_pcc


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
        return [(32, 2 * 32)]

    def output_shapes(self):
        return [(2 * 32, 32)]


class ReshapeNegativeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, new_shape):
        return torch.reshape(x, new_shape)

    def input_shapes(self):
        return [(32, 2 * 32)]

    def output_shapes(self):
        return [(-1,)]


class Reshape4DModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, new_shape):
        return torch.reshape(x, new_shape)

    def input_shapes(self):
        return [(64, 32, 16, 32)]

    def output_shapes(self):
        return [(16, 32, 64, 32)]


class Reshape4DNegativeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, new_shape):
        return torch.reshape(x, new_shape)

    def input_shapes(self):
        return [(1, 4, 64, 32)]

    def output_shapes(self):
        return [(1, -1, 2, 32)]


class PermuteModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, order):
        return torch.permute(x, order)

    def input_shapes(self):
        return [(4, 4)]


class ReluModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.relu(x)

    def input_shapes(self):
        return [(4, 4)]


class AddMmModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, mat1, mat2):
        return torch.addmm(input, mat1, mat2)

    def input_shapes(self):
        return [(4, 4), (4, 4), (4, 4)]


class DivModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, numerator, denominator):
        return torch.div(numerator, denominator)

    def input_shapes(self):
        return [(4, 4), (4, 4)]


class DivScalarDenomModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, numerator, denominator):
        return torch.div(numerator, denominator)

    def input_shapes(self):
        return [(4, 4)]


class GeluModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.nn.functional.gelu(input)

    def input_shapes(self):
        return [(4, 4)]


class RSubModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        return torch.rsub(x, y)

    def input_shapes(self):
        return [(4, 4), (4, 4)]


class RSubScalarModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, scalar):
        return torch.rsub(x, scalar)

    def input_shapes(self):
        return [(4, 4)]


class EmbeddingModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, weight):
        embedding = torch.nn.Embedding.from_pretrained(weight)
        return embedding(input)

    def input_shapes(self):
        return [((1, 2, 4, 5), (4, 3, 2, 9)), (10, 4)]


class EmbeddingTileLayoutModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, weights):
        return torch.nn.functional.embedding(input, weights)

    def input_shapes(self):
        # from test_bloom_embedding at tt-metal/tests/ttnn/unit_tests/operations/test_embedding.py
        batch_size = 8
        sentence_size = 384
        vocabulary_size = 250880
        hidden_embedding_dim = 1024
        return [
            (0, vocabulary_size - 1, (batch_size, sentence_size)),
            ((vocabulary_size, hidden_embedding_dim)),
        ]


class CloneFromNodeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        a = input + input
        return torch.clone(a)

    def input_shapes(self):
        return [(4, 4)]


class CloneFromArgModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.clone(input)

    def input_shapes(self):
        return [(4, 4)]


class LayerNormModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, embedding, weight, bias):
        return torch.nn.functional.layer_norm(
            embedding, normalized_shape=[embedding.shape[-1]], weight=weight, bias=bias
        )

    def input_shapes(self):
        batch, sentence_length, embedding_dim = 2, 32, 64
        # [embedding, weight, bias]
        return [
            (batch, sentence_length, embedding_dim),
            (embedding_dim),
            (embedding_dim),
        ]


class LayerNormWithOtherOpModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, embedding, weight, bias):
        layer_norm = torch.nn.functional.layer_norm(
            embedding, normalized_shape=[embedding.shape[-1]], weight=weight, bias=bias
        )
        return layer_norm + layer_norm

    def input_shapes(self):
        batch, sentence_length, embedding_dim = 2, 32, 64
        # [embedding, weight, bias]
        return [
            (batch, sentence_length, embedding_dim),
            (embedding_dim),
            (embedding_dim),
        ]


class NegModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.neg(input)

    def input_shapes(self):
        return [(4)]


class OnesModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, shape):
        return torch.ones(shape)

    def input_shapes(self):
        return [(32, 32)]


class TrilModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.tril(input)

    def input_shapes(self):
        return [(4, 4)]


class ArangeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, end):
        # start = 0, step = 1
        return torch.arange(end)

    def input_shapes(self):
        return [100]


class ArangeStartModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, start, end):
        # step = 1
        return torch.arange(start, end)

    def input_shapes(self):
        # ttnn.arange does not support star values less than 2?
        return [2, 100]


class ArangeStartStepModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, start, end, step):
        return torch.arange(start, end, step)

    def input_shapes(self):
        # ttnn.arange does not support star values less than 2?
        return [4, 100, 3]


class EqTensorModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor1, tensor2):
        return torch.eq(tensor1, tensor2)

    def input_shapes(self):
        return [(4, 4), (4, 4)]


class EqScalarModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, tensor, scalar):
        return torch.eq(tensor, scalar)

    def input_shapes(self):
        return [(64, 128)]


class LogicalNotModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.logical_not(input)

    def input_shapes(self):
        return [(4, 4)]


class ZerosLikeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.zeros_like(input)

    def input_shapes(self):
        return [(4, 4)]


class MeanDimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, dim, keepdim=False):
        return torch.mean(input, dim, keepdim)

    def input_shapes(self):
        return [(1, 32, 32), -1]


class PowTensorScalarModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        # torch.fx lowers this into aten.pow.Tensor_Scalar
        square = torch.square(input)
        return square

    def input_shapes(self):
        return [(4, 4)]


class RsqrtModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.rsqrt(input)

    def input_shapes(self):
        return [(4, 4)]


class SiluModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch.nn.functional.silu(input)

    def input_shapes(self):
        return [(4, 4)]


class AdaptiveAvgPool2dModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return torch._adaptive_avg_pool2d(input, (1, 1))

    def input_shapes(self):
        return [(1, 2048, 7, 7)]


class ClampModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, min, max):
        return torch.clamp(input, min=min, max=max)

    def input_shapes(self):
        return [(4, 4)]


class SqueezeDimModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, dim):
        return torch.squeeze(input, dim)

    def input_shapes(self):
        return [(1, 32, 16)]


class FullModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, size, fill_value):
        return torch.full(size, fill_value)

    def input_shapes(self):
        return [(64, 128)]


class LtTensorModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, tensor):
        return torch.lt(input, tensor)

    def input_shapes(self):
        return [(4, 4), (4, 4)]


class LtScalarModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, scalar):
        return torch.lt(input, scalar)

    def input_shapes(self):
        return [(64, 128)]


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

    def input_shapes(self):
        # input, batch1, batch2
        return [(10, 64, 128), (10, 64, 32), (10, 32, 128)]


class CosModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.cos(x)

    def input_shapes(self):
        return [(4, 4)]


class SigmoidModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sigmoid(x)

    def input_shapes(self):
        return [(4, 4)]


class ToCopyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.to(torch.bfloat16)

    def input_shapes(self):
        return [(4, 4)]


class ToCopyWithOpAfterModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        to = x.to(torch.bfloat16)
        return torch.add(to, to)

    def input_shapes(self):
        return [(4, 4)]


class TestModules(unittest.TestCase):
    def setUp(self):
        # Open device 0
        self.device: ttnn.Device = ttnn.open_device(device_id=0)
        # For AutoFormat
        tt_lib.device.SetDefaultDevice(self.device)

    def tearDown(self):
        # Close the device
        ttnn.close_device(self.device)

    def test_sub(self):
        m = SubModule()
        input_shapes = m.input_shapes()
        inputs = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
        result_before = m.forward(*inputs)
        option = torch_ttnn.TorchTtnnOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend=torch_ttnn.backend, options=option)
        result_after = m.forward(*inputs)
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        self.assertTrue(nodes[8].target == ttnn.sub)
        self.assertTrue(nodes[8].args[0].target == ttnn.to_device)
        self.assertTrue(nodes[8].args[0].args[0].target == ttnn.to_layout)
        self.assertTrue(nodes[8].args[0].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[9].target == ttnn.from_device)
        self.assertTrue(nodes[10].target == ttnn.to_layout)
        self.assertTrue(nodes[11].target == ttnn.to_torch)
        # Check inference result
        self.assertTrue(torch.allclose(result_before, result_after))

    def test_mul(self):
        m = MulModule()
        input_shapes = m.input_shapes()
        inputs = [
            torch.randint(1, 5, shape).type(torch.bfloat16) for shape in input_shapes
        ]
        result_before = m.forward(*inputs)
        option = torch_ttnn.TorchTtnnOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend=torch_ttnn.backend, options=option)
        result_after = m.forward(*inputs)
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        self.assertTrue(nodes[8].target == ttnn.mul)
        self.assertTrue(nodes[8].args[0].target == ttnn.to_device)
        self.assertTrue(nodes[8].args[0].args[0].target == ttnn.to_layout)
        self.assertTrue(nodes[8].args[0].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[9].target == ttnn.from_device)
        self.assertTrue(nodes[10].target == ttnn.to_layout)
        self.assertTrue(nodes[11].target == ttnn.to_torch)
        # Check inference result
        self.assertTrue(torch.allclose(result_before, result_after))

    def test_softmax(self):
        m = SoftmaxModule()
        input_shapes = m.input_shapes()
        inputs = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
        axis = -1
        result_before = m.forward(*inputs, axis)
        option = torch_ttnn.TorchTtnnOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend=torch_ttnn.backend, options=option)
        result_after = m.forward(*inputs, axis)
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        self.assertTrue(nodes[4].target == ttnn.softmax)
        self.assertTrue(nodes[4].args[0].target == ttnn.to_device)
        self.assertTrue(nodes[4].args[0].args[0].target == ttnn.to_layout)
        self.assertTrue(nodes[4].args[0].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[5].target == ttnn.from_device)
        self.assertTrue(nodes[6].target == ttnn.to_layout)
        self.assertTrue(nodes[7].target == ttnn.to_torch)
        # Check inference result
        self.assertTrue(torch.allclose(result_before, result_after, rtol=0.2))

    def test_tanh(self):
        m = TanhModule()
        input_shapes = m.input_shapes()
        inputs = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
        result_before = m.forward(*inputs)
        option = torch_ttnn.TorchTtnnOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend=torch_ttnn.backend, options=option)
        result_after = m.forward(*inputs)
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        self.assertTrue(nodes[4].target == ttnn.tanh)
        self.assertTrue(nodes[4].args[0].target == ttnn.to_device)
        self.assertTrue(nodes[4].args[0].args[0].target == ttnn.to_layout)
        self.assertTrue(nodes[4].args[0].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[5].target == ttnn.from_device)
        self.assertTrue(nodes[6].target == ttnn.to_layout)
        self.assertTrue(nodes[7].target == ttnn.to_torch)
        # Check inference result
        self.assertTrue(torch.allclose(result_before, result_after, rtol=0.2))

    @unittest.skip(
        "NOTE(kevinwuTT) ttnn.reshape conversion needs to be reworked to support the many restrictions."
    )
    def test_reshape(self):
        m = ReshapeModule()
        input_shapes = m.input_shapes()
        inputs = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
        new_shape = m.output_shapes()
        result_before = m.forward(*inputs, *new_shape)
        option = torch_ttnn.TorchTtnnOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend=torch_ttnn.backend, options=option)
        result_after = m.forward(*inputs, *new_shape)
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        self.assertTrue(nodes[2].target == ttnn.reshape)
        self.assertTrue(nodes[2].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[3].target == ttnn.to_torch)
        # Check inference result
        self.assertTrue(torch.allclose(result_before, result_after))

    @unittest.skip(
        "NOTE(kevinwuTT) ttnn.reshape conversion needs to be reworked to support the many restrictions."
    )
    def test_reshape_negative(self):
        m = ReshapeNegativeModule()
        input_shapes = m.input_shapes()
        inputs = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
        new_shape = m.output_shapes()
        result_before = m.forward(*inputs, *new_shape)
        option = torch_ttnn.TorchTtnnOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend=torch_ttnn.backend, options=option)
        result_after = m.forward(*inputs, *new_shape)
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        self.assertTrue(nodes[2].target == ttnn.reshape)
        self.assertTrue(nodes[2].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[3].target == ttnn.to_torch)
        # Check inference result
        self.assertTrue(torch.allclose(result_before, result_after))

    @unittest.skip(
        "NOTE(kevinwuTT) ttnn.reshape conversion needs to be reworked to support the many restrictions."
    )
    def test_reshape_4d(self):
        m = Reshape4DModule()
        input_shapes = m.input_shapes()
        inputs = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
        new_shape = m.output_shapes()
        result_before = m.forward(*inputs, *new_shape)
        option = torch_ttnn.TorchTtnnOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend=torch_ttnn.backend, options=option)
        result_after = m.forward(*inputs, *new_shape)
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

    @unittest.skip(
        "NOTE(kevinwuTT) ttnn.reshape conversion needs to be reworked to support the many restrictions."
    )
    def test_reshape_4d_negative(self):
        m = Reshape4DNegativeModule()
        input_shapes = m.input_shapes()
        inputs = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
        new_shape = m.output_shapes()
        result_before = m.forward(*inputs, *new_shape)
        option = torch_ttnn.TorchTtnnOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend=torch_ttnn.backend, options=option)
        result_after = m.forward(*inputs, *new_shape)
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

    @unittest.skip(
        "NOTE(yoco) This test failed because currently the ttnn.permute does nothing. Seems like the ttnn.permute is not implemented yet."
    )
    def test_permute(self):
        m = PermuteModule()
        input_shapes = m.input_shapes()
        inputs = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
        order = (1, 0)
        result_before = m.forward(*inputs, order)
        option = torch_ttnn.TorchTtnnOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend=torch_ttnn.backend, options=option)
        result_after = m.forward(*inputs, order)
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        self.assertTrue(nodes[4].target == ttnn.permute)
        self.assertTrue(nodes[4].args[0].target == ttnn.to_device)
        self.assertTrue(nodes[4].args[0].args[0].target == ttnn.to_layout)
        self.assertTrue(nodes[4].args[0].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[5].target == ttnn.from_device)
        self.assertTrue(nodes[6].target == ttnn.to_layout)
        self.assertTrue(nodes[7].target == ttnn.to_torch)
        # Check inference result
        self.assertTrue(torch.allclose(result_before, result_after))

    def test_relu(self):
        m = ReluModule()
        input_shapes = m.input_shapes()
        inputs = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
        result_before = m.forward(*inputs)
        option = torch_ttnn.TorchTtnnOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend=torch_ttnn.backend, options=option)
        result_after = m.forward(*inputs)
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        self.assertTrue(nodes[4].target == ttnn.relu)
        self.assertTrue(nodes[4].args[0].target == ttnn.to_device)
        self.assertTrue(nodes[4].args[0].args[0].target == ttnn.to_layout)
        self.assertTrue(nodes[4].args[0].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[5].target == ttnn.from_device)
        self.assertTrue(nodes[6].target == ttnn.to_layout)
        self.assertTrue(nodes[7].target == ttnn.to_torch)
        # Check inference result
        self.assertTrue(torch.allclose(result_before, result_after))

    def test_addmm(self):
        m = AddMmModule()
        input_shapes = m.input_shapes()
        inputs = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
        result_before = m.forward(*inputs)
        option = torch_ttnn.TorchTtnnOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend=torch_ttnn.backend, options=option)
        result_after = m.forward(*inputs)
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        self.assertTrue(nodes[9].target == ttnn.matmul)
        self.assertTrue(nodes[9].args[0].target == ttnn.to_device)
        self.assertTrue(nodes[9].args[0].args[0].target == ttnn.to_layout)
        self.assertTrue(nodes[9].args[0].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[9].args[1].target == ttnn.to_device)
        self.assertTrue(nodes[9].args[1].args[0].target == ttnn.to_layout)
        self.assertTrue(nodes[9].args[1].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[13].target == ttnn.add)
        self.assertTrue(nodes[13].args[0].target == ttnn.to_device)
        self.assertTrue(nodes[13].args[0].args[0].target == ttnn.to_layout)
        self.assertTrue(nodes[13].args[0].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[13].args[1].target == ttnn.matmul)
        self.assertTrue(nodes[14].target == ttnn.from_device)
        self.assertTrue(nodes[15].target == ttnn.to_layout)
        self.assertTrue(nodes[16].target == ttnn.to_torch)
        # Check inference result
        self.assertTrue(check_with_pcc(result_before, result_after))

    def test_div(self):
        m = DivModule()
        input_shapes = m.input_shapes()
        inputs = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
        result_before = m.forward(*inputs)
        option = torch_ttnn.TorchTtnnOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend=torch_ttnn.backend, options=option)
        result_after = m.forward(*inputs)
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        self.assertTrue(nodes[5].target == ttnn.reciprocal)
        self.assertTrue(nodes[5].args[0].target == ttnn.to_device)
        self.assertTrue(nodes[5].args[0].args[0].target == ttnn.to_layout)
        self.assertTrue(nodes[5].args[0].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[9].target == ttnn.mul)
        self.assertTrue(nodes[9].args[0].target == ttnn.to_device)
        self.assertTrue(nodes[9].args[0].args[0].target == ttnn.to_layout)
        self.assertTrue(nodes[9].args[0].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[9].args[1].target == ttnn.reciprocal)
        self.assertTrue(nodes[10].target == ttnn.from_device)
        self.assertTrue(nodes[11].target == ttnn.to_layout)
        self.assertTrue(nodes[12].target == ttnn.to_torch)
        # Check inference result
        self.assertTrue(check_with_pcc(result_before, result_after))

    def test_div_scalar_denom(self):
        m = DivScalarDenomModule()
        input_shapes = m.input_shapes()
        inputs = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
        result_before = m.forward(*inputs, 5.0)
        option = torch_ttnn.TorchTtnnOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend=torch_ttnn.backend, options=option)
        result_after = m.forward(*inputs, 5.0)
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        self.assertTrue(nodes[1].target == ttnn.full)
        self.assertTrue(nodes[3].target == ttnn.reciprocal)
        self.assertTrue(nodes[3].args[0].target == ttnn.to_layout)
        self.assertTrue(nodes[7].target == ttnn.mul)
        self.assertTrue(nodes[7].args[0].target == ttnn.to_device)
        self.assertTrue(nodes[7].args[0].args[0].target == ttnn.to_layout)
        self.assertTrue(nodes[7].args[0].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[7].args[1].target == ttnn.reciprocal)
        self.assertTrue(nodes[8].target == ttnn.from_device)
        self.assertTrue(nodes[9].target == ttnn.to_layout)
        self.assertTrue(nodes[10].target == ttnn.to_torch)
        # Check inference result
        self.assertTrue(check_with_pcc(result_before, result_after))

    def test_gelu(self):
        m = GeluModule()
        input_shapes = m.input_shapes()
        inputs = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
        result_before = m.forward(*inputs)
        option = torch_ttnn.TorchTtnnOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend=torch_ttnn.backend, options=option)
        result_after = m.forward(*inputs)
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        self.assertTrue(nodes[4].target == ttnn.gelu)
        self.assertTrue(nodes[4].args[0].target == ttnn.to_device)
        self.assertTrue(nodes[4].args[0].args[0].target == ttnn.to_layout)
        self.assertTrue(nodes[4].args[0].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[5].target == ttnn.from_device)
        self.assertTrue(nodes[6].target == ttnn.to_layout)
        self.assertTrue(nodes[7].target == ttnn.to_torch)
        # Check inference result
        print(result_before, "\n", result_after)
        self.assertTrue(check_with_pcc(result_before, result_after))

    def test_rsub(self):
        m = RSubModule()
        input_shapes = m.input_shapes()
        inputs = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
        result_before = m.forward(*inputs)
        option = torch_ttnn.TorchTtnnOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend=torch_ttnn.backend, options=option)
        result_after = m.forward(*inputs)
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        self.assertTrue(nodes[8].target == ttnn.sub)
        self.assertTrue(nodes[8].args[0].target == ttnn.to_device)
        self.assertTrue(nodes[8].args[0].args[0].target == ttnn.to_layout)
        self.assertTrue(nodes[8].args[0].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[9].target == ttnn.from_device)
        self.assertTrue(nodes[10].target == ttnn.to_layout)
        self.assertTrue(nodes[11].target == ttnn.to_torch)
        # Check inference result
        self.assertTrue(torch.allclose(result_before, result_after))

    def test_rsub_scalar(self):
        m = RSubScalarModule()
        input_shapes = m.input_shapes()
        inputs = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
        result_before = m.forward(*inputs, 5)
        option = torch_ttnn.TorchTtnnOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend=torch_ttnn.backend, options=option)
        result_after = m.forward(*inputs, 5)
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        self.assertTrue(nodes[1].target == ttnn.full)
        self.assertTrue(nodes[6].target == ttnn.sub)
        self.assertTrue(nodes[6].args[0].target == ttnn.to_layout)
        self.assertTrue(nodes[6].args[1].target == ttnn.to_device)
        self.assertTrue(nodes[6].args[1].args[0].target == ttnn.to_layout)
        self.assertTrue(nodes[6].args[1].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[7].target == ttnn.from_device)
        self.assertTrue(nodes[8].target == ttnn.to_layout)
        self.assertTrue(nodes[9].target == ttnn.to_torch)
        # Check inference result
        self.assertTrue(check_with_pcc(result_before, result_after, 0.9998))

    @unittest.skip(
        "NOTE(kevinwuTT) Re-enable after conversion to ttnn.embedding with both ROW_MAJOR_LAYOUT and TILE_LAYOUT"
    )
    def test_embedding(self):
        m = EmbeddingModule()
        input_shapes = m.input_shapes()
        input = torch.tensor(input_shapes[0])
        weight = torch.rand(input_shapes[1], dtype=torch.bfloat16)
        result_before = m.forward(input, weight)
        option = torch_ttnn.TorchTtnnOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend=torch_ttnn.backend, options=option)
        result_after = m.forward(input, weight)
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        self.assertTrue(nodes[6].target == ttnn.embedding)
        self.assertTrue(nodes[6].args[0].target == ttnn.to_device)
        self.assertTrue(nodes[6].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[6].args[1].target == ttnn.to_device)
        self.assertTrue(nodes[6].args[1].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[7].target == ttnn.from_device)
        self.assertTrue(nodes[8].target == ttnn.to_torch)
        # Check inference result
        self.assertTrue(torch.allclose(result_before, result_after))

    def test_embedding_tile_layout(self):
        m = EmbeddingTileLayoutModule()
        input_shapes = m.input_shapes()
        input = torch.randint(*input_shapes[0])
        weights = torch.zeros(*input_shapes[1], dtype=torch.bfloat16).uniform_(
            -0.1, 0.1
        )
        result_before = m.forward(input, weights)
        option = torch_ttnn.TorchTtnnOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend=torch_ttnn.backend, options=option)
        result_after = m.forward(input, weights)
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        self.assertTrue(nodes[6].target == ttnn.embedding)
        self.assertTrue(nodes[6].args[0].target == ttnn.to_device)
        self.assertTrue(nodes[6].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[6].args[1].target == ttnn.to_device)
        self.assertTrue(nodes[6].args[1].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[7].target == ttnn.from_device)
        self.assertTrue(nodes[8].target == ttnn.to_torch)
        # Check inference result
        self.assertTrue(torch.allclose(result_before, result_after))

    def test_clone_from_arg(self):
        m = CloneFromArgModule()
        input_shapes = m.input_shapes()
        inputs = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
        result_before = m.forward(*inputs)
        option = torch_ttnn.TorchTtnnOption(device=self.device)
        option.gen_graphviz = False
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend=torch_ttnn.backend, options=option)
        result_after = m.forward(*inputs)
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        self.assertTrue(nodes[5].target == ttnn.clone)
        self.assertTrue(nodes[5].args[0].target == ttnn.to_device)
        self.assertTrue(nodes[5].args[0].args[0].target == ttnn.to_layout)
        self.assertTrue(nodes[5].args[0].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[5].args[1].target == ttnn.MemoryConfig)
        self.assertTrue(nodes[6].target == ttnn.from_device)
        self.assertTrue(nodes[7].target == ttnn.to_layout)
        self.assertTrue(nodes[8].target == ttnn.to_torch)
        # Check inference result
        self.assertTrue(torch.allclose(result_before, result_after))

    def test_clone_from_node(self):
        m = CloneFromNodeModule()
        input_shapes = m.input_shapes()
        inputs = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
        result_before = m.forward(*inputs)
        option = torch_ttnn.TorchTtnnOption(device=self.device)
        option.gen_graphviz = False
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend=torch_ttnn.backend, options=option)
        result_after = m.forward(*inputs)
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        self.assertTrue(nodes[6].target == ttnn.clone)
        self.assertTrue(nodes[6].args[0].target == ttnn.add)
        self.assertTrue(nodes[6].args[1].target == ttnn.MemoryConfig)
        self.assertTrue(nodes[7].target == ttnn.from_device)
        self.assertTrue(nodes[8].target == ttnn.to_layout)
        self.assertTrue(nodes[9].target == ttnn.to_torch)
        # Check inference result
        self.assertTrue(torch.allclose(result_before, result_after))

    def test_layer_norm(self):
        m = LayerNormModule()
        input_shapes = m.input_shapes()
        inputs = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
        result_before = m.forward(*inputs)
        option = torch_ttnn.TorchTtnnOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend=torch_ttnn.backend, options=option)
        result_after = m.forward(*inputs)
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        self.assertTrue(nodes[12].target == ttnn.layer_norm)
        self.assertTrue(nodes[12].args[0].target == ttnn.to_device)
        self.assertTrue(nodes[12].args[0].args[0].target == ttnn.to_layout)
        self.assertTrue(nodes[12].args[0].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[12].kwargs["weight"].target == ttnn.to_device)
        self.assertTrue(nodes[12].kwargs["weight"].args[0].target == ttnn.to_layout)
        self.assertTrue(
            nodes[12].kwargs["weight"].args[0].args[0].target == ttnn.from_torch
        )
        self.assertTrue(nodes[12].kwargs["bias"].target == ttnn.to_device)
        self.assertTrue(nodes[12].kwargs["bias"].args[0].target == ttnn.to_layout)
        self.assertTrue(
            nodes[12].kwargs["bias"].args[0].args[0].target == ttnn.from_torch
        )
        self.assertTrue(nodes[13].target == ttnn.from_device)
        self.assertTrue(nodes[14].target == ttnn.to_layout)
        self.assertTrue(nodes[15].target == ttnn.to_torch)
        # Check inference result
        self.assertTrue(check_with_pcc(result_before, result_after, 0.9998))

    def test_layer_norm_with_other_op(self):
        m = LayerNormWithOtherOpModule()
        input_shapes = m.input_shapes()
        inputs = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
        result_before = m.forward(*inputs)
        option = torch_ttnn.TorchTtnnOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend=torch_ttnn.backend, options=option)
        result_after = m.forward(*inputs)
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        # self.assertTrue(nodes[12].target == ttnn.layer_norm)
        # self.assertTrue(nodes[12].args[0].target == ttnn.to_device)
        # self.assertTrue(nodes[12].args[0].args[0].target == ttnn.to_layout)
        # self.assertTrue(nodes[12].args[0].args[0].args[0].target == ttnn.from_torch)
        # self.assertTrue(nodes[12].kwargs["weight"].target == ttnn.to_device)
        # self.assertTrue(nodes[12].kwargs["weight"].args[0].target == ttnn.to_layout)
        # self.assertTrue(nodes[12].kwargs["weight"].args[0].args[0].target == ttnn.from_torch)
        # self.assertTrue(nodes[12].kwargs["bias"].target == ttnn.to_device)
        # self.assertTrue(nodes[12].kwargs["bias"].args[0].target == ttnn.to_layout)
        # self.assertTrue(nodes[12].kwargs["bias"].args[0].args[0].target == ttnn.from_torch)
        # self.assertTrue(nodes[13].target == ttnn.to_layout)
        # self.assertTrue(nodes[14].target == ttnn.from_device)
        # self.assertTrue(nodes[15].target == ttnn.to_torch)
        # Check inference result
        self.assertTrue(check_with_pcc(result_before, result_after, 0.9998))

    def test_neg(self):
        m = NegModule()
        input_shapes = m.input_shapes()
        inputs = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
        result_before = m.forward(*inputs)
        option = torch_ttnn.TorchTtnnOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend=torch_ttnn.backend, options=option)
        result_after = m.forward(*inputs)
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        self.assertTrue(nodes[4].target == ttnn.neg)
        self.assertTrue(nodes[4].args[0].target == ttnn.to_device)
        self.assertTrue(nodes[4].args[0].args[0].target == ttnn.to_layout)
        self.assertTrue(nodes[4].args[0].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[5].target == ttnn.from_device)
        self.assertTrue(nodes[6].target == ttnn.to_layout)
        self.assertTrue(nodes[7].target == ttnn.to_torch)
        # Check inference result
        self.assertTrue(torch.allclose(result_before, result_after))

    def test_ones(self):
        m = OnesModule()
        input_shapes = m.input_shapes()[0]
        result_before = m.forward(input_shapes)
        result_before = result_before.to(torch.bfloat16)
        option = torch_ttnn.TorchTtnnOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend=torch_ttnn.backend, options=option)
        result_after = m.forward(input_shapes)
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        self.assertTrue(nodes[0].target == ttnn.ones)
        self.assertTrue(nodes[1].target == ttnn.from_device)
        self.assertTrue(nodes[2].target == ttnn.to_layout)
        self.assertTrue(nodes[3].target == ttnn.to_torch)
        # Check inference result
        self.assertTrue(torch.allclose(result_before, result_after))

    def test_tril(self):
        m = TrilModule()
        input_shapes = m.input_shapes()
        inputs = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
        result_before = m.forward(*inputs)
        option = torch_ttnn.TorchTtnnOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend=torch_ttnn.backend, options=option)
        result_after = m.forward(*inputs)
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        self.assertTrue(nodes[4].target == ttnn.tril)
        self.assertTrue(nodes[4].args[0].target == ttnn.to_device)
        self.assertTrue(nodes[4].args[0].args[0].target == ttnn.to_layout)
        self.assertTrue(nodes[4].args[0].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[5].target == ttnn.from_device)
        self.assertTrue(nodes[6].target == ttnn.to_layout)
        self.assertTrue(nodes[7].target == ttnn.to_torch)
        # Check inference result
        self.assertTrue(torch.allclose(result_before, result_after))

    @unittest.skip(
        "NOTE(kevinwuTT) This test fails because ttnn.arange does not support start value of 0."
    )
    def test_arange(self):
        m = ArangeModule()
        input_shapes = m.input_shapes()
        # inputs = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
        result_before = m.forward(*input_shapes).to(torch.bfloat16)
        option = torch_ttnn.TorchTtnnOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend=torch_ttnn.backend, options=option)
        result_after = m.forward(*input_shapes)
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        self.assertTrue(nodes[0].target == ttnn.arange)
        self.assertTrue(nodes[1].target == ttnn.from_device)
        self.assertTrue(nodes[2].target == ttnn.to_layout)
        self.assertTrue(nodes[3].target == ttnn.to_torch)
        # Check inference result
        self.assertTrue(torch.allclose(result_before, result_after))

    def test_arange_start(self):
        m = ArangeStartModule()
        input_shapes = m.input_shapes()
        # inputs = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
        result_before = m.forward(*input_shapes).to(torch.bfloat16)
        option = torch_ttnn.TorchTtnnOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend=torch_ttnn.backend, options=option)
        result_after = m.forward(*input_shapes)
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        self.assertTrue(nodes[0].target == ttnn.arange)
        self.assertTrue(nodes[1].target == ttnn.from_device)
        self.assertTrue(nodes[2].target == ttnn.to_layout)
        self.assertTrue(nodes[3].target == ttnn.to_torch)
        # Check inference result
        self.assertTrue(torch.allclose(result_before, result_after))

    def test_arange_start_step(self):
        m = ArangeStartStepModule()
        input_shapes = m.input_shapes()
        result_before = m.forward(*input_shapes).to(torch.bfloat16)
        option = torch_ttnn.TorchTtnnOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend=torch_ttnn.backend, options=option)
        result_after = m.forward(*input_shapes)
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        self.assertTrue(nodes[0].target == ttnn.arange)
        self.assertTrue(nodes[1].target == ttnn.from_device)
        self.assertTrue(nodes[2].target == ttnn.to_layout)
        self.assertTrue(nodes[3].target == ttnn.to_torch)
        # Check inference result
        self.assertTrue(torch.allclose(result_before, result_after))

    def test_eq_tensor(self):
        m = EqTensorModule()
        input_shapes = m.input_shapes()
        inputs = [
            torch.randint(0, 2, shape, dtype=torch.bfloat16) for shape in input_shapes
        ]
        result_before = m.forward(*inputs)
        option = torch_ttnn.TorchTtnnOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend=torch_ttnn.backend, options=option)
        result_after = m.forward(*inputs)
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        self.assertTrue(nodes[8].target == ttnn.eq)
        self.assertTrue(nodes[8].args[0].target == ttnn.to_device)
        self.assertTrue(nodes[8].args[0].args[0].target == ttnn.to_layout)
        self.assertTrue(nodes[8].args[0].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[8].args[1].target == ttnn.to_device)
        self.assertTrue(nodes[8].args[1].args[0].target == ttnn.to_layout)
        self.assertTrue(nodes[8].args[1].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[9].target == ttnn.from_device)
        self.assertTrue(nodes[10].target == ttnn.to_layout)
        self.assertTrue(nodes[11].target == ttnn.to_torch)
        # Check inference result
        self.assertTrue(torch.allclose(result_before, result_after.to(torch.bool)))

    def test_eq_scalar(self):
        m = EqScalarModule()
        input_shapes = m.input_shapes()
        inputs = [
            torch.randint(0, 2, shape, dtype=torch.bfloat16) for shape in input_shapes
        ]
        scalar = torch.randint(0, 2, (1,)).item()
        result_before = m.forward(inputs[0], scalar)
        option = torch_ttnn.TorchTtnnOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend=torch_ttnn.backend, options=option)
        result_after = m.forward(inputs[0], scalar)
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        self.assertTrue(nodes[1].target == ttnn.full)
        self.assertTrue(nodes[5].target == ttnn.eq)
        self.assertTrue(nodes[5].args[0].target == ttnn.to_device)
        self.assertTrue(nodes[5].args[0].args[0].target == ttnn.to_layout)
        self.assertTrue(nodes[5].args[0].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[6].target == ttnn.from_device)
        self.assertTrue(nodes[7].target == ttnn.to_layout)
        self.assertTrue(nodes[8].target == ttnn.to_torch)
        # Check inference result
        self.assertTrue(torch.allclose(result_before, result_after.to(torch.bool)))

    def test_logical_not(self):
        m = LogicalNotModule()
        input_shapes = m.input_shapes()
        inputs = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
        result_before = m.forward(*inputs)
        option = torch_ttnn.TorchTtnnOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend=torch_ttnn.backend, options=option)
        result_after = m.forward(*inputs)
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        self.assertTrue(nodes[4].target == ttnn.logical_not)
        self.assertTrue(nodes[4].args[0].target == ttnn.to_device)
        self.assertTrue(nodes[4].args[0].args[0].target == ttnn.to_layout)
        self.assertTrue(nodes[4].args[0].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[5].target == ttnn.from_device)
        self.assertTrue(nodes[6].target == ttnn.to_layout)
        self.assertTrue(nodes[7].target == ttnn.to_torch)
        # Check inference result
        self.assertTrue(torch.allclose(result_before, result_after.to(torch.bool)))

    def test_zeros_like(self):
        m = ZerosLikeModule()
        input_shapes = m.input_shapes()
        inputs = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
        result_before = m.forward(*inputs)
        option = torch_ttnn.TorchTtnnOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend=torch_ttnn.backend, options=option)
        result_after = m.forward(*inputs)
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        self.assertTrue(nodes[3].target == ttnn.zeros_like)
        self.assertTrue(nodes[3].args[0].target == ttnn.to_device)
        self.assertTrue(nodes[3].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[4].target == ttnn.from_device)
        self.assertTrue(nodes[5].target == ttnn.to_torch)
        # Check inference result
        self.assertTrue(torch.allclose(result_before, result_after))

    def test_mean_dim(self):
        m = MeanDimModule()
        input_shapes = m.input_shapes()
        input = torch.zeros(input_shapes[0], dtype=torch.bfloat16).uniform_(-1, 1)
        dim = input_shapes[1]
        keepdim = True
        result_before = m.forward(input, dim, keepdim)
        option = torch_ttnn.TorchTtnnOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend=torch_ttnn.backend, options=option)
        result_after = m.forward(input, dim, keepdim)
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        self.assertTrue(nodes[4].target == ttnn.mean)
        self.assertTrue(nodes[4].args[0].target == ttnn.to_device)
        self.assertTrue(nodes[4].args[0].args[0].target == ttnn.to_layout)
        self.assertTrue(nodes[4].args[0].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[5].target == ttnn.from_device)
        self.assertTrue(nodes[6].target == ttnn.to_layout)
        self.assertTrue(nodes[7].target == ttnn.to_torch)
        # Check inference result
        self.assertTrue(check_with_pcc(result_before, result_after))

    def test_pow_tensor_scalar(self):
        m = PowTensorScalarModule()
        input_shapes = m.input_shapes()
        inputs = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
        result_before = m.forward(*inputs)
        option = torch_ttnn.TorchTtnnOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend=torch_ttnn.backend, options=option)
        result_after = m.forward(*inputs)
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        self.assertTrue(nodes[4].target == ttnn.pow)
        self.assertTrue(nodes[4].args[0].target == ttnn.to_device)
        self.assertTrue(nodes[4].args[0].args[0].target == ttnn.to_layout)
        self.assertTrue(nodes[4].args[0].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[5].target == ttnn.from_device)
        self.assertTrue(nodes[6].target == ttnn.to_layout)
        self.assertTrue(nodes[7].target == ttnn.to_torch)
        # Check inference result
        self.assertTrue(check_with_pcc(result_before, result_after))

    def test_rsqrt(self):
        m = RsqrtModule()
        input_shapes = m.input_shapes()
        inputs = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
        result_before = m.forward(*inputs)
        option = torch_ttnn.TorchTtnnOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend=torch_ttnn.backend, options=option)
        result_after = m.forward(*inputs)
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        self.assertTrue(nodes[4].target == ttnn.rsqrt)
        self.assertTrue(nodes[4].args[0].target == ttnn.to_device)
        self.assertTrue(nodes[4].args[0].args[0].target == ttnn.to_layout)
        self.assertTrue(nodes[4].args[0].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[5].target == ttnn.from_device)
        self.assertTrue(nodes[6].target == ttnn.to_layout)
        self.assertTrue(nodes[7].target == ttnn.to_torch)
        # Check inference result
        self.assertTrue(check_with_pcc(result_before, result_after))

    def test_silu(self):
        m = SiluModule()
        input_shapes = m.input_shapes()
        inputs = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
        result_before = m.forward(*inputs)
        option = torch_ttnn.TorchTtnnOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend=torch_ttnn.backend, options=option)
        result_after = m.forward(*inputs)
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        self.assertTrue(nodes[4].target == ttnn.silu)
        self.assertTrue(nodes[4].args[0].target == ttnn.to_device)
        self.assertTrue(nodes[4].args[0].args[0].target == ttnn.to_layout)
        self.assertTrue(nodes[4].args[0].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[5].target == ttnn.from_device)
        self.assertTrue(nodes[6].target == ttnn.to_layout)
        self.assertTrue(nodes[7].target == ttnn.to_torch)
        # Check inference result
        self.assertTrue(check_with_pcc(result_before, result_after))

    def test_adaptive_avg_pool_2d(self):
        m = AdaptiveAvgPool2dModule()
        input_shapes = m.input_shapes()
        inputs = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
        result_before = m.forward(*inputs)
        option = torch_ttnn.TorchTtnnOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend=torch_ttnn.backend, options=option)
        result_after = m.forward(*inputs)
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        self.assertTrue(nodes[4].target == ttnn.global_avg_pool2d)
        self.assertTrue(nodes[4].args[0].target == ttnn.to_device)
        self.assertTrue(nodes[4].args[0].args[0].target == ttnn.to_layout)
        self.assertTrue(nodes[4].args[0].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[5].target == ttnn.from_device)
        self.assertTrue(nodes[6].target == ttnn.to_layout)
        self.assertTrue(nodes[7].target == ttnn.to_torch)
        # Check inference result
        self.assertTrue(check_with_pcc(result_before, result_after))

    def test_clamp(self):
        m = ClampModule()
        input_shapes = m.input_shapes()
        inputs = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
        min, max = -0.5, 0.5
        result_before = m.forward(inputs[0], min, max)
        option = torch_ttnn.TorchTtnnOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend=torch_ttnn.backend, options=option)
        result_after = m.forward(inputs[0], min, max)
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        self.assertTrue(nodes[4].target == ttnn.clip)
        self.assertTrue(nodes[4].args[0].target == ttnn.to_device)
        self.assertTrue(nodes[4].args[0].args[0].target == ttnn.to_layout)
        self.assertTrue(nodes[4].args[0].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[5].target == ttnn.from_device)
        self.assertTrue(nodes[6].target == ttnn.to_layout)
        self.assertTrue(nodes[7].target == ttnn.to_torch)
        # Check inference result
        self.assertTrue(check_with_pcc(result_before, result_after))

    def test_squeeze_dim(self):
        m = SqueezeDimModule()
        input_shapes = m.input_shapes()
        inputs = [torch.zeros(shape, dtype=torch.bfloat16) for shape in input_shapes]
        dim = 0
        result_before = m.forward(inputs[0], dim)
        option = torch_ttnn.TorchTtnnOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend=torch_ttnn.backend, options=option)
        result_after = m.forward(inputs[0], dim)
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        self.assertTrue(nodes[4].target == ttnn.squeeze)
        self.assertTrue(nodes[4].args[0].target == ttnn.to_device)
        self.assertTrue(nodes[4].args[0].args[0].target == ttnn.to_layout)
        self.assertTrue(nodes[4].args[0].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[5].target == ttnn.from_device)
        self.assertTrue(nodes[6].target == ttnn.to_layout)
        self.assertTrue(nodes[7].target == ttnn.to_torch)
        # Check inference result
        self.assertTrue(torch.allclose(result_before, result_after))

    def test_full(self):
        m = FullModule()
        input_shapes = m.input_shapes()
        fill_value = 1.23
        result_before = m.forward(input_shapes[0], fill_value).to(torch.bfloat16)
        option = torch_ttnn.TorchTtnnOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend=torch_ttnn.backend, options=option)
        result_after = m.forward(input_shapes[0], fill_value)
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        self.assertTrue(nodes[0].target == ttnn.full)
        self.assertTrue(nodes[1].target == ttnn.from_device)
        self.assertTrue(nodes[2].target == ttnn.to_layout)
        self.assertTrue(nodes[3].target == ttnn.to_torch)
        # Check inference result
        self.assertTrue(torch.allclose(result_before, result_after))

    def test_lt_tensor(self):
        m = LtTensorModule()
        input_shapes = m.input_shapes()
        inputs = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
        result_before = m.forward(*inputs)
        option = torch_ttnn.TorchTtnnOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend=torch_ttnn.backend, options=option)
        result_after = m.forward(*inputs)
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        self.assertTrue(nodes[8].target == ttnn.lt)
        self.assertTrue(nodes[8].args[0].target == ttnn.to_device)
        self.assertTrue(nodes[8].args[0].args[0].target == ttnn.to_layout)
        self.assertTrue(nodes[8].args[0].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[8].args[1].target == ttnn.to_device)
        self.assertTrue(nodes[8].args[1].args[0].target == ttnn.to_layout)
        self.assertTrue(nodes[8].args[1].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[9].target == ttnn.from_device)
        self.assertTrue(nodes[10].target == ttnn.to_layout)
        self.assertTrue(nodes[11].target == ttnn.to_torch)
        # Check inference result
        self.assertTrue(torch.allclose(result_before, result_after.to(torch.bool)))

    def test_lt_scalar(self):
        m = LtScalarModule()
        input_shapes = m.input_shapes()
        inputs = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
        scalar = inputs[0][0][0].item()
        result_before = m.forward(inputs[0], scalar)
        option = torch_ttnn.TorchTtnnOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend=torch_ttnn.backend, options=option)
        result_after = m.forward(inputs[0], scalar)
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        self.assertTrue(nodes[1].target == ttnn.full)
        self.assertTrue(nodes[5].target == ttnn.lt)
        self.assertTrue(nodes[5].args[0].target == ttnn.to_device)
        self.assertTrue(nodes[5].args[0].args[0].target == ttnn.to_layout)
        self.assertTrue(nodes[5].args[0].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[6].target == ttnn.from_device)
        self.assertTrue(nodes[7].target == ttnn.to_layout)
        self.assertTrue(nodes[8].target == ttnn.to_torch)
        # Check inference result
        self.assertTrue(check_with_pcc(result_before, result_after))

    def test_baddbmm(self):
        m = BaddbmmModule()
        input_shapes = m.input_shapes()
        inputs = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
        option = torch_ttnn.TorchTtnnOption(device=self.device)
        option.gen_graphviz = True

        # (1) Test with default alpha and beta values
        result_before = m.forward(*inputs)
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m_ttnn = torch.compile(m, backend=torch_ttnn.backend, options=option)
        result_after = m_ttnn.forward(*inputs)
        option._out_fx_graphs[-1].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[-1].nodes)
        self.assertTrue(nodes[9].target == ttnn.matmul)
        self.assertTrue(nodes[9].args[0].target == ttnn.to_device)
        self.assertTrue(nodes[9].args[0].args[0].target == ttnn.to_layout)
        self.assertTrue(nodes[9].args[0].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[9].args[1].target == ttnn.to_device)
        self.assertTrue(nodes[9].args[1].args[0].target == ttnn.to_layout)
        self.assertTrue(nodes[9].args[1].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[13].target == ttnn.add)
        self.assertTrue(nodes[13].args[0].target == ttnn.to_device)
        self.assertTrue(nodes[13].args[0].args[0].target == ttnn.to_layout)
        self.assertTrue(nodes[13].args[0].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[13].args[1].target == ttnn.matmul)
        self.assertTrue(nodes[14].target == ttnn.from_device)
        self.assertTrue(nodes[15].target == ttnn.to_layout)
        self.assertTrue(nodes[16].target == ttnn.to_torch)
        # Check inference result
        self.assertTrue(check_with_pcc(result_before, result_after))

        # (2) Test with alpha and default beta value
        result_before = m.forward(*inputs, alpha=2)
        m_ttnn = torch.compile(m, backend=torch_ttnn.backend, options=option)
        result_after = m_ttnn.forward(*inputs, alpha=2)
        option._out_fx_graphs[-1].print_tabular()

        nodes = list(option._out_fx_graphs[-1].nodes)
        self.assertTrue(nodes[9].target == ttnn.matmul)
        self.assertTrue(nodes[9].args[0].target == ttnn.to_device)
        self.assertTrue(nodes[9].args[0].args[0].target == ttnn.to_layout)
        self.assertTrue(nodes[9].args[0].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[9].args[1].target == ttnn.to_device)
        self.assertTrue(nodes[9].args[1].args[0].target == ttnn.to_layout)
        self.assertTrue(nodes[9].args[1].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[10].target == ttnn.multiply)
        self.assertTrue(nodes[10].args[0].target == ttnn.matmul)
        self.assertTrue(nodes[14].target == ttnn.add)
        self.assertTrue(nodes[14].args[0].target == ttnn.to_device)
        self.assertTrue(nodes[14].args[0].args[0].target == ttnn.to_layout)
        self.assertTrue(nodes[14].args[0].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[14].args[1].target == ttnn.multiply)
        self.assertTrue(nodes[15].target == ttnn.from_device)
        self.assertTrue(nodes[16].target == ttnn.to_layout)
        self.assertTrue(nodes[17].target == ttnn.to_torch)
        self.assertTrue(check_with_pcc(result_before, result_after))

        # (3) Test with beta and default alpha value
        result_before = m.forward(*inputs, beta=2)
        m_ttnn = torch.compile(m, backend=torch_ttnn.backend, options=option)
        result_after = m_ttnn.forward(*inputs, beta=2)
        option._out_fx_graphs[-1].print_tabular()

        nodes = list(option._out_fx_graphs[-1].nodes)
        self.assertTrue(nodes[9].target == ttnn.matmul)
        self.assertTrue(nodes[9].args[0].target == ttnn.to_device)
        self.assertTrue(nodes[9].args[0].args[0].target == ttnn.to_layout)
        self.assertTrue(nodes[9].args[0].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[9].args[1].target == ttnn.to_device)
        self.assertTrue(nodes[9].args[1].args[0].target == ttnn.to_layout)
        self.assertTrue(nodes[9].args[1].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[13].target == ttnn.multiply)
        self.assertTrue(nodes[13].args[0].target == ttnn.to_device)
        self.assertTrue(nodes[13].args[0].args[0].target == ttnn.to_layout)
        self.assertTrue(nodes[13].args[0].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[14].target == ttnn.add)
        self.assertTrue(nodes[14].args[0].target == ttnn.multiply)
        self.assertTrue(nodes[14].args[1].target == ttnn.matmul)
        self.assertTrue(nodes[15].target == ttnn.from_device)
        self.assertTrue(nodes[16].target == ttnn.to_layout)
        self.assertTrue(nodes[17].target == ttnn.to_torch)
        self.assertTrue(check_with_pcc(result_before, result_after))

        # (4) Test with beta and alpha values
        result_before = m.forward(*inputs, beta=2, alpha=2)
        m_ttnn = torch.compile(m, backend=torch_ttnn.backend, options=option)
        result_after = m_ttnn.forward(*inputs, beta=2, alpha=2)
        option._out_fx_graphs[-1].print_tabular()

        nodes = list(option._out_fx_graphs[-1].nodes)
        self.assertTrue(nodes[9].target == ttnn.matmul)
        self.assertTrue(nodes[9].args[0].target == ttnn.to_device)
        self.assertTrue(nodes[9].args[0].args[0].target == ttnn.to_layout)
        self.assertTrue(nodes[9].args[0].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[9].args[1].target == ttnn.to_device)
        self.assertTrue(nodes[9].args[1].args[0].target == ttnn.to_layout)
        self.assertTrue(nodes[9].args[1].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[10].target == ttnn.multiply)
        self.assertTrue(nodes[10].args[0].target == ttnn.matmul)
        self.assertTrue(nodes[14].target == ttnn.multiply)
        self.assertTrue(nodes[14].args[0].target == ttnn.to_device)
        self.assertTrue(nodes[14].args[0].args[0].target == ttnn.to_layout)
        self.assertTrue(nodes[14].args[0].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[15].target == ttnn.add)
        self.assertTrue(nodes[15].args[0].target == ttnn.multiply)
        self.assertTrue(nodes[15].args[1].target == ttnn.multiply)
        self.assertTrue(nodes[16].target == ttnn.from_device)
        self.assertTrue(nodes[17].target == ttnn.to_layout)
        self.assertTrue(nodes[18].target == ttnn.to_torch)
        self.assertTrue(check_with_pcc(result_before, result_after))

        # (5) Test special case when beta is 0
        result_before = m.forward(*inputs, beta=0, alpha=2)
        m_ttnn = torch.compile(m, backend=torch_ttnn.backend, options=option)
        result_after = m_ttnn.forward(*inputs, beta=0, alpha=2)
        option._out_fx_graphs[-1].print_tabular()

        nodes = list(option._out_fx_graphs[-1].nodes)
        self.assertTrue(nodes[9].target == ttnn.matmul)
        self.assertTrue(nodes[9].args[0].target == ttnn.to_device)
        self.assertTrue(nodes[9].args[0].args[0].target == ttnn.to_layout)
        self.assertTrue(nodes[9].args[0].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[9].args[1].target == ttnn.to_device)
        self.assertTrue(nodes[9].args[1].args[0].target == ttnn.to_layout)
        self.assertTrue(nodes[9].args[1].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[10].target == ttnn.multiply)
        self.assertTrue(nodes[10].args[0].target == ttnn.matmul)
        self.assertTrue(nodes[11].target == ttnn.from_device)
        self.assertTrue(nodes[12].target == ttnn.to_layout)
        self.assertTrue(nodes[13].target == ttnn.to_torch)
        self.assertTrue(check_with_pcc(result_before, result_after))

    def test_cos(self):
        m = CosModule()
        input_shapes = m.input_shapes()
        inputs = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
        result_before = m.forward(*inputs)
        option = torch_ttnn.TorchTtnnOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend=torch_ttnn.backend, options=option)
        result_after = m.forward(*inputs)
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        self.assertTrue(nodes[4].target == ttnn.cos)
        self.assertTrue(nodes[4].args[0].target == ttnn.to_device)
        self.assertTrue(nodes[4].args[0].args[0].target == ttnn.to_layout)
        self.assertTrue(nodes[4].args[0].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[5].target == ttnn.from_device)
        self.assertTrue(nodes[6].target == ttnn.to_layout)
        self.assertTrue(nodes[7].target == ttnn.to_torch)
        # Check inference result
        self.assertTrue(torch.allclose(result_before, result_after, rtol=0.2))

    def test_sigmoid(self):
        m = SigmoidModule()
        input_shapes = m.input_shapes()
        inputs = [torch.rand(shape, dtype=torch.bfloat16) for shape in input_shapes]
        result_before = m.forward(*inputs)
        option = torch_ttnn.TorchTtnnOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend=torch_ttnn.backend, options=option)
        result_after = m.forward(*inputs)
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        self.assertTrue(nodes[4].target == ttnn.sigmoid)
        self.assertTrue(nodes[4].args[0].target == ttnn.to_device)
        self.assertTrue(nodes[4].args[0].args[0].target == ttnn.to_layout)
        self.assertTrue(nodes[4].args[0].args[0].args[0].target == ttnn.from_torch)
        self.assertTrue(nodes[5].target == ttnn.from_device)
        self.assertTrue(nodes[6].target == ttnn.to_layout)
        self.assertTrue(nodes[7].target == ttnn.to_torch)
        # Check inference result
        self.assertTrue(torch.allclose(result_before, result_after, rtol=0.2))

    def test_to_copy(self):
        m = ToCopyModule()
        input_shapes = m.input_shapes()
        inputs = [torch.rand(shape) for shape in input_shapes]
        result_before = m.forward(*inputs)
        option = torch_ttnn.TorchTtnnOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend=torch_ttnn.backend, options=option)
        result_after = m.forward(*inputs)
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        self.assertTrue(nodes[2].target == ttnn.as_tensor)
        self.assertTrue(nodes[3].target == ttnn.from_device)
        self.assertTrue(nodes[4].target == ttnn.to_layout)
        self.assertTrue(nodes[5].target == ttnn.to_torch)
        # Check inference result
        self.assertTrue(torch.allclose(result_before, result_after, rtol=0.2))

    def test_to_copy_with_op_after(self):
        m = ToCopyWithOpAfterModule()
        input_shapes = m.input_shapes()
        inputs = [torch.rand(shape) for shape in input_shapes]
        result_before = m.forward(*inputs)
        option = torch_ttnn.TorchTtnnOption(device=self.device)
        option.gen_graphviz = True
        # The compilation is lazy, so we need to run forward once to trigger the compilation
        m = torch.compile(m, backend=torch_ttnn.backend, options=option)
        result_after = m.forward(*inputs)
        option._out_fx_graphs[0].print_tabular()

        # Check the graph has be rewritten and contain ttnn ops
        nodes = list(option._out_fx_graphs[0].nodes)
        self.assertTrue(nodes[2].target == ttnn.as_tensor)
        self.assertTrue(nodes[3].target == ttnn.add)
        self.assertTrue(nodes[3].args[0].target == ttnn.as_tensor)
        self.assertTrue(nodes[3].args[1].target == ttnn.as_tensor)
        self.assertTrue(nodes[4].target == ttnn.from_device)
        self.assertTrue(nodes[5].target == ttnn.to_layout)
        self.assertTrue(nodes[6].target == ttnn.to_torch)
        # Check inference result
        self.assertTrue(torch.allclose(result_before, result_after, rtol=0.2))


if __name__ == "__main__":
    unittest.main()
