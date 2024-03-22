import torch

try:
    import ttnn
except ImportError:
    print("ttnn is not installed, use mock_ttnn instead")
    from .. import mock_ttnn as ttnn

from torch.fx.passes.infra.pass_base import PassBase, PassResult


class ReplaceMoreTt(torch.fx.Transformer):
    def call_function(self, target, args, kwargs):
        if target == torch.ops.aten.sub.Tensor:
            return super().call_function(ttnn.sub, args, kwargs)
        elif target == torch.ops.aten.mul.Tensor:
            return super().call_function(ttnn.mul, args, kwargs)
        elif target == torch.ops.aten._softmax.default:
            return super().call_function(ttnn.softmax, args[:2], kwargs)
        elif target == torch.ops.aten.tanh.default:
            return super().call_function(ttnn.tanh, args, kwargs)
        elif target == torch.ops.aten.view.default:
            return super().call_function(ttnn.reshape, args, kwargs)
        elif target == torch.ops.aten.permute.default:
            return super().call_function(ttnn.permute, args, kwargs)
        elif target == torch.ops.aten.relu.default:
            return super().call_function(ttnn.relu, args, kwargs)
        elif target == torch.ops.aten.addmm.default:
            # TODO(kevinwuMCW): include beta, alpha, and optional args
            mm = super().call_function(ttnn.matmul, (args[1], args[2]), kwargs)
            return super().call_function(ttnn.add, (args[0], mm), kwargs)
        elif target == torch.ops.aten.div.Tensor:
            recip = super().call_function(ttnn.reciprocal, (args[1],), kwargs)
            return super().call_function(ttnn.mul, (args[0], recip), kwargs)
        elif target == torch.ops.aten.bmm.default:
            return super().call_function(ttnn.matmul, args, kwargs)
        elif target == torch.ops.aten.gelu.default:
            return super().call_function(ttnn.gelu, args, kwargs)
        return super().call_function(target, args, kwargs)


class ToTtPass(PassBase):
    def call(self, gm: torch.fx.GraphModule):
        # NOTE(yoco): In our case, subgraph_rewriter actually
        # is not a best choice. We should use torch.fx.Transformer.
        # However, we use subgraph_rewriter for demonstration
        # and as a code stub. Because Transformer only support
        # 1-to-N replacement. For N-to-M replacement, we need
        # to use subgraph_rewriter.
        from ..patterns import add
        from ..patterns import mm

        # Patterns and replacements
        pat_rep_list = list()
        pat_rep_list += add.pat_rep_list
        pat_rep_list += mm.pat_rep_list

        # Replace patterns
        modified = False
        for pat, rep in pat_rep_list:
            replaced_pats = torch.fx.subgraph_rewriter.replace_pattern(gm, pat, rep)
            modified = modified or len(replaced_pats) > 0

        # Replace more patterns with torch.fx.Transformer
        gm = ReplaceMoreTt(gm).transform()

        return PassResult(gm, True)
