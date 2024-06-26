import torch
import ttnn

from torch.fx.passes.infra.pass_base import PassBase, PassResult


class ReplaceMoreTt(torch.fx.Transformer):
    def call_function(self, target, args, kwargs):
        if target == torch.ops.aten.add.Tensor:
            return super().call_function(ttnn.add, args, kwargs)
        elif target == torch.ops.aten.mm.default:
            return super().call_function(ttnn.matmul, args, kwargs)
        elif target == torch.ops.aten.sub.Tensor:
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
        return super().call_function(target, args, kwargs)


class ToTtPass(PassBase):
    def call(self, gm: torch.fx.GraphModule):
        # NOTE(yoco): In our case, subgraph_rewriter actually
        # is not a best choice. We should use torch.fx.Transformer.
        # However, we use subgraph_rewriter for demonstration
        # and as a code stub. Because Transformer only support
        # 1-to-N replacement. For N-to-M replacement, we need
        # to use subgraph_rewriter.
        # import patterns.add
        # import patterns.mm

        # Patterns and replacements
        # pat_rep_list = list()
        # pat_rep_list += patterns.add.pat_rep_list
        # pat_rep_list += patterns.mm.pat_rep_list

        # Replace patterns
        #  for pat, rep in pat_rep_list:
        #      replaced_pats = torch.fx.subgraph_rewriter.replace_pattern(gm, pat, rep)
        #      modified = modified or len(replaced_pats) > 0

        # Replace more patterns with torch.fx.Transformer
        gm = ReplaceMoreTt(gm).transform()

        return PassResult(gm, True)
