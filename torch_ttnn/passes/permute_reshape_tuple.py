import torch
import ttnn

from torch.fx.passes.infra.pass_base import PassBase, PassResult


class PermuteReshapeTuple(PassBase):
    def call(self, gm: torch.fx.GraphModule):
        modified = False
        for node in gm.graph.nodes:
            # Only process ttnn.permute and ttnn.reshape
            if node.op != "call_function":
                continue
            if node.target not in [ttnn.permute, ttnn.reshape]:
                continue
            if isinstance(node.args[1], list):
                args = list(node.args)
                args[1] = tuple(node.args[1])
                args = tuple(args)
                node.args = args
                modified = True

        return PassResult(gm, modified)
