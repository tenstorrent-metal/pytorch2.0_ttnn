import torch
from .. import fx_graphviz

from torch.fx.passes.infra.pass_base import PassBase, PassResult


class GraphvizPass(PassBase):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename

    def call(self, gm: torch.fx.GraphModule):
        fx_graphviz.to_svg(gm.graph, self.filename)
        return PassResult(gm, False)
