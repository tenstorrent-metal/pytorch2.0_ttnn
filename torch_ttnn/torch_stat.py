import torch.linalg
import torch
from typing import List
import torch._dynamo
import os
from collections import Counter
from functorch.compile import make_boxed_func

torch._dynamo.config.suppress_errors = False
torch._dynamo.config.verbose = True


# The backend for torch.compile that statistics the graph information.
# The "option" parameter is a dict that contains one key "torch_stat_option".
# The value of "torch_stat_option" is a TorchStatOption object.
# See document for detail.
def aten_backend(
    gm: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    options: dict,
    direction: str = "fw",
) -> torch.fx.GraphModule:
    """
    The backend for torch.compile that statistics the graph information.
    The graph is wrapped in torch._dynamo.backends.common.aot_autograd, which
    trace into low level ATen ops not only high level torch ops.
    """

    option: TorchStatOption = options["torch_stat_option"]
    assert option.model_name != "" and "should give the model_name"

    # Rewrite with ttnn ops, will insert redundant data movement
    from torch.fx.passes.infra.pass_manager import PassManager
    from .passes.stat_pass import StatPass

    stat_filename = os.path.join(
        option.out,
        "raw",
        f"{direction}_{option.model_name}_{option.counter['val']}.json",
    )
    os.makedirs(os.path.dirname(stat_filename), exist_ok=True)
    passes = [StatPass(filename=stat_filename, example_inputs=example_inputs)]
    if option.gen_graphviz:
        from .passes.graphviz_pass import GraphvizPass

        graphviz_filename = os.path.join(
            option.out,
            "graphviz",
            f"{direction}_{option.model_name}_{option.counter['val']}",
        )
        os.makedirs(os.path.dirname(graphviz_filename), exist_ok=True)
        passes.append(GraphvizPass(filename=graphviz_filename))

    pm = PassManager(passes=passes)
    gm, modified = pm(gm)

    gm.recompile()
    option.out_fx_graphs.append(gm.graph)
    option.counter["val"] += 1
    return make_boxed_func(gm)


from torch._dynamo.backends.common import aot_autograd
from functools import partial


# The option for torch_stat.backend
class TorchStatOption:
    def __init__(
        self,
        model_name: str = "",
        backward=False,
        out=os.path.join(os.getcwd(), "stat"),
        gen_graphviz=False,
    ):
        self.model_name = model_name
        self.backward = backward
        self.out = out
        self.gen_graphviz = gen_graphviz
        self.out_fx_graphs = []
        self.counter = Counter()


# The wrapper of aot_autograd that takes a TorchStatOption as options.
def backend(torch_stat_option: TorchStatOption):
    options = {"torch_stat_option": torch_stat_option}
    if torch_stat_option.backward:
        return aot_autograd(
            fw_compiler=partial(aten_backend, options=options, direction="fw"),
            bw_compiler=partial(aten_backend, options=options, direction="bw"),
        )
    else:
        return aot_autograd(
            fw_compiler=partial(aten_backend, options=options, direction="fw")
        )
