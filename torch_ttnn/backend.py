import torch.linalg
import torch
from typing import List
import torch._dynamo
from functorch.compile import make_boxed_func

torch._dynamo.config.suppress_errors = False
torch._dynamo.config.verbose = True

try:
    import ttnn
except ImportError:
    print("ttnn is not installed, use mock_ttnn instead")
    from . import mock_ttnn as ttnn


# The backend for torch.compile that converts a graph to use ttnn.
# The "option" parameter is a dict that contains one key "torch_ttnn_option".
# The value of "torch_ttnn_option" is a TorchTtnnOption object.
# See document for detail.
def aten_backend(
    gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor], options: dict
) -> torch.fx.GraphModule:
    """
    The backend for torch.compile that converts a graph to use ttnn.
    The graph is wrapped in torch._dynamo.backends.common.aot_autograd, which
    trace into low level ATen ops not only high level torch ops.
    """

    option: TorchTtnnOption = options["torch_ttnn_option"]
    torch.fx.graph._register_custom_builtin("ttnn_Specified_Device", "", option.device)
    torch.fx.graph._register_custom_builtin(
        "ttnn_ROW_MAJOR_LAYOUT", "", ttnn.ROW_MAJOR_LAYOUT
    )
    torch.fx.graph._register_custom_builtin(
        "ttnn_TILE_LAYOUT", "", ttnn.TILE_LAYOUT
    )

    # Rewrite with ttnn ops, will insert redundant data movement
    from torch.fx.passes.infra.pass_manager import PassManager
    from torch.fx.passes.dialect.common.cse_pass import CSEPass
    from .passes.to_tt_pass import ToTtPass
    from .passes.add_data_move_pass import AddDataMovePass
    from .passes.graphviz_pass import GraphvizPass
    from .passes.eliminate_data_move_pass import EliminateDataMovePass
    from .passes.permute_reshape_tuple import PermuteReshapeTuple

    passes = [
        ToTtPass(),
        AddDataMovePass(),
        EliminateDataMovePass(),
        CSEPass(),
        PermuteReshapeTuple(),
    ]

    # Add graphviz pass interleavly if needed
    if option.gen_graphviz:
        graphviz_filenames = [
            "00.origin",
            "01.to-tt",
            "02.add-data-move",
            "03.elimate-data-move",
            "04.cse",
            "05.permute-reshape-tuple",
        ]
        assert len(graphviz_filenames) == len(passes) + 1
        for idx in range(len(graphviz_filenames)):
            passes.insert(idx * 2, GraphvizPass(graphviz_filenames[idx]))

    pm = PassManager(passes=passes)
    gm, modified = pm(gm)

    gm.recompile()
    gm.graph.print_tabular()
    print(gm.code)
    option._out_fx_graphs.append(gm.graph)
    return make_boxed_func(gm)


from torch._dynamo.backends.common import aot_autograd
from functools import partial


# The option for torch_ttnn.backend
class TorchTtnnOption:
    def __init__(self, device: ttnn.Device):
        self.device = device
        self.gen_graphviz = False
        self._out_fx_graphs = list()


# The wrapper of aot_autograd that takes a TorchTtnnOption as options.
def backend(torch_ttnn_option: TorchTtnnOption):
    options = {"torch_ttnn_option": torch_ttnn_option}
    return aot_autograd(fw_compiler=partial(aten_backend, options=options))
