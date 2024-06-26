import torch.linalg
import torch
import ttnn
import torch._dynamo
from typing import List, Optional, Union, Mapping, Any

torch._dynamo.config.suppress_errors = False
torch._dynamo.config.verbose = True


# The option for torch_ttnn.backend
class TenstorrentBackendOption:
    def __init__(self, device: ttnn.Device, tracer_option = None):
        self.device = device
        self.gen_graphviz = False
        self._out_fx_graphs = list()
        self.tracer_option = tracer_option

# The backend for torch.compile that converts a graph to use ttnn.
# The "option" parameter is a TorchTtnnOption object
# See document for detail.
def aten_backend(
    gm: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    options: TenstorrentBackendOption = None,
) -> torch.fx.GraphModule:
    """
    The backend for torch.compile that converts a graph to use ttnn.
    The graph is wrapped in torch._dynamo.backends.common.aot_autograd, which
    trace into low level ATen ops not only high level torch ops.
    """

    option: TenstorrentBackendOption = options
    torch.fx.graph._register_custom_builtin("ttnn_Specified_Device", "", option.device)
    torch.fx.graph._register_custom_builtin(
        "ttnn_ROW_MAJOR_LAYOUT", "", ttnn.ROW_MAJOR_LAYOUT
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
    return gm


from torch._dynamo.backends.common import aot_autograd
from functools import partial

# The backend for torch.compile that converts a graph to use ttnn.
# The "option" parameter is a TorchTtnnOption object
# See document for detail.
# This function is a wrapper of aten_backend, and is used by torch.compile.
# This function is also registered as a backend for torch.compile.
def ttnn_backend(
    gm: torch.fx.GraphModule,
    example_inputs: List[torch.Tensor],
    options: TenstorrentBackendOption = None,
) -> torch.fx.GraphModule:
    tracer_option = options.tracer_option
    if tracer_option is not None:
        from ..tracer import Tracer
        out_prefix = f"fw_{tracer_option['model_name']}"
        out_folder = tracer_option["out_folder"]
        trace_orig = tracer_option["trace_orig"] if "trace_orig" in tracer_option else True
        trace_modi = tracer_option["trace_modi"] if "trace_modi" in tracer_option else False
        fw_compiler = Tracer(partial(aten_backend, options=options), out_prefix, out_folder, trace_orig, trace_modi)
        return aot_autograd(fw_compiler=fw_compiler)(
            gm, example_inputs
        )
    else:
        return aot_autograd(fw_compiler=partial(aten_backend, options=options))(
            gm, example_inputs
        )
