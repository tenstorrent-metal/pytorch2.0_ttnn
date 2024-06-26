import os
import json
from typing import Any
import torch
from collections import Counter
from dataclasses import dataclass
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch.fx.passes.shape_prop import ShapeProp


@dataclass
class Tracer:
    functor: callable
    out_prefix: str
    out_folder: str = os.path.join(os.getcwd(), "stat")
    trace_orig: bool = True
    trace_modi: bool = False

    def __post_init__(self):
        self.counter = Counter()

    def __call__(self, gm, example_inputs):
        if self.trace_orig or self.trace_modi:
            os.makedirs(os.path.join(self.out_folder, "raw"), exist_ok=True)
        if self.trace_orig:
            out_path = os.path.join(
                self.out_folder,
                "raw",
                f"{self.out_prefix}_orig_{self.counter['orig']}.json",
            )
            out = self.parse_fx_graph(gm, example_inputs)
            with open(out_path, "w") as f:
                json.dump(out, f, indent=4)
            self.counter["orig"] += 1

        gm = self.functor(gm, example_inputs)

        if self.trace_modi:
            out_path = os.path.join(
                self.out_folder,
                "raw",
                f"{self.out_prefix}_modi_{self.counter['modi']}.json",
            )
            out = self.parse_fx_graph(gm, example_inputs)
            with open(out_path, "w") as f:
                json.dump(out, f, indent=4)
            self.counter["modi"] += 1

        return gm

    @staticmethod
    def parse_fx_graph(gm: torch.fx.GraphModule, example_inputs):
        try:
            FakeTensorProp(gm).propagate(*example_inputs)
            # ShapeProp(gm).propagate(*example_inputs)
            prop_success = True
        except Exception as e:
            print(e)
            print("FakeTensorProp propagate failed")
            prop_success = False

        def get_tensor_info(t):
            def no_symInt_in_list(the_list):
                return not any(
                    isinstance(element, torch.SymInt) for element in the_list
                )

            # Only record if the tensor is torch.Tensor
            # some shape is referenced by a variable, like [2, 256, s0, s1]
            if isinstance(t, torch.Tensor) and no_symInt_in_list(list(t.shape)):
                return {"shape": list(t.shape), "dtype": str(t.dtype)}
            else:
                return {}

        out = []
        for node in gm.graph.nodes:
            if node.op not in ["call_function", "call_method"]:
                continue
            node_info = {}
            # set node's op_type
            node_info["op_type"] = str(node.target)
            # skip set node's inputs/outpus if FakeTensorProp failed
            if prop_success == False:
                out.append(node_info)
                continue
            # set node's inputs info
            node_info["inputs"] = []
            inputs_info = node.args
            for input_info in inputs_info:
                input = {}
                if isinstance(input_info, torch.fx.node.Node):
                    input = get_tensor_info(input_info.meta["val"])
                node_info["inputs"].append(input)
            # set node's outputs info
            node_info["outputs"] = []
            outputs_info = node.meta["val"]
            if isinstance(outputs_info, torch.SymInt):
                continue
            elif isinstance(outputs_info, tuple) or isinstance(outputs_info, list):
                for output_info in outputs_info:
                    output = get_tensor_info(output_info)
                    node_info["outputs"].append(output)
            elif isinstance(outputs_info, torch.Tensor):
                output = get_tensor_info(outputs_info)
                node_info["outputs"].append(output)
            else:
                assert 0 and "unsupport outputs_info"
            out.append(node_info)
        return out
