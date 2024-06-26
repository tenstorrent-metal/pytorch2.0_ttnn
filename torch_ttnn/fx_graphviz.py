import graphviz
import torch
import ttnn
import os
import math
from collections import defaultdict


def _tensor_weight(t):
    s = t.shape
    if s is None:
        return 1

    result = 1
    for d in s:
        if d is None:
            continue
        else:
            result *= d

    return result


def _tensor_width(t):
    w = _tensor_weight(t)
    # Strange, some tensor has 0 element
    if w <= 0:
        w = 1
    l = math.log10(w) * 2
    return max(l, 4)


def _tensor_label(t):
    return str(t.shape)


def node_name(node):
    # FX graph node
    # The node name is uquie and can be used as a key
    if isinstance(node, torch.fx.node.Node):
        return node.name

    # Other Python Object, like "device" or constant value
    # TODO(yoco): not all device are the same, maybe we should use a different name
    return str(node) + hex(id(node))


def node_label(node):
    # If it is not a FX node, just return the string
    if not isinstance(node, torch.fx.node.Node):
        return str(node)

    # If it is a FX call function
    if node.op == "call_function":
        # A func wrapper with "_name" attribute
        if hasattr(node.target, "_name"):
            return node.target._name
        # Normal function: __module__.__name__
        return f"{node.target.__module__}.{node.target.__name__}"

    # call_module, call_method, output, and placeholder
    return str(node.op + "\n" + node.name)


def node_fillcolor(node):
    # Green for supported ops
    if node.target in set(
        [
            ttnn.add,
            ttnn.matmul,
            ttnn.sub,
            ttnn.mul,
            ttnn.softmax,
            ttnn.tanh,
            ttnn.reshape,
            ttnn.permute,
        ]
    ):
        return "#aaffaa"

    # Yellow for device ops
    if node.target in [ttnn.to_device, ttnn.from_device]:
        return "#ffffaa"

    # Orange for torch tensor conversion ops
    if node.target in [ttnn.from_torch, ttnn.to_torch]:
        return "#ffddaa"

    # Orange for torch tensor conversion ops
    if node.target in [ttnn.to_layout]:
        return "#ffeeaa"

    # Red for call_function (unknown ops)
    if node.op == "call_function":
        return "#ffaaaa"

    # Gray for others
    return "#dddddd"


def to_port(in_size, it_idx):
    port_table = [
        [],
        [""],
        [":nw", ":ne"],
        [":nw", ":n", ":ne"],
        [":w", ":nw", ":n", ":ne"],
        [":w", ":nw", ":n", ":ne", ":e"],
        [":nw", ":nw", ":n", ":n", ":ne", ":ne"],
    ]
    if in_size > 6:
        return ""
    result = port_table[in_size][it_idx]
    return result


def to_svg(g: torch.fx.Graph, filename: str):
    # Setup dot
    dot = graphviz.Digraph()
    dot.graph_attr["ranksep"] = "0.4"
    dot.node_attr["style"] = "rounded,filled"
    dot.node_attr["height"] = "0.3"
    dot.node_attr["shape"] = "box"
    dot.node_attr["fillcolor"] = "#dddddd"
    dot.edge_attr["color"] = "#77777777"

    # setup nodes
    map_node_idx = dict()
    map_idx_node = dict()
    for op_idx, node in enumerate(g.nodes):
        map_node_idx[node] = op_idx
        map_idx_node[op_idx] = node
        dot.node(
            node_name(node), label=node_label(node), fillcolor=node_fillcolor(node)
        )

    # setup edges
    edge_color_table = [
        "#77ffff77",
        "#ff77ff77",
        "#7777ff77",
        "#ff777777",
        "#77ff7777",
    ]

    # A table of non-fx.node, like constant or other object that has been used.
    # For those objects, we will use a different name for each of them.
    var_obj_cnt = defaultdict(int)
    for node in g.nodes:
        # If the node is output, the "args" is a tuple like ((v1, v2),).
        # else, the "args" is a tuple like (v1, v2).
        # It is very stange, but we need to handle it.
        if node.op == "output":
            in_nodes = node.args[0]
        else:
            in_nodes = node.args

        for idx, in_node in enumerate(in_nodes):
            # For each input:
            # If it is a fx.node, use the node name.
            # else create a duplicated visual node for the input.
            if isinstance(in_node, torch.fx.node.Node):
                src_name = node_name(in_node)
            else:
                serial_num = var_obj_cnt[in_node]
                src_name = f"{node_name(in_node)}_{serial_num}"
                src_label = node_label(in_node)
                var_obj_cnt[in_node] += 1
                dot.node(src_name, label=src_label)

            # Draw the edge
            dst_name = node_name(node) + to_port(len(in_nodes), idx)
            edge_color = edge_color_table[(id(node) + idx) % 5]
            dot.edge(
                src_name,
                dst_name,
                color=edge_color,
                penwidth=str(4),
            )

    # Write .dot & .svg
    dot.save(f"{filename}.dot")
    os.system(f"dot -T svg -O {filename}.dot")
