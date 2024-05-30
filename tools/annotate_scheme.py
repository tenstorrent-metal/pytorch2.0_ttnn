import glob
import json
from typing import List, Dict
import collections
import torch
import os

torch._convolution


def load_op_scheme():
    """
    Scheme is a {opname: List[str]} dictionary.
    An op might have multiple schemes, because it has multiple signatures.
    """
    op_scheme = collections.defaultdict(list)
    with open("tools/op_scheme_doc.txt", "r") as fin:
        for line in fin.readlines():
            line = line.strip()
            opname_full, scheme = line.split("\t")
            opname_tokens = opname_full.split(".")
            lib = opname_tokens[0]
            opname = opname_tokens[1]
            op_scheme[lib + "::" + opname].append(scheme)
    with open("tools/op_scheme_pyi.txt") as fin:
        for line in fin.readlines():
            line = line.strip()
            line = line[4:]
            left_paranthesis_pos = line.find("(")
            opname = line[:left_paranthesis_pos]
            op_scheme[opname].append(line)
    return op_scheme


def get_op_scheme(opname, scheme_table):
    """
    Get the scheme of an operator.
    """
    if opname in scheme_table:
        return scheme_table[opname]

    if opname.startswith("aten::"):
        opname = opname[6:]

    if opname in scheme_table:
        return scheme_table[opname]

    # remove tailing underscores
    if opname.endswith("_"):
        opname = opname[:-1]

    # Look up in the scheme table
    if opname in scheme_table:
        return scheme_table[opname]

    return []


def annotate_scheme_file(
    in_file_name: str,
    out_file_name: str,
    op_scheme: Dict[str, List[str]],
    op_ok: set,
    op_ng: set,
):
    """
    Open the input file (JSON), annotate the operator's scheme, and save it to the output file.
    """
    print(f"Processing {in_file_name} -> {out_file_name} ...")
    with open(in_file_name, "r") as f:
        trace = json.load(f)

    new_trace = []

    for event in trace["traceEvents"]:
        opname = event["name"]
        # If the operator is not aten:: or prim::,
        # it is PyTorch internal op like profiling or some side effect op, skip it
        if not opname.startswith("aten::") and not opname.startswith("prim::"):
            continue

        # Get the schemes of the operator
        schemes = get_op_scheme(opname, op_scheme)
        if len(schemes) == 0:
            op_ng.add(opname)
        else:
            op_ok.add(opname)

        # Annotate the operation with the schemes
        try:
            e = {
                "opname": event["name"],
                "schemes": schemes,
            }
            if "Input type" in event["args"]:
                e.update(
                    {
                        "types": event["args"]["Input type"],
                        "shapes": event["args"]["Input Dims"],
                        "values": event["args"]["Concrete Inputs"],
                    }
                )
        except Exception as e:
            print(e)
            print("===============")
            print(event)
            print("===============")
            raise e

        # Add the annotated operation to the new trace
        new_trace.append(e)

    # Save the annotated trace to the output file
    with open(out_file_name, "w") as f:
        json.dump(new_trace, f, indent=2)


def annotate_scheme_dir(in_dir: str, out_dir: str, op_scheme: Dict[str, List[str]]):
    op_ok = set()
    op_ng = set()
    # Create output folder if not exists
    os.makedirs(out_dir, exist_ok=True)
    # Glob all files in the input directory
    files = glob.glob(in_dir + "/*")
    # For each file, annotate the scheme and save it to the output directory
    for f in files:
        out_file = out_dir + "/" + f.split("/")[-1]
        annotate_scheme_file(f, out_file, op_scheme, op_ok, op_ng)

    # Print out the operators that are not in the scheme table
    with open("/tmp/ok", "w") as fout:
        for opname in sorted(op_ok):
            fout.write(opname + "\n")
    with open("/tmp/ng", "w") as fout:
        for opname in sorted(op_ng):
            fout.write(opname + "\n")


def main():
    op_scheme = load_op_scheme()
    annotate_scheme_dir("stat/profile/", "/tmp/trace", op_scheme)
    #  annotate_scheme_file("stat/profile/resnet18", "/tmp/ker", op_scheme)


if __name__ == "__main__":
    main()