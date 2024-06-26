import glob
import json
from typing import List, Dict, Optional
import collections
import os



def load_op_scheme():
    """
    Scheme is a {opname: List[str]} dictionary.
    An op might have multiple schemes, because it has multiple signatures.
    """
    op_scheme = collections.defaultdict(list)
    # The key is "lib::opname", for example, "aten::conv2d"
    with open("tools/op_scheme_doc.txt", "r") as fin:
        for line in fin.readlines():
            line = line.strip()
            opname_full, scheme = line.split("\t")
            opname_tokens = opname_full.split(".")
            lib = opname_tokens[0]
            opname = opname_tokens[1]
            op_scheme[lib + "::" + opname].append(scheme)
    # # The key is "opname", for example, "conv2d"
    # with open("tools/op_scheme_pyi.txt") as fin:
    #     for line in fin.readlines():
    #         line = line.strip()
    #         line = line[4:]
    #         left_paranthesis_pos = line.find("(")
    #         opname = line[:left_paranthesis_pos]
    #         op_scheme[opname].append(line)
    # # The key is "lib::opname", for example, "aten::conv2d"
    # with open("tools/op_scheme_torch_mlir.txt", "r") as fin:
    #     for line in fin.readlines():
    #         line = line.strip()
    #         if not (line.startswith("def aten") or line.startswith("def prims")):
    #             continue
    #         line = line[4:]
    #         left_paranthesis_pos = line.find("(")
    #         opname_full = line[:left_paranthesis_pos]
    #         # There has special character in the parsed data
    #         opname_tokens = opname_full.replace("〡", "〇").split("〇")
    #         lib = opname_tokens[0]
    #         opname = opname_tokens[1]
    #         line = line.replace("〡", ".").replace("〇", ".")
    #         op_scheme[lib + "::" + opname].append(line)
    # The key is "lib::opname", for example, "aten::conv2d"
    with open("tools/op_scheme_aten_declarations.txt", "r") as fin:
        keyword = '// {"schema":'
        for line in fin.readlines():
            line = line.strip()
            if keyword not in line:
                continue
            keyword_pos = line.find(keyword)
            comment_str = line[keyword_pos+3:]
            comment_dict = eval(comment_str)
            scheme = comment_dict['schema']
            left_paranthesis_pos = scheme.find("(")
            opname = scheme[:left_paranthesis_pos].split(".")[0]
            op_scheme[opname].append(scheme)
    return op_scheme


def get_op_scheme(opname, scheme_table):
    """
    Get the scheme of an operator.
    """
    if opname in scheme_table:
        return scheme_table[opname]
    if opname.endswith("_") and opname[:-1] in scheme_table:
        return scheme_table[opname[:-1]]

    if opname.startswith("aten::"):
        opname = opname[6:]

    if opname in scheme_table:
        return scheme_table[opname]
    if opname.endswith("_") and opname[:-1] in scheme_table:
        return scheme_table[opname[:-1]]

    return []


def annotate_scheme_file(
    in_file_name: str,
    out_file_name: str,
    op_scheme: Dict[str, List[str]],
    op_ok: Optional[set] = None,
    op_ng: Optional[set] = None,
):
    """
    Open the input file (JSON), annotate the operator's scheme, and save it to the output file.
    """
    print(f"Processing {in_file_name} -> {out_file_name} ...")
    with open(in_file_name, "r") as f:
        trace = json.load(f)

    new_trace = []
    model_ng_op = set()

    for event in trace["traceEvents"]:
        opname = event["name"]
        # If the operator is not aten:: or prim::,
        # it is PyTorch internal op like profiling or some side effect op, skip it
        if not opname.startswith("aten::") and not opname.startswith("prims::"):
            continue

        # Get the schemes of the operator
        schemes = get_op_scheme(opname, op_scheme)
        if len(schemes) == 0:
            if op_ng is not None:
                op_ng.add(opname)
            if opname not in model_ng_op:
                model_ng_op.add(opname)
                print(f"  Operator {opname} not in the scheme table")
        else:
            if op_ok is not None:
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
                        "input_types": event["args"]["Input type"],
                        "input_shapes": event["args"]["Input Dims"],
                        "input_values": event["args"]["Concrete Inputs"],
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
    annotate_scheme_dir("stat/profile/", "stat/trace", op_scheme)
    #  annotate_scheme_file("stat/profile/resnet18", "/tmp/ker", op_scheme)


if __name__ == "__main__":
    main()
