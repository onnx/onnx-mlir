#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

################# OpLevelParallel.py ###########################################
#
# Copyright 2019-2024 The IBM Research Authors.
#
################################################################################
#
# This code analyzes to the profile data and compiled MLIR code,
# (1) Detects sets of operations to be parallelized by fork and join, and
# (2) Genearts parallelized MLIR code
#
################################################################################

import os
import sys
import re
import networkx as nx
import argparse


num_in_str = lambda str: [int(c) if c.isdigit() else c for c in re.split(r"(\d+)", str)]

MAX_NODE_NUM_IN_BLOCK = 100
MIN_PARALLEL_NUM = 2
MIN_EXECUTION_TIME_IN_BLOCK = -1.0
KEY_OPERATIONS = ["onnx.Conv", "onnx.MatMul", "onnx.LSTM"]
NO_PARENT_OPERATIONS = ["onnx.Constant", "onnx.NoValue"]
INST_OPERATIONS = [
    "onnx.Conv",
    "onnx.MatMul",
    "onnx.LSTM",
    "onnx.Softplus",
    "onnx.Erf",
    "onnx.Add",
    "onnx.Div",
    "onnx.Mul",
]

LINE_ONNXPARALLEL = '{} = "onnx.Parallel"() ({{'  # endvar
LINE_ONNXPARALLELYIELD = "onnx.Yield {} : {}"  # endvar, endvars_shape
LINE_ONNXPARALLELEND = "}}) : () -> ({})"  # endvar_shape
LINE_ONNXYIELD = '"onnx.Yield"({}):({}) -> ()'  # endvars, endvars_shape
LINE_ONNXFORK = '{} = "onnx.Fork"() ({{'  # forkvar
LINE_ONNXFORKEND = "}}) {{id={}:si64}}:() -> {}"  # th_id, endvar_shape

LINE_INSTRUMENT = '"krnl.runtime_instrument"() {{nodeName = "{}", opName = "{}", tag = {} : i64 }}: () -> ()'  # nodeName, opName, tag
INSTRUMENT_BEFORE_OP = 0x1
INSTRUMENT_AFTER_OP = 0x2
INSTRUMENT_REPORT_TIME = 0x4
INSTRUMENT_REPORT_MEMORY = 0x8
INSTRUMENT_INIT = 0x10
INST_START = INSTRUMENT_BEFORE_OP | INSTRUMENT_REPORT_TIME
INST_FINISH = INSTRUMENT_AFTER_OP | INSTRUMENT_REPORT_TIME


def valid_onnx_input(fname):
    valid_exts = ["mlir"]  # ["onnx", "mlir", "onnxtext"]
    ext = os.path.splitext(fname)[1][1:]

    if ext not in valid_exts:
        parser.error(
            "Only accept an input model with one of extensions {}".format(valid_exts)
        )
    return fname


# Read profile file, and return dictionay for execution time from node
def read_profile(profile):
    time_dict = {}
    count_dict = {}
    with open(profile) as f:
        for line in f:
            columns = line.split(",")
            first_column = columns[0].strip()
            if first_column != "==PERF-REPORT==":
                continue
            nodes = columns[2].strip()
            before_after = columns[3].strip()
            time = float(columns[4].strip())
            nodes_split = nodes.split("-")
            num_nodes = len(list(nodes_split))
            for node in nodes_split:
                if before_after == "before":
                    node = "BEFORE"
                if not node in time_dict:
                    time_dict[node] = 0.0
                    count_dict[node] = 0
                time_dict[node] += time / num_nodes
                count_dict[node] += 1
    return time_dict, count_dict


NODENAME_STR = "onnx_node_name = "
NONEOP_STR = "NONE"


# Parse line in MLIR file, and return outvar, invars, operation, nodename and outvar's shape
def parse_line_in_model(line):
    columns = line.strip().split(" ")
    if (columns[0] == "module") or (columns[0] == "func.func") or (columns[0] == "}"):
        return ("", [], "", "", "")
    # get outvar
    vars_list = line[: line.find("=") - 1].strip().split(",")
    for vars in vars_list:
        if vars.strip()[0] != "%":
            return ("", [], "", "", "")
    # XXXXX TODO: The current version processes the first output only. And
    # second and futher outputs are ignored. it is O.K for ONNXIR, but not
    # for other dialects (e.g. KernelIR).
    outvar = vars_list[0]
    # get operation
    opstr = line[line.find("=") + 1 :]  # remove heading str until '=' if found
    opstr = opstr[opstr.find('"') + 1 :]  # remove heading str until '"' if found
    opstr = opstr[: opstr.find('"')]  # remove tailing str from ')' if found
    operation = opstr.strip().split(" ")[0]
    # get invars
    s = line[line.find("=") + 1 :]  # remove heading str until "=" if found
    s = s[s.find("(") + 1 :]  # remove heading str until "(" if found
    s = s[: s.find(")") + 1]  # remove tailing str from ")" if found
    s = s[: opstr.find(":")]  # remove tailing str from ")" if found
    invars = s.replace(" ", "").split(",") if s else []
    # get nodename
    index = line.find(NODENAME_STR)
    nodename = NONEOP_STR
    if index >= 0:
        index_start = index + len(NODENAME_STR) + 1
        index_end = index_start + line[index_start:].find('"')
        nodename = line[index_start:index_end].strip()
    # Get operation shape
    opshape = line[line.rfind(":") + 1 :]  # remove heading str until last ':'
    opshape = opshape.replace(" ", "").strip()
    return (outvar, invars, operation, nodename, opshape)


# Generate networkx graph from the model file in MLIR
def generate_model_graph(model, profile_dict):
    model_graph = nx.DiGraph()
    key_operations = []
    with open(model) as f:
        for line in f:
            outvar, invars, operation, nodename, opshape = parse_line_in_model(line)
            if not outvar:
                continue
            time = profile_dict[nodename] if (nodename in profile_dict) else 0.0
            # print("GENMODEL: outvar={}, invars={}, operation={}, nodename={}, opshape={}, time={}: {}".format(outvar, invars, operation, nodename, opshape, time, line.strip()))
            model_graph.add_node(
                outvar,
                invars=invars,
                operation=operation,
                nodename=nodename,
                opshape=opshape,
                time=time,
                line=line,
            )
            for invar in invars:
                model_graph.add_edge(invar, outvar)
            if operation in KEY_OPERATIONS:
                key_operations.append(outvar)
    return (model_graph, key_operations)


def get_operation_from_node(node, model_graph):
    nodeattr = model_graph.nodes[node]
    operation = nodeattr["operation"] if ("operation" in nodeattr) else 0.0
    return operation


def get_time_from_node(node, model_graph):
    nodeattr = model_graph.nodes[node]
    time = nodeattr["time"] if ("time" in nodeattr) else 0.0
    return float(time)


def get_node_str(
    node,
    model_graph,
    get_node=False,
    get_nodename=True,
    get_operation=True,
    get_opshape=True,
    key_opshape_only=True,
    get_time=False,
):
    nodeattr = model_graph.nodes[node]
    nodename = nodeattr["nodename"] if ("nodename" in nodeattr) else ""
    operation = nodeattr["operation"] if ("operation" in nodeattr) else ""
    opshape = nodeattr["opshape"] if ("opshape" in nodeattr) else ""
    time = float(nodeattr["time"] if ("time" in nodeattr) else 0.0)
    nodestr = ""
    sep = ""
    if get_node:
        nodestr += sep + node
        sep = ":"
    if get_nodename:
        nodestr += sep + nodename
        sep = ":"
    if get_operation:
        nodestr += sep + operation
        sep = ":"
    if get_opshape and (not key_opshape_only or (operation in KEY_OPERATIONS)):
        nodestr += sep + opshape
        sep = ":"
    if get_time:
        nodestr += sep + "{:.3f}:".format(time * 1000)
        sep = ":"
    return nodestr


def print_graph(model_graph):
    for outvar, attr in model_graph.nodes(data=True):
        print("PRINTGRAPH: outvar={}, attr={}".format(outvar, attr))
        invars = attr["invars"] if ("invars" in attr) else ""
        operation = attr["operation"] if ("operation" in attr) else ""
        nodename = attr["nodename"] if ("nodename" in attr) else ""
        time = attr["time"] if ("time" in attr) else ""
        print(
            "PRINTGRAPH: outvar={}, invars={}, operation={}, nodename={}, time={}".format(
                outvar, invars, operation, nodename, time
            )
        )


def has_inputs_from_outside(node, subgraph, model_graph, parent=None):
    pred_list = list(model_graph.predecessors(node))
    new_node_list = list(set(pred_list) - set(subgraph) - set([node]))
    for new_node in new_node_list:
        # new node should be parent node or no input node
        if new_node != parent and list(model_graph.predecessors(new_node)):
            return True
    return False


def has_multiple_outputs_to_outside(node, subgraph, model_graph):
    succ_list = list(model_graph.successors(node))
    new_node_list = list(set(succ_list) - set(subgraph) - set([node]))
    return True if len(new_node_list) > 1 else False


def get_no_input_node_set(node, model_graph):
    pred_list = list(model_graph.predecessors(node))
    no_input_node_set = set()
    for pred_node in pred_list:
        for input_node in list(model_graph.predecessors(pred_node)):
            if not list(model_graph.predecessors(input_node)):
                no_input_node_set = no_input_node_set | set([input_node])
    return no_input_node_set


def get_successor_subgraph_ending_one_node(parent, node, model_graph):
    work_list = [node]  # list of outputs of nodes
    pending_list = []  # list of pending nodes
    subgraph = []  # list of nodes in the current subgraph
    no_input_node_set = set()  # set of no input node subgraph
    total_time = 0.0  # total time in the current subgraph
    best_candidate = ([], {}, 0.0)  # current best successor subgraph
    while work_list and len(subgraph) < args.max_node_num_in_block:
        # Get the current node from work_list
        curr = work_list.pop(0)
        # If the current node has inputs from outside of subgraph,
        # append the current node in pending_list and continue
        if has_inputs_from_outside(
            curr, subgraph, model_graph, parent
        ) or has_multiple_outputs_to_outside(curr, subgraph, model_graph):
            pending_list.append(curr)
            continue
        # Append the current node to the subgraph
        subgraph.append(curr)
        no_input_node_set |= get_no_input_node_set(curr, model_graph)
        total_time += get_time_from_node(curr, model_graph)
        # Remove the current node from pending_list if exist
        if curr in pending_list:
            pending_list.remove(curr)
        # Get new outputs of the current node not in the subgraph and work_list
        curr_succ_list = list(model_graph.successors(curr))
        new_outputs = list(set(curr_succ_list) - set(subgraph) - set(work_list))
        # Add the new output nodes to the work_list
        work_list = sorted(new_outputs + work_list)
        # Remove the new output nodes from pending_list
        pending_list = list(set(pending_list) - set(new_outputs))
        # Update the best_candidate if no nodes in working_list and pending_list
        if len(work_list) <= 1 and not pending_list:
            best_candidate = (subgraph, no_input_node_set, total_time)
    # sort the candidate
    subgraph, no_input_node_set, total_time = best_candidate
    sorted_no_input_node_list = sorted(list(best_candidate[1]), key=num_in_str)
    sorted_best_candidate = (subgraph, sorted_no_input_node_list, total_time)
    return sorted_best_candidate


def has_key_operation(subgraph, model_graph):
    for node in subgraph:
        operation = get_operation_from_node(node, model_graph)
        if operation in KEY_OPERATIONS:
            return True
    return False


def node_in_candidate_dict(node, candidate_dict):
    for outvar in candidate_dict:
        for block in candidate_dict[outvar]:
            subgraph, _, _ = block
            if node in subgraph:
                return True


def get_candidates(model_graph):
    candidate_dict = {}
    time_list = []
    for outvar, attr in model_graph.nodes(data=True):
        # skip outvar if outvar is already in candidate_dict
        if node_in_candidate_dict(outvar, candidate_dict):
            continue
        invars = attr["invars"] if ("invars" in attr) else ""
        operation = attr["operation"] if ("operation" in attr) else ""
        if not operation or (operation in NO_PARENT_OPERATIONS):
            continue
        nodename = attr["nodename"] if ("nodename" in attr) else ""
        time = attr["time"] if ("time" in attr) else ""
        succ_list = list(model_graph.successors(outvar))
        block_list = []
        if len(succ_list) >= args.min_parallel_num:
            for succ in succ_list:
                subgraph, no_input_nodes, time = get_successor_subgraph_ending_one_node(
                    outvar, succ, model_graph
                )
                if (
                    has_key_operation(subgraph, model_graph)
                    and time >= args.min_execution_time_in_block
                ):
                    block_list.append((subgraph, no_input_nodes, time))
        if len(block_list) >= args.min_parallel_num:
            candidate_dict[outvar] = block_list
    return candidate_dict


def print_key_operations(key_operations, model_graph):
    print("KEYOPERATIONS: {} [ ".format(len(key_operations), end=""))
    sep = ""
    for node in key_operations:
        print("{}{}".format(sep, get_node_str(node, model_graph)), end="")
        sep = ", "
    print(" ]")


def print_block(index, block, model_graph, full=True):
    nodes = block[0]
    no_input_nodes = block[1]
    time = block[2]
    print("    {}: BLOCK {} {} [ ".format(index, no_input_nodes, len(nodes)), end="")
    if args.profile:
        print(
            " time={:.3f} msec, ".format(time * 1000),
            end="",
        )
    if full:
        print("\n", end="")
        for node in nodes:
            print("    {}".format(model_graph.nodes[node]["line"]), end="")
    else:
        sep = ""
        for node in nodes:
            print("{}{}".format(sep, get_node_str(node, model_graph)), end="")
            sep = ", "
    print(" ]")


def print_candidate(index, parent, blocks, model_graph, full=True):
    if full:
        print("  {}: PARENT {} [".format(index, len(blocks)))
        print("{}".format(model_graph.nodes[parent]["line"]), end="")
    else:
        print(
            "  {}: PARENT {} {} [".format(
                index, get_node_str(parent, model_graph), len(blocks)
            )
        )
    for idx, block in enumerate(blocks):
        print_block(idx, block, model_graph, full)
    print("  ]")


def print_candidates(candidates, model_graph):
    print("CANDIDATES: {} [".format(len(candidates)))
    for idx, parent in enumerate(candidates.keys()):
        print_candidate(idx, parent, candidates[parent], model_graph)
    print("]")


def print_line(indent, line, file, end="\n"):
    indent_str = " " * indent
    print("{}{}".format(indent_str, line), file=file, end=end)


def instrument(indent, parent, thread_id, op, tag, file):
    if not args.instrument:
        return
    node = "P" + parent.replace("%", "")
    if thread_id >= 0:
        node = node + "_" + str(thread_id)
    print_line(indent, LINE_INSTRUMENT.format(node, op, tag), file)


# print a line in input code
def print_input_line(indent, line, file, parent="NONE", th_id=-1):
    _, _, operation, _, _ = parse_line_in_model(line)
    if operation in INST_OPERATIONS:
        inst_indent = indent + (len(line) - len(line.lstrip()))
        instrument(inst_indent, parent, th_id, operation, INST_START, file)
        print_line(indent, line, file, end="")
        instrument(inst_indent, parent, th_id, operation, INST_FINISH, file)
    else:
        print_line(indent, line, file, end="")


def get_var_shape(var, outvar_lineno_dict, lines_list):
    line = lines_list[outvar_lineno_dict[var]]
    _, _, _, _, opshape = parse_line_in_model(line)
    var_shape = opshape[opshape.find("->") + 2 :]
    return var_shape


def get_endvar_list_from_candidate(candidate, lineno_dict, lines_list):
    endvar_list = []
    endvar_shape_list = []
    for cand in candidate:
        endvar = cand[0][-1]
        endvar_list.append(endvar)
        endvar_shape_list.append(get_var_shape(endvar, lineno_dict, lines_list))
    return endvar_list, endvar_shape_list


def generate_paracode_for_candidate(
    parent,
    parent_lineno,
    candidate,
    lines_list,
    outvar_lineno_dict,
    processed_lineno_dict,
    f,
    dummy=False,
):
    parent_line = lines_list[parent_lineno]
    print_input_line(0, parent_line, f)
    indent = len(parent_line) - len(parent_line.lstrip())
    processed_lineno_dict[parent_lineno] = True
    th_id = 1
    endvar_list, endvar_shape_list = get_endvar_list_from_candidate(
        candidate, outvar_lineno_dict, lines_list
    )
    endvar_str = ", ".join(endvar_list)
    endvar_shape_str = ", ".join(endvar_shape_list)
    # Generate beginning part of parallel op
    if not dummy:
        instrument(indent, parent, -1, "onnx.Parallel", INST_START, f)
        print_line(indent, LINE_ONNXPARALLEL.format(endvar_str), f)
    for cand in candidate:
        blocks = cand[0]
        no_input_nodes = cand[1]
        time = cand[2]
        endvar = blocks[-1]
        endvar_line = lines_list[outvar_lineno_dict[endvar]]
        endvar_shape = endvar_line[endvar_line.rfind("->") + 2 :].strip()
        forkvar = endvar
        # Generate no_input ops before fork op
        for no_input in no_input_nodes:
            no_input_lineno = outvar_lineno_dict[no_input]
            if no_input_lineno not in processed_lineno_dict:
                print_input_line(0, lines_list[no_input_lineno], f)
                processed_lineno_dict[no_input_lineno] = True
        # Generate begining part of fork op
        if not dummy:
            instrument(indent, parent, th_id, "onnx.Fork", INST_START, f)
            print_line(indent, LINE_ONNXFORK.format(forkvar), f)
        # Generate block part
        instrument(indent + 2, parent, th_id, "onnx.Block", INST_START, f)
        for block in blocks:
            block_lineno = outvar_lineno_dict[block]
            print_input_line(2, lines_list[block_lineno], f, parent, th_id)
            processed_lineno_dict[block_lineno] = True
        instrument(indent + 2, parent, th_id, "onnx.Block", INST_FINISH, f)
        # Generate ending part of fork op
        if not dummy:
            print_line(indent + 2, LINE_ONNXYIELD.format(endvar, endvar_shape), f)
            print_line(indent, LINE_ONNXFORKEND.format(th_id, endvar_shape), f)
            instrument(indent, parent, th_id, "onnx.Fork", INST_FINISH, f)
        th_id += 1
    # Generate ending part of parallel op
    if not dummy:
        print_line(
            indent,
            LINE_ONNXPARALLELYIELD.format(endvar_str, endvar_shape_str),
            f,
        )
        print_line(indent, LINE_ONNXPARALLELEND.format(endvar_shape_str), f)
        instrument(indent, parent, -1, "onnx.Parallel", INST_FINISH, f)


def generate_paracode(candidates, model, output, dummy=False):
    # create outvar_list and line_outvar_dict
    lines_list = []
    outvar_lineno_dict = {}
    lineno_outvar_dict = {}
    last_return_lineno = 0
    with open(model) as f:
        lineno = 0
        for line in f:
            lines_list.append(line)
            outvar, _, _, _, _ = parse_line_in_model(line)
            outvar_lineno_dict[outvar] = lineno
            lineno_outvar_dict[lineno] = outvar
            first_column = line.strip().split(" ")[0]
            if first_column == "onnx.Return" or first_column == "return":
                last_return_lineno = lineno
            lineno += 1
    lines_list_len = len(lines_list)
    with open(output, "w") as f:
        processed_lineno_dict = {}
        total_instrumentation_started = False
        for lineno, line in enumerate(lines_list):
            outvar = lineno_outvar_dict[lineno]
            # generate instrument op for total start just before the first execution op
            if args.instrument and outvar and not total_instrumentation_started:
                tag = INSTRUMENT_INIT | INSTRUMENT_BEFORE_OP | INSTRUMENT_REPORT_TIME
                print_line(4, LINE_INSTRUMENT.format("Total", "Total", tag), f)
                total_instrumentation_started = True
            # generate instrument op for total end just before the last line
            if args.instrument and lineno == last_return_lineno:
                tag = INSTRUMENT_AFTER_OP | INSTRUMENT_REPORT_TIME
                print_line(4, LINE_INSTRUMENT.format("Total", "Total", tag), f)
            if outvar in candidates:
                blocks = candidates[outvar]
                generate_paracode_for_candidate(
                    outvar,
                    lineno,
                    blocks,
                    lines_list,
                    outvar_lineno_dict,
                    processed_lineno_dict,
                    f,
                    dummy,
                )
            elif lineno not in processed_lineno_dict:
                print_input_line(0, line, f)
                processed_lineno_dict[lineno] = True


# Command arguments.
parser = argparse.ArgumentParser()
parser.add_argument(
    "-p",
    "--profile",
    type=str,
    default="",
    help="Path to a profile file generated by --InstrumentReportTime option",
)
parser.add_argument(
    "-m",
    "--model",
    type=lambda s: valid_onnx_input(s),
    default="",
    help="Path to an ONNX model (.onnx or .mlir)",
)
parser.add_argument(
    "--max-node-num-in-block",
    type=int,
    default=MAX_NODE_NUM_IN_BLOCK,
    help="Maximum node number in block (default={})".format(MAX_NODE_NUM_IN_BLOCK),
)
parser.add_argument(
    "--min-parallel-num",
    type=int,
    default=MIN_PARALLEL_NUM,
    help="Minimum parallel number (default={})".format(MIN_PARALLEL_NUM),
)
parser.add_argument(
    "--min-execution-time-in-block",
    type=float,
    default=MIN_EXECUTION_TIME_IN_BLOCK,
    help="Minimum execution time (sec) in block (default={}). (work with --profile)".format(
        MIN_EXECUTION_TIME_IN_BLOCK
    ),
)
parser.add_argument(
    "--print-model-graph",
    action="store_true",
    help="Flag to print model graph (default={})".format(False),
)
parser.add_argument(
    "--print-key-operations",
    action="store_true",
    help="Flag to print key operations(default={})".format(False),
)
parser.add_argument(
    "--print-candidates",
    action="store_true",
    help="Flag to print candidates to be parallelized(default={})".format(False),
)
parser.add_argument(
    "--generate-originalcode",
    type=str,
    default="",
    help="Path to generate original code (not generated if not specified)",
)
parser.add_argument(
    "--generate-dummyparacode",
    type=str,
    default="",
    help="Path to generate dummypara code (not generated if not specified)",
)
parser.add_argument(
    "--generate-paracode",
    type=str,
    default="",
    help="Path to generate paralized code (not generated if not specified)",
)
parser.add_argument(
    "--instrument",
    action="store_true",
    help="Flag to set instrumentation(default={})".format(False),
)


args = parser.parse_args()
if not args.model:
    print("error: no model file, use argument --model")
    print(parser.format_usage())
    exit(1)
if args.min_execution_time_in_block > 0.0 and not args.profile:
    print("error: --min-execution-time-in-block optionworks with --profile option")
    print(parser.format_usage())
    exit(1)


#
# Main program
#
def main():
    profile_dict = {}
    if args.profile:
        profile_dict, _ = read_profile(args.profile)
    model_graph, key_operations = generate_model_graph(args.model, profile_dict)
    if args.print_model_graph:
        print_graph(model_graph)
    candidates = get_candidates(model_graph)
    if args.profile:
        print("PROFILE: {}".format(args.profile))
    if args.print_key_operations:
        print_key_operations(key_operations, model_graph)
    if args.print_candidates:
        print_candidates(candidates, model_graph)
    if args.generate_dummyparacode:
        generate_paracode(
            candidates, args.model, args.generate_dummyparacode, dummy=True
        )
    if args.generate_originalcode:
        generate_paracode({}, args.model, args.generate_originalcode)
    if args.generate_paracode:
        generate_paracode(candidates, args.model, args.generate_paracode)


if __name__ == "__main__":
    main()
