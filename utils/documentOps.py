#!/usr/bin/python3

# SPDX-License-Identifier: Apache-2.0

##################### documentOps.py ########################################
#
# Copyright 2022, 2024 The IBM Research Authors.
#
################################################################################
#
# This file scans for certain patterns (listed below) and generate an md table,
# which list the operations supported, and optionally the unsupported operations.
# Among the options, we can also list the TODOs in the table.
# Invoke with the `-h` argument to list options.
#
# Limitation: currently handle at most one OP/LIMIT/TODO line per operation.
# Script currently invoked by the `onnx_mlir_supported_ops` make target.
#
################################################################################

################################################################################
# Default min/max opset supported (when not explicitly specified).
# Default values are used when no explicit ==MIN==/==MAX== values are used.
min_opset_default = 6
max_opset_default = "*"
# NNPA supported. Ordered list, with oldest first and most recent last.
nnpa_supported_list = ["z16"]

# Derived variables (do not update).
nnpa_level_default = nnpa_supported_list[-1]  # Most recent arch (last).
nnpa_supported_set = set(nnpa_supported_list)
nnpa_most_current_str = ""
if len(nnpa_supported_list) > 1:
    nnpa_most_current_str = " - ^"

import sys
import getopt
import fileinput
import re
import json
import subprocess

################################################################################
# SEMANTIC for LABELING (one line per directive)
#
# ==ARCH== <arch>
#   where <arch> is cpu or NNPA.. this option is valid until reset by another ARCH dir
#
# ==LEVEL== <levels>
#   where <levels> is a comma separated names of versions supported by NNPA.
#
# ==OP== <op> <text>
#   where <op> is the ONNX op name
#   where <text> is optional text, currently unused
#
# ==LIM== <text>
#   where <text> qualifies the current restrictions to the implementation.
#
# ==TODO== <text>
#   where <text> add "private" info about what needs to be fixed.
#
# ==MIN== <num>
#   where <num> is the minimum release version supported.
#
# ==UNSUPPORTED== <num>
#   where <num> is the unsupported release version of the operator.
#
################################################################################


################################################################################
# Usage.
def print_usage(msg=""):
    if msg:
        print("Error:", msg, "\n")
    print(
        "\nGenerate MD document tables for the supported ops using the labeling left in files."
    )
    print("For labeling format, consult the python script directly.")
    print("documentOps [-a <arch>] [-dnu] -i <file> [-p <path>")
    print('  -a, --arch <arch>: report on "==ARCH== <arch>".')
    print("  -d, --debug: include debug.")
    print("  -i, --input <file name>: input file.")
    if "NNPA" in target_arch:
        print('  -l, --level <level>: report on "==LEVEL== <level>".')
    print("  -n, --notes: include notes/TODOs.")
    print("  -p, --path <util path>: path to onnx-mlir util directory.")
    print("  -u, --unsupported: list unsupported ops.")
    sys.exit()


################################################################################
# Handling of info: global dictionaries.

hightest_opset = None  # Highest opset found in the description.
opset_dict = {}  # <op> -> <text> in "==OP== <op> <text>".
limit_dict = {}  # <op> -> <text> in "==LIM== <text>".
level_dict = {}  # <op> -> <text> in "==LEVEL== <text>".
min_dict = {}  # <op> -> <num> in "==MIN== <num>".
max_dict = {}  # <op> -> <num> in "==MAX== <num>".
todo_dict = {}  # <op> -> <text> in "==TODO== <text>".
list_op_version = {}  # List of operation versions from gen_onnx_mlir;
# <op> -> [supported versions]
additional_top_paragraph = ""  # <text> in "==ADDITIONAL_TOP_PARAGRAPH <text>"

################################################################################
# Parse input file. Add only info if it is the proper target arch. Other entries
# and non-relevant data is simply ignored. At this time, does not support
# multiple entries of any kind. Everything is case sensitive.


def dotted_sentence(str):
    if re.match(r".*\.\s*$", str) is None:
        return str + "."
    return str


def parse_file(file_name):
    global additional_top_paragraph
    try:
        file = open(file_name, "r")
    except OSError:
        print_usage("Could not open file `" + file_name + "`")
    op = ""
    arch = ""
    for line in file:
        l = line.rstrip()
        # Scan arch.
        p = re.search(r"==ARCH==\s+(\w+)", l)
        if p is not None:
            arch = p[1]
            if debug:
                print("process arch", arch)
                continue
        if arch != target_arch:
            continue
        # Additional top paragraph
        p = re.search(r"==ADDITIONAL_PARAGRAPH==\s+(.*)\s*$", l)
        if p is not None:
            additional_top_paragraph = dotted_sentence(p[1])
            if debug:
                print("process paragraph", additional_top_paragraph)
            continue
        # Scan op.
        p = re.search(r"==OP==\s+(\w+)", l)
        if p is not None:
            op = p[1]
            assert op not in opset_dict, "Redefinition of op " + op
            assert op in list_op_version, (
                "Define an op "
                + op
                + " that is not listed in the ops we currently handle."
            )
            versions = list_op_version[op]
            opset_dict[op] = versions
            if debug:
                print("got supported op", op, "at level", list_op_version[op])
            continue
        # Scan NNPA Level
        if "NNPA" in target_arch:
            p = re.search(r"==LEVEL==\s+(\w+)(?:,\s*(\w+))*", l)
            if p is not None:
                assert op is not None, "Level without op."
                assert op not in level_dict, "Redefinition of level for op " + op
                current_set = set(p.groups())
                join_set = current_set & nnpa_supported_set
                if not join_set:
                    if debug:
                        print(
                            "process NNPA level, no overlap between",
                            current_set,
                            "and",
                            nnpa_supported_set,
                        )
                else:
                    # Find the first and last in set according to the order in nnpa_supported_list.
                    first_in_set = None
                    last_in_set = None
                    for x in nnpa_supported_list:
                        if x in join_set:
                            last_in_set = x
                            if first_in_set == None:
                                first_in_set = x
                    assert first_in_set and last_in_set, "should both be defined"
                    if debug:
                        print(
                            "join set is",
                            join_set,
                            "first",
                            first_in_set,
                            "last",
                            last_in_set,
                        )
                    if last_in_set == nnpa_level_default:  # First to default (current).
                        level_dict[op] = first_in_set + nnpa_most_current_str
                    elif first_in_set == last_in_set:  # Only one.
                        level_dict[op] = first_in_set
                    else:  # Interval finishing before current.
                        level_dict[op] = first_in_set + " - " + last_in_set
                    if debug:
                        print("process NNPA level", level_dict[op])
                continue
        # Limits.
        p = re.search(r"==LIM==\s+(.*)\s*$", l)
        if p is not None:
            assert op is not None, "Limit without op."
            assert op not in limit_dict, "Redefinition of limit for op " + op
            limit_dict[op] = dotted_sentence(p[1])
            if debug:
                print("Got limit for op", op, ":", limit_dict[op])
            continue
        p = re.search(r"==TODO==\s+(.*)\s*$", l)
        if p is not None:
            assert op is not None, "Todo without op."
            assert op not in todo_dict, "Redefinition of todo for op " + op
            todo_dict[op] = dotted_sentence(p[1])
            if debug:
                print("got todo for op", op, ":", todo_dict[op])
            continue
        # Min release supported.
        p = re.search(r"==MIN==\s+(\d+)\s*$", l)
        if p is not None:
            assert op is not None, "Min without op."
            assert op not in min_dict, "Redefinition of min for op " + op
            op_min = max(int(p[1]), min_opset_default)
            min_dict[op] = op_min
            # By default, assume we support the latest op version. (overridden by UNSUPPORTED)
            max_dict[op] = max_opset_default
            if debug:
                print("Got min for op", op, ":", min_dict[op])
            continue
        # Max release supported.
        p = re.search(r"==UNSUPPORTED==\s+(.*)\s*$", l)
        if p is not None:
            assert op is not None, "Unsupported without op."
            assert op in min_dict and min_dict[op], (
                "Unsupported without min for op " + op
            )
            assert max_dict[op] == max_opset_default, (
                "Redefinition of Unsupported for op " + op
            )
            assert (
                int(p[1]) > min_dict[op]
            ), f"Unsupported version {p[1]} should be greater than min {min_dict[op]} for op {op}"
            # Show the last compatible Opset for the version of the Op
            max_dict[op] = int(p[1]) - 1
            if debug:
                print("Got unsupported for op", op, ":", max_dict[op])
            continue


################################################################################
# Print info.


def print_row(array):
    str = "| "
    for a in array:
        str += a + " |"
    print(str)


def print_md():
    # Header.
    print("<!--- Automatically generated, do not edit. -->")
    print("<!--- To update, run `make onnx_mlir_supported_ops_" + target_arch + "' -->")
    # Title
    print("\n# Supported ONNX Operation for Target *" + target_arch + "*.\n")
    # Top paragraph.
    print(
        "Onnx-mlir currently supports ONNX operations targeting up to "
        + "opset "
        + str(hightest_opset)
        + ". Limitations are listed when applicable."
        + " This documentation highlights the minimum and maximum opset versions that"
        + " are fully supported by onnx-mlir and not the version changes."
        + "\n"
    )
    print(
        "* Operations are defined by the [ONNX Standard]"
        + "(https://github.com/onnx/onnx/blob/main/docs/Operators.md)."
    )
    print(
        "* **Supported Opsets** indicates the lowest and highest opset a model"
        + " may have for onnx-mlir to support compiling a model with the operator."
    )
    print(
        "   * A * indicates onnx-mlir is compatible with the latest"
        + " version of that operator available as of opset "
        + str(hightest_opset)
        + "."
    )
    if "NNPA" in target_arch:
        print(
            "   * A ^ indicates onnx-mlir is compatible with the latest"
            + " level of the NNPA Architecture which is "
            + str(nnpa_level_default)
            + "."
        )

    print("\n")
    # Additional top paragraph.
    if additional_top_paragraph:
        print(additional_top_paragraph)
        print("\n")
    # Table.
    if "NNPA" in target_arch:
        header = [
            "Op",
            "Supported Opsets (inclusive)",
            "Minimum NNPA Level(Inclusive)",
            "Limitations",
        ]
        separator = ["---", "---", "---", "---"]
    else:
        header = ["Op", "Supported Opsets (inclusive)", "Limitations"]
        separator = ["---", "---", "---"]
    if emit_notes:
        header.append("Notes")
        separator.append("---")
    print_row(header)
    print_row(separator)
    for op in sorted(list_op_version.keys()):
        supported_op = op in min_dict
        if supported_op and "NNPA" in target_arch and op not in level_dict:
            supported_op = False
        if supported_op:
            info = ["**" + op + "**", f"{min_dict[op]} - {max_dict[op]}"]
            if "NNPA" in target_arch:
                info = [
                    "**" + op + "**",
                    f"{min_dict[op]} - {max_dict[op]}",
                    f"{level_dict[op]}",
                ]
        else:
            if not emit_unsupported:
                continue
            info = ["**" + op + "**", "none", ""]
        if op in limit_dict:
            info.append(limit_dict[op])
        else:
            info.append("")
        if emit_notes:
            if op in todo_dict:
                info.append(todo_dict[op])
            else:
                info.append("")
        print_row(info)


def main(argv):
    global debug, target_arch, emit_notes, emit_unsupported, input_command, additional_top_paragraph
    global list_op_version, hightest_opset
    debug = 0
    target_arch = "cpu"
    emit_notes = 0
    emit_unsupported = 0
    util_path = "."
    file_name = ""
    input_command = "python3 documentOps.py"

    try:
        opts, args = getopt.getopt(
            argv,
            "a:dhi:np:u",
            ["arch=", "debug", "help", "input=", "notes", "path=", "unsupported"],
        )
    except getopt.GetoptError:
        print_usage()
    for opt, arg in opts:
        if opt in ("-a", "--arch"):
            target_arch = arg
            input_command += " --arch " + target_arch
        elif opt in ("-l", "--level"):
            nnpa_level = arg
            input_command += " --level " + nnpa_level
        elif opt in ("-d", "--debug"):
            debug = 1
        elif opt in ("-h", "--help"):
            print_usage()
        elif opt in ("-i", "--input"):
            file_name = arg
            input_command += " --input " + file_name
        elif opt in ("-n", "--notes"):
            emit_notes = True
            input_command += " --notes"
        elif opt in ("-p", "--path"):
            util_path = arg
            input_command += " --path " + util_path
        elif opt in ("-u", "--unsupported"):
            emit_unsupported = True
            input_command += " --unsupported"

    if not file_name:
        print("Command requires an input file name.\n")
        print_usage()

    # Load gen_onnx_mlir operation version.
    proc = subprocess.Popen(
        ["python3", util_path + "/gen_onnx_mlir.py", "--list-operation-version"],
        stdout=subprocess.PIPE,
    )
    str = ""
    for line in proc.stdout:
        str += line.decode("utf-8").rstrip()
    list_op_version = eval(str)
    hightest_opset = max([max(i) for i in list_op_version.values()])
    if debug:
        print("List op version is: ", list_op_version)

    # Parse and print md table.
    parse_file(file_name)
    print_md()


if __name__ == "__main__":
    main(sys.argv[1:])
