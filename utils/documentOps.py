#!/usr/local/bin/python3

# SPDX-License-Identifier: Apache-2.0

##################### documentOps.py ########################################
#
# Copyright 2022 The IBM Research Authors.
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
#   where <arch> is cpu/NNPA/... this option is valid until reset by another ARCH dir
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
#
################################################################################
# Usage.

def print_usage():
    print('\nGenerate MD document tables for the supported ops using the labeling left in files.')
    print("For labeling format, consult the python script directly.")
    print('documentOps [-a <arch>] [-dnu] -i <file> [-p <path>')
    print('  -a, --arch <arch>: report on "==ARCH== <arch>".')
    print('  -d, --debug: include debug.')
    print('  -i, --input <file name>: input file.')
    print('  -n, --notes: include notes/TODOs.')
    print('  -p, --path <util path>: path to onnx-mlir util directory.')
    print('  -u, --unsupported: list unsupported ops.')
    sys.exit()

################################################################################
# Handling of info: global dictionaries.

hightest_opset = 1   # Highest opset is.
opset_dict = {}        # <op> -> <text> in "==OP== <op> <text>".
limit_dict = {}        # <op> -> <text> in "==LIM== <text>".
todo_dict = {}         # <op> -> <text> in "==TODO== <text>".
list_op_version = {}   # List of operation versions from gen_onnx_mlir;
                       # <op> -> [supported versions]
additional_top_paragraph = "" # <text> in "==ADDITIONAL_TOP_PARAGRAPH <text>"

################################################################################
# Parse input file. Add only info if it is the proper target arch. Other entries
# and non-relevant data is simply ignored. At this time, does not support 
# multiple entries of any kind. Everything is case sensitive.

def dotted_sentence(str):
    if re.match(r'.*\.\s*$', str) is None:
        return str + "."
    return str

def parse_file(file_name):
    global hightest_opset, additional_top_paragraph
    file = open(file_name, 'r')
    op = ""
    arch = ""
    for line in file:
        l = line.rstrip()
        # Scan arch.
        p = re.search(r'==ARCH==\s+(\w+)', l)
        if p is not None: 
            arch = p[1]
            if debug:
                print("process arch", arch)
            continue
        if arch != target_arch:
            continue
        # Additional top paragraph
        p = re.search(r'==ADDITIONAL_PARAGRAPH==\s+(.*)\s*$', l)
        if p is not None:
            additional_top_paragraph = dotted_sentence(p[1])
            if debug:
                print("process paragraph", additional_top_paragraph)
            continue
        # Scan op.
        p = re.search(r'==OP==\s+(\w+)', l)
        if p is not None: 
            op = p[1]
            assert op not in opset_dict, "Redefinition of op " + op
            assert op in list_op_version, "Define an op " + op + " that is not listed in the ops we currently handle."
            versions = list_op_version[op]
            opset_dict[op] = ', '.join(map(lambda x: str(x), versions))
            m = max(versions)
            if m > hightest_opset:
                hightest_opset = m
            if debug:
                print("got supported op", op, "at level", list_op_version[op])
            continue
        # Limits.
        p = re.search(r'==LIM==\s+(.*)\s*$', l)
        if p is not None:
            assert op is not None, "Limit without op."
            assert op not in limit_dict, "Redefinition of limit for op " + op
            limit_dict[op] = dotted_sentence(p[1])
            if debug:
                print("Got limit for op", op, ":", limit_dict[op])
            continue
        p = re.search(r'==TODO==\s+(.*)\s*$', l)
        if p is not None:
            assert op is not None, "Todo without op."
            assert op not in todo_dict, "Redefinition of todo for op " + op
            todo_dict[op] = dotted_sentence(p[1])
            if debug:
                print("got todo for op", op, ":", todo_dict[op])
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
    print("<!---", input_command, "-->")
    # Title
    print("\n# Supported ONNX Operation for Target *" + target_arch + "*.\n")
    # Top paragraph.
    print("Onnx-mlir currently supports ONNX operations targeting up to " +
        "opset " + str(hightest_opset) + ". Limitations are listed when applicable.\n")
    print("* Operations are defined by the [ONNX Standard]" +
        "(https://github.com/onnx/onnx/blob/main/docs/Operators.md).")
    print("* Opset indicates, for each operation, the ONNX opset that " +
        "(1) last modified that operation and " +
        "(2) is supported by the current version of onnx-mlir. " +
        "For example, \"Add\" was modified in Opset 14 and carries on unmodified " +
        "to Opset 16. If onnx-mlir supports Opset 14, we thus list \"14\" as the Opset " +
        "associated with the \"Add\" operation.")
    print("\n")
    # Additional top paragraph.
    if additional_top_paragraph:
        print(additional_top_paragraph)
        print("\n")
    # Table.
    header = ["Op", "Up to Opset", "Limitations"]
    separator = ["---", "---", "---"]
    if emit_notes:
        header.append("Notes")
        separator.append("---")
    print_row(header)
    print_row(separator)
    for op in sorted(list_op_version.keys()):
        supported_op = op in opset_dict;
        if supported_op:
            info = ["**"+op+"**", opset_dict[op]]
        else:
            if not emit_unsupported:
                continue
            info = ["**"+op+"**", ""]
        if op in limit_dict:
            info.append(limit_dict[op])
        elif not supported_op:
            info.append("unsupported")
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
    global list_op_version
    debug = 0
    target_arch = "cpu"
    emit_notes = 0
    emit_unsupported = 0
    util_path = "."
    file_name = ""
    input_command = "python documentOps.py"

    try:
        opts, args = getopt.getopt(
            argv, "a:dhi:np:u", ["arch=", "debug", "help", "input=", "notes", "path=", "unsupported"])
    except getopt.GetoptError:
        print_usage()
    for opt, arg in opts:
        if opt in ("-a", "--arch"):
            target_arch = arg
            input_command += " --arch " + arg
        elif opt in ('-d', "--debug"):
            debug = 1
        elif opt in ('-h', "--help"):
            print_usage()
        elif opt in ('-i', "--input"):
            file_name = arg
            input_command += " --input " + file_name
        elif opt in ('-n', "--notes"):
            emit_notes = True
            input_command += " --notes"
        elif opt in ('-p', "--path"):
            util_path = arg
            input_command += " --path " + util_path
        elif opt in ('-u',  "--unsupported"):
            emit_unsupported = True
            input_command += " --unsupported"

    if not file_name:
        print("Command requires an input file name.\n")
        print_usage()

    # Load gen_onnx_mlir operation version.
    proc = subprocess.Popen(['python', util_path + '/gen_onnx_mlir.py', '--list-operation-version'], stdout=subprocess.PIPE)
    str = ""
    for line in  proc.stdout:
        str += line.decode("utf-8").rstrip()
    list_op_version = eval(str)
    if debug:
        print("List op version is: ", list_op_version)

    # Parse and print md table.
    parse_file(file_name)
    print_md()

if __name__ == "__main__":
    main(sys.argv[1:])
