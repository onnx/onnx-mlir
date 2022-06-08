#!/usr/local/bin/python3

# SPDX-License-Identifier: Apache-2.0

##################### documentOps.py ########################################
#
# Copyright 2022 The IBM Research Authors.
#
################################################################################
#
# This file convert .mlir from stdin into a FileCheck file format. It also
# performs renaming of variables for better readability. In debug mode, it
# can be used to simply better read mlir files as variables have more
# comprehensive names
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
#   where <text> qualifies the opset currently being supported. When "current" is 
#   provided, the postprocessing will automatically changed highest opset currently
#   supported. When no <text> is provided, the operation is assumed to be fully 
#   unsupported.
#
# ==LIM== <text> 
#   where <text> qualifies the current restrictions to the implementation.
#
# ==TODO== <text>
#   where <text> add "private" info about what needs to be fixed. 
#
# egrep pattern: (script automatically ignores any non-labels anyway).
#   egrep "==ARCH==|==OP==|==LIM==|==TODO==" 
#
################################################################################
# Usage.

def print_usage():
    print('\nGenerate MD document tables for the supported ops using the labeling left in files.')
    print("For labeling format, consult the python script directly.")
    print('documentOps [-a <arch>] [-dut] -i file>')
    print('  -a, --arch <arch>: report on "==ARCH== <arch>".')
    print('  -d, --debug: include debug.')
    print('  -i, --input <file name>: input file.')
    print('  -u, --unsupported: list unsupported ops.')
    print('  -t, --todo: include todos.')
    sys.exit()

################################################################################
# Handling of info: global dictionaries.

current_opset = "16"   # Opset to substitute when opset is "current".
opset_dict = {}        # <op> -> <text> in "==OP== <op> <text>".
limit_dict = {}        # <op> -> <text> in "==LIM== <text>".
todo_dict = {}         # <op> -> <text> in "==TODO== <text>".
list_op_version = {}   # List of operation versions from gen_onnx_mlir;
                       # <op> -> [supported versions]
              
################################################################################
# Parse input file. Add only info if it is the proper target arch. Other entries
# and non-relevant data is simply ignored. At this time, does not support 
# multiple entries of any kind. Everything is case sensitive.

def dotted_sentence(str):
    if re.match(r'.*\.\s*$', str) is None:
        return str + "."
    return str

def parse_file(file_name):
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
        # Scan unsupported op (op followed by spaces only).
        p = re.search(r'==OP==\s+(\w+)\s*$', l)
        if p is not None: 
            op = p[1]
            assert op not in opset_dict, "Redefinition of op " + op
            opset_dict[op] = "unsupported"
            if debug:
                print("got unsupported op", op)
            continue
        # Supported op.
        p = re.search(r'==OP==\s+(\w+)\s+(.*)\s*$', l)
        if p is not None: 
            op = p[1]
            assert op not in opset_dict, "Redefinition of op " + op
            #if (p[2] == "current"):
            #    opset_dict[op] = -1
            #else:
            #    opset_dict[op] = p[2]
            if op in list_op_version:
                print("hi alex,", list_op_version[op])
                opset_dict[op] = ', '.join(map(lambda x: str(x), list_op_version[op]))
            elif debug or True:
                print("Got supported op", op, "at level", opset_dict[op],
                    "without list_op_version")
            if debug:
                print("Got supported op", op, "at level", opset_dict[op])
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
# Print info

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
    print("Onnx-mlir currently support ONNX operations targeting " + 
      "opset " + current_opset + ". Limitations are listed when applicable.\n")
    # Table.
    header = ["Op", "Opset", "Limitations"]
    separator = ["---", "---", "---"]
    if emit_todos:
        header.append("Todo")
        separator.append("---")
    print_row(header)
    print_row(separator)
    for op in sorted(opset_dict.keys()):
        if not emit_unsupported and opset_dict[op] == "unsupported":
            continue
        info = ["**"+op+"**", opset_dict[op]]
        if op in limit_dict:
            info.append(limit_dict[op])
        else:
            info.append("")
        if emit_todos:
            if op in todo_dict:
                info.append(todo_dict[op])
            else:
                info.append("")
        print_row(info)

        
def main(argv):
    global debug, target_arch, emit_todos, emit_unsupported, input_command
    global list_op_version
    debug = 0
    target_arch = "cpu"
    emit_todos = 0
    emit_unsupported = 0
    file_name = ""
    input_command = "python documentOps.py"

    try:
        opts, args = getopt.getopt(
            argv, "a:dhi:tu", ["arch=", "debug", "help", "input=", "todo", "unsupported"])
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
        elif opt in ('-t', "--todo"):
            emit_todos = True
            input_command += " --todo"
        elif opt in ('-u', "--unsupported"):
            emit_unsupported = True
            input_command += " --unsupported"

    if not file_name:
        print("Command requires an input file name.\n")
        print_usage()

    # Load gen_onnx_mlir operation version.
    proc = subprocess.Popen(['python', 'gen_onnx_mlir.py', '--list-operation-version'], stdout=subprocess.PIPE)
    str = ""
    for line in  proc.stdout:
        str += line.decode("utf-8").rstrip()
    list_op_version = eval(str)
    if debug:
        print("List op version is: ", list_op_version)
    parse_file(file_name)
    print_md()

if __name__ == "__main__":
    main(sys.argv[1:])
