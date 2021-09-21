#!/usr/local/bin/python3

##################### mlirAffine2Cpp.py ########################################
#
# Copyright 2021 The IBM Research Authors.
#
################################################################################
# Translate stdin affine dialect to a suitable Cpp file
#
# Generate the mlir file (better to use cannonicalize):
#   onnx-mlir-opt ... -convert-krnl-to-affine -canonicalize input.mlir >file.mlir
#
# Generate the cpp file:
#  cat file.mlir | python mlirAffine2Cpp.py | clang-format > file.cpp
#
# You can add custom cpp code using the prefix: "// COP". Note that the mlir
# variables "%name" are changed to "v_name" since MLIR allows names such as "%1".
#
# Current limitations: 
#   Ops and types can be added in the dictionaries type_dict and op_dict.
#   Alloc/alloca currently only support static types, sorry.
#
# Debug info can be added by setting "debug" to 1 or 2, see below.
#
################################################################################

import sys
import getopt
import fileinput
import re
import json
import subprocess

################################################################################
# Global def.

type_dict = {"i32": "int", "index": "int", "f32": "float"}
op_dict = {"mulf": "*", "addf": "+"}

debug = 0 # 0: none; 1: original statements; 2: details about translation. 

################################################################################
# Usage.

def print_usage():
    sys.exit()

################################################################################
# Support.

def process_map_dim_sym(line):
    map_sym_pat = re.compile(r'\(([^\)]*)\)\[([^\]]*)\]')
    definitions = map_sym_pat.findall(line)
    for d in definitions:
        (dim, sym) = d
        x = "(" + dim + ")" + "[" + sym + "]" 
        y = "(" + dim
        if dim and sym:
            y += ","
        y += sym + ")"
        if debug > 2:
            print(x, " into ", y)
        line = line.replace(x, y) 
    return line

def process_names(names):
    names = names.replace("#", "")
    names = names.replace("%", "v_")
    return names

def process_stripped_name(name):
    return process_names("%" + name)

def mlir_to_c_op(m_op):
    assert m_op in op_dict, "unsupported op "+ m_op
    return op_dict[m_op]

def mlir_to_c_type(m_type):
    assert m_type in type_dict, "unsupported type " + m_type
    return type_dict[m_type]

# return elementary type and simensions
def compute_memref_type(type_str):
    vals = type_str.split("x")
    type = mlir_to_c_type(vals.pop())
    if len(vals) == 0: # No empty sizes, at least one.
        vals = ["1"]
    dims =  "[" + "][".join(vals) + "]"
    return (type, dims)

def process_main(args):
    print("\nint main() {")
    print("// Args processed as local variables.")
    arg_pat = re.compile(r'%(\w+):\s+memref<([^>]+)>')
    definitions = arg_pat.findall(args)
    for d in definitions:
        (name, type_str) = d
        name = process_stripped_name(name)
        if debug > 1:
            print("// got arg:", name, "type:", type_str)
        (type, dims) = compute_memref_type(type_str)
        print(type + " " + name + dims + ";")
    print("// Function code.")
def process_for(name, from_val, to_val, step_val):
    if debug > 1:
        print("// got for step; name:", name, "from:", from_val, "to:", to_val, "step val:", step_val)
    # get rid of #
    name = process_names(name)
    from_val = process_names(from_val)
    to_val = process_names(to_val)
    step_val = process_names(step_val)
    print("for(int " + name + "=" + from_val + "; "  + name + "<" + to_val + "; " + name + "+=" + step_val + ") {")

def process_binary_op(name, m_op, p1, p2, m_type):
    if debug > 1:
        print("//got binary op ", m_op, "of type: ", m_type, "res:", name, "p1:", p1, "p2:", p2)
    c_type = mlir_to_c_type(m_type)
    c_op = mlir_to_c_op(m_op)
    print(c_type + " " + name + " = " + p1 + " " + c_op + " " + p2 + ";")


################################################################################
# Main function.

def main(argv):
    input_command = "mlirAffine2C.py"
    
    had_builtin = False

    print("""
#include <stdio.h>
// Support functions.
int min(int a) { return a; }
int min(int a, int b) { return a<b ? a : b; }
int min(int a, int b, int c) { return min(a, min(b, c)); }
int min(int a, int b, int c, int d) { return min(min(a, b), min(c, d)); }
int max(int a) { return a; }
int max(int a, int b) { return a>b ? a : b; }
int max(int a, int b, int c) { return max(a, max(b, c)); }
int max(int a, int b, int c, int d) { return max(max(a, b), max(c, d)); }
// map support, if any""")

    for line in sys.stdin:
        line = line.rstrip()

        # Print line if requested.
        if debug > 1: print("")
        if debug > 0:
            res = re.match(r'\s*(.*)', line)
            if res is not None:
                print("//", res.group(1))
        if debug > 1: print("")

        # Strip "(d1, d2)[s1, s2]"" into more friendly "(d1, d2, s1, s2)"."
        line = process_map_dim_sym(line)

        # Process affine map.
        map_pat = re.compile(r'\#(\w*) = affine_map<\(([^\)]*)\) -> ([^>]*)>')
        res = map_pat.match(line)
        if res is not None:
            # There are no percent in names in affine maps.
            map_name = res.group(1)
            param = res.group(2)
            body = res.group(3)
            if debug > 1:
                print("// got affine map; name:", map_name, "param:", param, "body:", body)
            print("#define " + map_name + "(" + param + ") " + body)
            continue

        # Process module header.
        main_pat = re.compile(r'\s*builtin.module')
        if main_pat.match(line) is not None:
            had_builtin = True
            print("namespace test {")
            continue

        # Process function header.
        main_pat = re.compile(r'\s+builtin.func[^\(]+\(([^\)]+)\)')
        res = main_pat.match(line)
        if res is not None:
            args = res.group(1)
            if debug > 1:
                print("// got func with args:", args)
            process_main(args)
            continue

        # Process constant.
        const_pat = re.compile(r'\s+%(\w+)\s+=\s+constant\s+([^:]+):\s+(.*)')
        res = const_pat.match(line) 
        if res is not None:
            name = process_stripped_name(res.group(1))
            val = process_names(res.group(2))
            type = mlir_to_c_type(res.group(3))
            if debug > 1:
                print("// got const; name:", name, "val:", val, "type:", type)
            print(type, name, "=", val, ";")
            continue

        # Process alloc/alloca: treat both the same.
        alloc_pat = re.compile(r'\s+%(\w+)\s+=\s+memref\.alloc[a]?\(([^\)]*)\)\s*(\{[^\}]*\}\s+)?:\s+memref<([^>]+)>')
        res = alloc_pat.match(line) 
        if res is not None:
            name = process_stripped_name(res.group(1))
            args = process_names(res.group(2))
            type_str = res.group(4)
            if debug > 1:
                print("// got alloc(a); name:", name, "args:", args, "memref type:", type_str)
            (type, dims) = compute_memref_type(type_str)
            print(type + " " + name + dims + ";")
            continue

        # Process affine for.
        for_pat = re.compile(r'\s+affine\.for\s+%(\w+)\s+=\s+(.*)\s+to\s+(.*)\s+\{')
        for_step_pat = re.compile(r'\s+affine\.for\s+%(\w+)\s+=\s+(.*)\s+to\s+(.*)\s+step\s+(\d+)\s+\{')        
        res = for_step_pat.match(line) 
        if res is not None:
            process_for("%"+res.group(1), res.group(2), res.group(3), res.group(4))
            continue
        res = for_pat.match(line) 
        if res is not None:
            process_for("%"+res.group(1), res.group(2), res.group(3), "1")
            continue

        # Process affine apply.
        apply_pat = re.compile(r'\s+%(\w+)\s+=\s+affine\.apply\s+(.*)')
        res = apply_pat.match(line) 
        if res is not None:
            name = process_stripped_name(res.group(1))
            val = process_names(res.group(2))
            if debug > 1:
                print("// got apply; name:", name, "value:", val)
            print("int " + name + " = " + val + ";")
            continue

        # Process affine min/max.
        apply_pat = re.compile(r'\s+%(\w+)\s+=\s+affine\.(min|max)\s+(.*)')
        res = apply_pat.match(line) 
        if res is not None:
            name = process_stripped_name(res.group(1))
            op = res.group(2)
            val = process_names(res.group(3))
            if debug > 1:
                print("// got min/max; name:", name, "op:", op, "value:", val)
            print("int " + name + " = " + op + " " + val + ";")
            continue

        # Process affine load.
        load_pat = re.compile(r'\s+%(\w+)\s+=\s+affine\.load\s+%(\w+)(\[[^\]]*\])\s+:\s+memref<(.*)>')
        res = load_pat.match(line) 
        if res is not None:
            name = process_stripped_name(res.group(1))
            array = process_stripped_name(res.group(2))
            addr = process_names(res.group(3))
            memref = res.group(4)
            if debug > 1:
                print("// got load; val:", name, "array:", array, "addr", addr, "memref:", memref)
            (type, dims) = compute_memref_type(memref)
            addr = addr.replace(",", "][") # Transform separators in multi-dim array ref.
            addr = addr.replace("[]", "[0]") # No empty "array[]", want "array[0]"
            print(type + " " + name + "=" + array + addr + ";")
            continue

        # Process affine store.
        load_pat = re.compile(r'\s+affine\.store\s+%(\w+)\s*,\s*%(\w+)(\[[^\]]*\])')
        res = load_pat.match(line) 
        if res is not None:
            name = process_stripped_name(res.group(1))
            array = process_stripped_name(res.group(2))
            addr = process_names(res.group(3))
            if debug > 1:
                print("// got store; val:", name, "array:", array, "addr", addr)
            addr = addr.replace(",", "][") # Transform separators in multi-dim array ref.
            addr = addr.replace("[]", "[0]") # No empty "array[]", want "array[0]"
            print(array + addr + "=" + name + ";")
            continue

        # Process binary op.
        binary_pat = re.compile(r'\s+%(\w+)\s+=\s+(\w+)\s+%(\w+)\s*,\s*%(\w+)\s+:\s+(\w+)')
        res = binary_pat.match(line)
        if res is not None:
            name = process_stripped_name(res.group(1))
            mop = res.group(2)
            p1 = process_stripped_name(res.group(3))
            p2 = process_stripped_name(res.group(4))
            m_type = res.group(5)
            process_binary_op(name, mop, p1, p2, m_type)
            continue

        # Process C code placed in the mlir code as comment.
        c_code_pat = re.compile(r'\s+\/\/\s*COP\s+(.*)')
        res = c_code_pat.match(line)
        if res is not None:
            stmt = res.group(1)
            if debug > 1:
                print("//got c stmt:"+stmt)
            print(stmt)
            continue

        # Process end of structure.
        if re.match(r'\s*\}', line):
            print("}")
            continue

        # Process return.
        return_pat = re.compile(r'\s+return\s+%(\w+)')
        res = return_pat.match(line)
        if res is not None:
            name = process_stripped_name(res.group(1))
            if debug > 1:
                print("// got return of val:", name)
            print("return 0;")
            continue

        # Process end of program.
        if re.match(r'.*krnl\.entry_point', line):
            #exit(1)
            continue

        # Generate a warning for non-empty unprocessed lines.
        res = re.match(r'\s*(.*)', line)
        if res is not None:
            stmt = res.group(1)
            if stmt:
                print("#warning following stmt not processed:", stmt)
    
    # Generate call to main
    if had_builtin:
        print("int main() { return test::main(); }")

if __name__ == "__main__":
    main(sys.argv[1:])
