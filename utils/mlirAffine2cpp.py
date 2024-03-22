#!/usr/local/bin/python3

##################### mlirAffine2Cpp.py ########################################
#
# Copyright 2021 The IBM Research Authors.
#
################################################################################
# Translate stdin affine dialect to a suitable Cpp file
#
# Generate the mlir file (better to use canonicalize):
#   onnx-mlir-opt ... -convert-krnl-to-affine -canonicalize input.mlir >file.mlir
#   If you have maps, please also run --normalize-memrefs
#
# Generate the cpp file:
#  cat file.mlir | python mlirAffine2Cpp.py | clang-format > file.cpp
#
# You can add custom cpp code using the prefix: "// COP". Note that the mlir
# variables "%name" are changed to "v_name" since MLIR allows names such as "%1".
#
# Support it given to initialize and print arrays, so as to have meaningful
# computations. For example, below are some "// COP" comments that could be
# added to an .mlir file prior to be translated to C++
#
#     builtin.func @main_graph(%arg0: memref<2x3x4x5xf32>, %arg1: memref<3xf32>, %arg2: memref<3xf32>) -> memref<2x3x4x5xf32> attributes {input_names = ["x", "s", "bias"], output_names = ["y"]} {
#       // COP init4((float *)v_arg0, 2, 3, 4, 5, 0);
#       // COP init1((float *)v_arg1, 3, 10);
#       // COP init1((float *)v_arg2, 3, 100);
#       ...
#       affine.for %arg3 = 0 to 2 {
#          affine.for %arg4 = 0 to 3 {
#            ...
#            %5 = affine.load %4[] : memref<f32>
#            %6 = divf %5, %2 : f32
#            // COP printf("%lld %lld: sum %f mean %f\n", v_arg3, v_arg4, v_5, v_6);
#
# Current limitations:
#   Ops and types can be added in the dictionaries type_dict, binary_op_dict,
#   and unary_op_dict.
#
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

type_dict = {
    "i64": "long long",
    "i32": "int",
    "i1": "int",
    "index": "long long",
    "f32": "float",
    "f64": "double",
}
binary_op_dict = {
    "mulf": "*",
    "addf": "+",
    "subf": "-",
    "divf": "/",
    "muli": "*",
    "addi": "+",
    "subi": "-",
    "floordivsi": "/",
    "remsi": "%",
    "cmp_eq": "==",
    "cmp_slt": "<",
}
unary_op_dict = {"math.sqrt": "sqrt"}

debug = 0  # 0: none; 1: original statements; 2: details about translation.

################################################################################
# Usage.


def print_usage():
    sys.exit()


################################################################################
# Python code to init arrays. To use in a jupyter notebook to match the init
# that can be added the the compiled code.

import numpy as np


def init_array(shape, start):
    return np.reshape(np.arange(start, start + np.prod(shape), dtype=np.float32), shape)


################################################################################
# Support.


def process_map_dim_sym(line):
    map_sym_pat = re.compile(r"\(([^\)]*)\)\[([^\]]*)\]")
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


def process_compare(line):
    # First, space between "cmpi" and "eq" is a problem, replace by "_".
    # Second, format of binary op don't have a "," after operator, remove.
    line = re.sub(r"cmpi\s+(eq|slt)\s*,", r"cmp_\1", line)
    return line


def process_names(names):
    names = names.replace("#", "")
    names = names.replace("%", "v_")
    return names


def process_stripped_name(name):
    return process_names("%" + name)


def mlir_to_c_binary_op(m_op):
    assert m_op in binary_op_dict, "unsupported op " + m_op
    return binary_op_dict[m_op]


def mlir_to_c_unary_op(m_op):
    assert m_op in unary_op_dict, "unsupported op " + m_op
    return unary_op_dict[m_op]


def mlir_to_c_type(m_type):
    assert m_type in type_dict, "unsupported type " + m_type
    return type_dict[m_type]


# return elementary type and dimensions
def compute_memref_type(type_str):
    type_str = type_str.replace("index", "inde")  # Remove x because of next step.
    vals = type_str.split("x")
    type = vals.pop()
    type = type.replace("inde", "index")  # add x again in index.
    type = mlir_to_c_type(type)
    if len(vals) == 0:  # No empty sizes, at least one.
        vals = ["1"]
    dims = "[" + "][".join(vals) + "]"
    return (type, dims)


def process_main(args):
    print("\nint main() {")
    print("// Args processed as local variables.")
    arg_pat = re.compile(r"%(\w+):\s+memref<([^>]+)>")
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
        print(
            "// got for step; name:",
            name,
            "from:",
            from_val,
            "to:",
            to_val,
            "step val:",
            step_val,
        )
    # get rid of #
    name = process_names(name)
    from_val = process_names(from_val)
    to_val = process_names(to_val)
    step_val = process_names(step_val)
    print(
        "for(long long "
        + name
        + "="
        + from_val
        + "; "
        + name
        + "<"
        + to_val
        + "; "
        + name
        + "+="
        + step_val
        + ") {"
    )


def process_binary_op(name, m_op, p1, p2, m_type):
    if debug > 1:
        print(
            "//got binary op ",
            m_op,
            "of type: ",
            m_type,
            "res:",
            name,
            "p1:",
            p1,
            "p2:",
            p2,
        )
    c_type = mlir_to_c_type(m_type)
    c_op = mlir_to_c_binary_op(m_op)
    print(c_type + " " + name + " = " + p1 + " " + c_op + " " + p2 + ";")


def process_unary_op(name, m_op, p1, m_type):
    if debug > 1:
        print("//got binary op ", m_op, "of type: ", m_type, "res:", name, "p1:", p1)
    c_type = mlir_to_c_type(m_type)
    c_op = mlir_to_c_unary_op(m_op)
    print(c_type + " " + name + " = " + c_op + "(" + p1 + ");")


def process_conversion(name, m_op, p1, m_type_from, m_type_to):
    if debug > 1:
        print(
            "//got conversion op ",
            m_op,
            "from type: ",
            m_type_from,
            "to type: ",
            m_type_to,
            "res:",
            name,
            "p1:",
            p1,
        )
    c_type_to = mlir_to_c_type(m_type_to)
    print(c_type_to + " " + name + " = " + p1 + ";")


################################################################################
# Main function.


def main(argv):
    input_command = "mlirAffine2C.py"

    had_builtin = False

    print(
        """
#include <math.h>
#include <stdio.h>
#include <string>

// Support functions for min/max.
long long min(long long a) { return a; }
long long min(long long a, long long b) { return a<b ? a : b; }
long long min(long long a, long long b, long long c) { return min(a, min(b, c)); }
long long min(long long a, long long b, long long c, long long d) { return min(min(a, b), min(c, d)); }
long long max(long long a) { return a; }
long long max(long long a, long long b) { return a>b ? a : b; }
long long max(long long a, long long b, long long c) { return max(a, max(b, c)); }
long long max(long long a, long long b, long long c, long long d) { return max(max(a, b), max(c, d)); }

// Support functions for init.
void init1(float *a, long long d0, float v) {
    for(long long i0=0; i0<d0; ++i0)
      a[i0] = ++v;
}
void init2(float *a, long long d0, long long d1, float v) {
    for(long long i0=0; i0<d0; ++i0)
      for(long long i1=0; i1<d1; ++i1)
        a[i1 + d1 * i0] = ++v;
}
void init3(float *a, long long d0, long long d1, long long d2, float v) {
    for(long long i0=0; i0<d0; ++i0)
      for(long long i1=0; i1<d1; ++i1)
        for(long long i2=0; i2<d2; ++i2)
          a[i2 + d2 * (i1 + d1 * i0)] = ++v;
}
void init4(float *a, long long d0, long long d1, long long d2, long long d3, float v) {
    for(long long i0=0; i0<d0; ++i0)
      for(long long i1=0; i1<d1; ++i1)
        for(long long i2=0; i2<d2; ++i2)
          for(long long i3=0; i3<d3; ++i3)
            a[i3 + d3 * (i2 + d2 * (i1 + d1 * i0))] = ++v;
}

// Support functions for print.
void print1(std::string msg, float *a, long long d0) {
    printf("%s with dims %lldxf32\\n", msg.c_str(), d0);
    for(long long i0=0; i0<d0; ++i0)
      printf("%s, %3lld, %f\\n", msg.c_str(), i0, a[i0]);
}
void print2(std::string msg, float *a, long long d0, long long d1) {
    printf("%s with dims %lldx%lldxf32\\n", msg.c_str(), d0, d1);
    for(long long i0=0; i0<d0; ++i0)
      for(long long i1=0; i1<d1; ++i1)
        printf("%s, %3lld, %3lld, %f\\n", msg.c_str(), i0, i1, a[i1 + d1 * i0]);
}
void print3(std::string msg, float *a, long long d0, long long d1, long long d2) {
    printf("%s with dims %lldx%lldx%lldxf32\\n", msg.c_str(), d0, d1, d2);
    for(long long i0=0; i0<d0; ++i0)
      for(long long i1=0; i1<d1; ++i1)
        for(long long i2=0; i2<d2; ++i2)
          printf("%s, %3lld, %3lld, %3lld, %f\\n", msg.c_str(), i0, i1, i2, a[i2 + d2 * (i1 + d1 * i0)]);
}
void print4(std::string msg, float *a, long long d0, long long d1, long long d2, long long d3) {
    printf("%s with dims %lldx%lldx%lldx%lldxf32\\n", msg.c_str(), d0, d1, d2, d3);
    for(long long i0=0; i0<d0; ++i0)
      for(long long i1=0; i1<d1; ++i1)
        for(long long i2=0; i2<d2; ++i2)
          for(long long i3=0; i3<d3; ++i3)
            printf("%s, %3lld, %3lld, %3lld, %3lld, %f\\n", msg.c_str(), i0, i1, i2, i3, a[i3 + d3 * (i2 + d2 * (i1 + d1 * i0))]);
}

// Map support, if any.
"""
    )

    for line in sys.stdin:
        line = line.rstrip()

        # Print line if requested.
        if debug > 1:
            print("")
        if debug > 0:
            res = re.match(r"\s*(.*)", line)
            if res is not None:
                print("//", res.group(1))
        if debug > 1:
            print("")

        # remove "arith."
        line = line.replace("arith.", "")
        line = line.replace("func.func", "builtin.func")
        # Strip "(d1, d2)[s1, s2]"" into more friendly "(d1, d2, s1, s2)"."
        line = process_map_dim_sym(line)
        line = process_compare(line)

        # Process affine map.
        map_pat = re.compile(r"\#(\w*) = affine_map<\(([^\)]*)\) -> ([^>]*)>")
        res = map_pat.match(line)
        if res is not None:
            # There are no percent in names in affine maps.
            map_name = res.group(1)
            param = res.group(2)
            body = res.group(3)
            if debug > 1:
                print(
                    "// got affine map; name:", map_name, "param:", param, "body:", body
                )
            print("#define " + map_name + "(" + param + ") " + body)
            continue

        # Process module header.
        main_pat = re.compile(r"\s*builtin.module")
        if main_pat.match(line) is not None:
            had_builtin = True
            print("namespace test {")
            continue

        # Process function header.
        main_pat = re.compile(r"\s+builtin.func[^\(]+\(([^\)]+)\)")
        res = main_pat.match(line)
        if res is not None:
            args = res.group(1)
            if debug > 1:
                print("// got func with args:", args)
            process_main(args)
            continue

        # Process constant bool true/false.
        const_bool_pat = re.compile(r"\s+%(\w+)\s+=\s+constant\s+(true|false)")
        res = const_bool_pat.match(line)
        if res is not None:
            name = process_stripped_name(res.group(1))
            val = res.group(2)
            if val == "true":
                print("int", name, " = 1;")
            else:
                print("int", name, " = 0;")
            if debug > 1:
                print("// got const; name:", name, "val:", val, "type: bool")

            continue
        # Process other constant.
        const_pat = re.compile(r"\s+%(\w+)\s+=\s+constant\s+([^:]+):\s+(.*)")
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
        alloc_pat = re.compile(
            r"\s+%(\w+)\s+=\s+memref\.alloc[a]?\(([^\)]*)\)\s*(\{[^\}]*\}\s+)?:\s+memref<([^>]+)>"
        )
        res = alloc_pat.match(line)
        if res is not None:
            name = process_stripped_name(res.group(1))
            args = process_names(res.group(2))
            type_str = res.group(4)
            if debug > 1:
                print(
                    "// got alloc(a); name:",
                    name,
                    "args:",
                    args,
                    "memref type:",
                    type_str,
                )
            (type, dims) = compute_memref_type(type_str)
            print(type + " " + name + dims + ";")
            continue

        # Process affine for.
        for_pat = re.compile(r"\s+affine\.for\s+%(\w+)\s+=\s+(.*)\s+to\s+(.*)\s+\{")
        for_step_pat = re.compile(
            r"\s+affine\.for\s+%(\w+)\s+=\s+(.*)\s+to\s+(.*)\s+step\s+(\d+)\s+\{"
        )
        res = for_step_pat.match(line)
        if res is not None:
            process_for("%" + res.group(1), res.group(2), res.group(3), res.group(4))
            continue
        res = for_pat.match(line)
        if res is not None:
            process_for("%" + res.group(1), res.group(2), res.group(3), "1")
            continue

        # Process affine apply.
        apply_pat = re.compile(r"\s+%(\w+)\s+=\s+affine\.apply\s+(.*)")
        res = apply_pat.match(line)
        if res is not None:
            name = process_stripped_name(res.group(1))
            val = process_names(res.group(2))
            if debug > 1:
                print("// got apply; name:", name, "value:", val)
            print("long long " + name + " = " + val + ";")
            continue

        # Process affine min/max.
        apply_pat = re.compile(r"\s+%(\w+)\s+=\s+affine\.(min|max)\s+(.*)")
        res = apply_pat.match(line)
        if res is not None:
            name = process_stripped_name(res.group(1))
            op = res.group(2)
            val = process_names(res.group(3))
            if debug > 1:
                print("// got min/max; name:", name, "op:", op, "value:", val)
            print("long long " + name + " = " + op + " " + val + ";")
            continue

        # Process affine/memref load.
        load_pat = re.compile(
            r"\s+%(\w+)\s+=\s+(memref|affine)\.load\s+%(\w+)(\[[^\]]*\])\s+:\s+memref<(.*)>"
        )
        res = load_pat.match(line)
        if res is not None:
            name = process_stripped_name(res.group(1))
            array = process_stripped_name(res.group(3))
            addr = process_names(res.group(4))
            memref = res.group(5)
            if debug > 1:
                print(
                    "// got load; val:",
                    name,
                    "array:",
                    array,
                    "addr",
                    addr,
                    "memref:",
                    memref,
                )
            (type, dims) = compute_memref_type(memref)
            addr = addr.replace(
                ",", "]["
            )  # Transform separators in multi-dim array ref.
            addr = addr.replace("[]", "[0]")  # No empty "array[]", want "array[0]"
            print(type + " " + name + "=" + array + addr + ";")
            continue

        # Process affine store.
        load_pat = re.compile(
            r"\s+(memref|affine)\.store\s+%(\w+)\s*,\s*%(\w+)(\[[^\]]*\])"
        )
        res = load_pat.match(line)
        if res is not None:
            name = process_stripped_name(res.group(2))
            array = process_stripped_name(res.group(3))
            addr = process_names(res.group(4))
            if debug > 1:
                print("// got store; val:", name, "array:", array, "addr", addr)
            addr = addr.replace(
                ",", "]["
            )  # Transform separators in multi-dim array ref.
            addr = addr.replace("[]", "[0]")  # No empty "array[]", want "array[0]"
            print(array + addr + "=" + name + ";")
            continue

        # process scf.if
        scf_if_pat = re.compile(r"\s+scf\.if\s+%(\w+)")
        res = scf_if_pat.match(line)
        if res is not None:
            name = process_stripped_name(res.group(1))
            if debug > 1:
                print("// got scf.if, cond=", name)
            print("if (", name, ") {")
            continue

        # Process conversion op (unary ":" type "to" type)
        convert_pat = re.compile(
            r"\s+%(\w+)\s+=\s+([\w\.]+)\s+%(\w+)\s*:\s+(\w+)\s+to\s+(\w+)"
        )
        res = convert_pat.match(line)
        if res is not None:
            name = process_stripped_name(res.group(1))
            mop = res.group(2)
            p1 = process_stripped_name(res.group(3))
            m_type_from = res.group(4)
            m_type_to = res.group(5)
            process_conversion(name, mop, p1, m_type_from, m_type_to)
            continue

        # Process binary op.
        binary_pat = re.compile(
            r"\s+%(\w+)\s+=\s+([\w\.]+)\s+%(\w+)\s*,\s*%(\w+)\s+:\s+(\w+)"
        )
        res = binary_pat.match(line)
        if res is not None:
            name = process_stripped_name(res.group(1))
            mop = res.group(2)
            p1 = process_stripped_name(res.group(3))
            p2 = process_stripped_name(res.group(4))
            m_type = res.group(5)
            process_binary_op(name, mop, p1, p2, m_type)
            continue

        # Process select op.
        # select_pat = re.compile(r'\s+%(\w+)\s+=\s+select\s+%(\w+)\s*,\s*%(\w+)\s*,\s*%(\w+)\s+:\s+(\w+)')
        select_pat = re.compile(
            r"\s+%(\w+)\s+=\s+select\s+%(\w+)\s*,\s*%(\w+)\s*,\s*%(\w+)\s+:\s+(\w+)"
        )
        res = select_pat.match(line)
        if res is not None:
            name = process_stripped_name(res.group(1))
            p1 = process_stripped_name(res.group(2))
            p2 = process_stripped_name(res.group(3))
            p3 = process_stripped_name(res.group(4))
            ctype = mlir_to_c_type(res.group(5))
            if debug > 1:
                print(
                    "// got select dest",
                    name,
                    ", comp",
                    p1,
                    ", args",
                    p2,
                    p3,
                    ", and type",
                    ctype,
                )
            print(ctype, name, "= (", p1, ") ? ", p2, ":", p3, ";")
            continue

        # Process unary op.
        binary_pat = re.compile(r"\s+%(\w+)\s+=\s+([\w\.]+)\s+%(\w+)\s*:\s+(\w+)")
        res = binary_pat.match(line)
        if res is not None:
            name = process_stripped_name(res.group(1))
            mop = res.group(2)
            p1 = process_stripped_name(res.group(3))
            m_type = res.group(4)
            process_unary_op(name, mop, p1, m_type)
            continue

        # Process C code placed in the mlir code as comment.
        c_code_pat = re.compile(r"\s+\/\/\s*COP\s+(.*)")
        res = c_code_pat.match(line)
        if res is not None:
            stmt = res.group(1)
            if debug > 1:
                print("//got c stmt:" + stmt)
            print(stmt)
            continue

        # Process end of structure.
        if re.match(r"\s*\}", line):
            print("}")
            continue

        # Process return.
        return_pat = re.compile(r"\s+return\s+%(\w+)")
        res = return_pat.match(line)
        if res is not None:
            name = process_stripped_name(res.group(1))
            if debug > 1:
                print("// got return of val:", name)
            print("return 0;")
            continue

        # Process end of program.
        if re.match(r".*krnl\.entry_point", line):
            # exit(1)
            continue

        # Generate a warning for non-empty unprocessed lines.
        res = re.match(r"\s*(.*)", line)
        if res is not None:
            stmt = res.group(1)
            if stmt:
                print("//#warning following stmt not processed:", stmt)

    # Generate call to main
    if had_builtin:
        print("int main() { return test::main(); }")


if __name__ == "__main__":
    main(sys.argv[1:])
