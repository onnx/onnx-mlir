#!/usr/local/bin/python3

##################### mlir2FileCheck.py ########################################
#
# Copyright 2020 The IBM Research Authors.
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
# Usage.


def print_usage():
    print('Translate mlir format from stdin to a suitable FileCheck format.')
    print('mlir2FileCheck [-cdh] [-a <arg arrays>] [-d <dict>]')
    print('  -a,--args <array>: Rename function arguments using a json array,')
    print(
        '                     such as \'["A", "B", "C"] for 1st, 2nd, & 3rd args.\'.')
    print('  -c,--check       : Run FileCheck on output, to verify that the output is good.')
    print('  -d,-debug        : Rename for easier debugging of code, disable FileCheck format.')
    print('  -h,--help        : Print help.')
    print('  -n,--names <dict>: Rename variables using a json dictionary')
    print(
        '                     such as \'{"cst": "ZERO"}\' to rename cst to ZERO.')
    sys.exit()


################################################################################
# Process a def-use chain.

# Associate a color with each line; two lines with the same color are independent
# and thus are eligible to use a CHECK-DAG.

def process_def_use_chains(line):
    global def_set, line_color, curr_color
    def_qual_pat = re.compile(r'\s+%([a-zA-Z0-9][a-zA-Z0-9_\-]*)(:\d+)?\s+=')
    definition = def_qual_pat.match(line)
    if not definition:
        # Do not have a "x = ...", disable DAG.
        def_set = []
        curr_color += 1
        line_color.append(curr_color)
        #print(curr_color, ":", line, "// no def, no DAG")
        curr_color += 1
        return
    # Has a def, add to def set.
    curr_def = definition.group(1)
    use_qual_pat = re.compile(r'%([a-zA-Z0-9][a-zA-Z0-9_\-]*)(:\d+)?')
    uses = use_qual_pat.findall(line)
    for u in uses:
        # Var u is and array with var [name, qualifier]. Look at name only.
        if u[0] in def_set:
            # Use found in def set, disable DAG
            def_set = [curr_def]
            curr_color += 1
            line_color.append(curr_color)
            #print(curr_color, ":", line, "// found use", u[0], "in defs")
            return
    def_set.append(curr_def)
    line_color.append(curr_color)
    #print(curr_color, ":", line, "// all good")
    return

################################################################################
# Handling of names: global dictionaries.


prepare_name_dict = {}  # orig_name -> new_name (with ref count)
name_dict = {}  # orig_name -> new_name (with ref count)
refcount_dict = {}  # new_name (without ref count) -> definition count

# Prepare name for def, for mapping given by the user.


def prepare_name_def(orig_name, new_name):
    global prepare_name_dict
    if orig_name in prepare_name_dict.keys():
        print("multiple definitions of original name in prepare:", orig_name)
        exit()
    prepare_name_dict[orig_name] = new_name

# Process name and make it a legit def for FileCheck.


def def_string(name, num):
    if debug:
        return name + "_" + num
    return "[[" + name + "_:%.+]]" + num

# Record an original name/number, with a suggested new name. Append_str is
# added at the end (to match the string looked for and to be substituted).


def record_name_def(orig_name, orig_num, new_name, append_str, num_for_zero, line):
    global name_dict, refcount_dict
    # original name: what we found in the text
    # new_name: if something was prepared for it, use the prepared name. But
    # that is not the case, use the new name.
    if orig_name in name_dict.keys():
        # When prepare, we add "orig_name -> new_name".
        if not new_name in refcount_dict.keys():
            # A name is already associated, but without a refcount; it was a prepare.
            # We will use the pre-associated name.
            new_name = name_dict[orig_name]
            num_for_zero = 0
        # else:
            # A name is already associated, with a refcount; it was a
            # normal use; do nothing special.
            #print("/// warning: name", orig_name, "is redefined at line", line)

    if new_name in refcount_dict.keys():
        # It is, increment the count.
        refcount = refcount_dict[new_name]
        refcount_dict[new_name] = refcount + 1
        # Append the count to the name.
        curr_name = new_name + "_" + str(refcount)
        # Map the binding old -> current.
        name_dict[orig_name] = curr_name
        return def_string(curr_name, orig_num) + append_str
    # Record the mapping of a first def.
    refcount_dict[new_name] = 1
    curr_name = new_name
    if num_for_zero:
        curr_name += "_0"
    name_dict[orig_name] = curr_name
    return def_string(curr_name, orig_num) + append_str

# Translate a FileCheck use of a pattern.


def translate_use_name(orig_name, orig_num, append_str):
    global name_dict, refcount_dict
    if orig_name in name_dict.keys():
        # Mapping found, return this.
        orig_name = name_dict[orig_name]
    if debug:
        return orig_name + "_" + orig_num + append_str
    return "[[" + orig_name + "_]]" + orig_num + append_str

# Pattern that we are substituting from. To is either obtained by
# record_name_def or translate_use_name


def use_name(orig_name, orig_num, append_str):
    return "%" + orig_name + orig_num + append_str


################################################################################
# process a line

# Process the substitution for a whole line for the given pattern.


def process_name(new_line, pattern, default_name, append_str, num_for_zero):
    definitions = pattern.findall(new_line)
    for d in definitions:
        (name, num) = d
        x = use_name(name, num, append_str)
        y = record_name_def(name, num, default_name,
                            append_str, num_for_zero, new_line)
        new_line = new_line.replace(x, y)
    return new_line


# Drive the processing of the current line.
def process_line(i, line):
    global debug, check, prepare_name_dict, name_dict, refcount_dict
    global line_color, curr_parallel_color
    def_arg_pat = re.compile(r'%([a-zA-Z0-9][a-zA-Z0-9_\-]*)():')
    def_qual_pat = re.compile(r'%([a-zA-Z0-9][a-zA-Z0-9_\-]*)(:\d+)?\s+=')
    def_pat = re.compile(r'%([a-zA-Z0-9][a-zA-Z0-9_\-]*)()\s+=')
    def_op_pat = re.compile(r'=\s+(?:\w+\.)?(\w+)')
    use_qual_pat = re.compile(r'%([a-zA-Z0-9][a-zA-Z0-9_\-]*)(:\d+)?')
    # If we have an affine map, print it now unmodified.
    if re.match(r'#\w+\s=\saffine_(map|set)<.*>', line) is not None:
        print("// CHECK-DAG:", line)
        return
    # Keep only function related text, which start by def with a space.
    if re.match(r'\s+', line) is None:
        return
    new_line = line

    # Process definition of variables.
    # Special handling of function header.
    if re.match(r'\s+func', line) is not None:
        # Have a function: reset dictionary and ref counts
        name_dict = prepare_name_dict.copy()
        refcount_dict.clear()
        #print("/// reset dict with dic", name_dict, "refcount", refcount_dict)
        new_line = process_name(new_line, def_arg_pat, "PARAM", ":", 1)
    # Special handling of loop iterations.
    elif re.match(r'\s+(\w+\.)?for', line) is not None:
        new_line = process_name(new_line, def_pat, "I", " =", 1)
    elif re.search(r'krnl\.define_loops', line) is not None:
        new_line = process_name(new_line, def_qual_pat, "LOOP", " =", 1)
    elif re.match(r'\s+(\w+\.)?iterate', line) is not None:
        new_line = process_name(new_line, def_pat, "I", " =", 1)
    # Special handling for alloc.
    elif re.search(r'=\s+alloc', line) is not None:
        new_line = process_name(new_line, def_pat, "RES", " =", 0)
    # Special handling for dim.
    elif re.search(r'=\s+dim', line) is not None:
        new_line = process_name(new_line, def_pat, "DIM", " =", 1)
    # Special handling for memory operations.
    elif re.search(r'(\w+\.)?load\s+', line) is not None:
        res = re.search(r'load\s+%([a-zA-Z0-9][a-zA-Z0-9_\-]*)', line)
        mem = res.group(1)
        if mem in name_dict.keys():
            mem = name_dict[mem]
        new_line = process_name(new_line, def_qual_pat,
                                "LOAD_"+mem+"_MEM", " =", 0)
    # Special handling for constant operations.
    elif re.search(r'\sconstant\s', line) is not None:
        res = re.search(r'\sconstant\s(-?[0-9\.]+)', line)
        num = res.group(1)
        num = res.group(1).replace("-", "minus_")
        num = num.replace(".", "_dot_")
        new_line = process_name(new_line, def_qual_pat, "CST_"+num, " =", 0)
    else:
        definitions = def_qual_pat.findall(new_line)
        for d in definitions:
            (name, num) = d
            x = use_name(name, num, " =")
            y = record_name_def(name, num, "VAR_"+name, " =", 0, new_line)
            new_line = new_line.replace(x, y)

    # Process uses.
    uses = use_qual_pat.findall(new_line)
    for u in uses:
        (name, num) = u
        x = use_name(name, num, "")
        y = translate_use_name(name, num, "")
        new_line = new_line.replace(x, y)

    # In debug mode, no need to perform FileCheck specific changes.
    if debug:
        print(new_line)
        return

    # Process output for FileCheck.
    # Avoid [[[ and ]]] and others text that are reserved by FileCheck.
    new_line = re.sub(r'\[\[\[', '{{.}}[[', new_line)  # get rid of [[[
    new_line = re.sub(r'\]\]\]', ']]{{.}}', new_line)  # get rid of ]]]
    # change [[1 -> *[1
    new_line = re.sub(r'\[\[\s*(\d)', r'{{.}}[\g<1>', new_line)
    # change [[-1 -> *[-1
    new_line = re.sub(r'\[\[\s*-\s*(\d)', r'{{.}}[-\g<1>', new_line)
    # change a]] -> 1]*
    new_line = re.sub(r'(\d)\s*\]\]', '\g<1>]{{.}}', new_line)
    if re.match(r'\s+func', line) is not None:
        # Split function line into 2 lines. Should make private optional
        new_line = re.sub(
            r'(\s+)(func\s+@[\w]+)\s*(\(.*)', r'// CHECK-LABEL:\1\2\n// CHECK-SAME: \1\3', new_line)
        new_line = re.sub(
            r'(\s+)(func\sprivate\s+@[\w]+)\s*(\(.*)', r'// CHECK-LABEL:\1\2\n// CHECK-SAME: \1\3', new_line)
        print(new_line)
    else:
        if line_color[i] == curr_parallel_color:
            # This line is in an established parallel region
            print("// CHECK-DAG:  ", new_line)
        elif line_color[i] == line_color[i+1]:
            # This line starts a parallel region, check if this break a parallel region.
            if curr_parallel_color != -1:
                # Previous lines were also part of a parallel region.
                # Need to separate them.
                print("// CHECK-NOT: separator of consecutive DAGs")
            curr_parallel_color = line_color[i]
            print("// CHECK-DAG:  ", new_line)
        else:
            # No parallel region, set the color to undefined
            curr_parallel_color = -1
            print("// CHECK:      ", new_line)


################################################################################
# main

def main(argv):
    global debug, check
    global def_set, line_color, curr_color, curr_parallel_color
    debug = 0
    check = 0
    input_command = "mlir2FileCheck.py"
    
    try:
        opts, args = getopt.getopt(
            argv, "hdca:n:", ["help", "debug", "check", "args=", "names="])
    except getopt.GetoptError:
        print_usage()
    for opt, arg in opts:
        if opt in ('-h', "--help"):
            print_usage()
        elif opt in ('-d', "--debug"):
            if check:
                print("cannot use check and debug at the same time")
                return
            debug = 1
        elif opt in ('-c', "--check"):
            if debug:
                print("cannot use check and debug at the same time")
                return
            check = 1
        elif opt in ("-a", "--args"):
            input_command += " -a\'" + arg + "\'"
            arg_names = json.loads(arg)
            print("//  use arg names:", arg_names)
            i = 0
            for new_name in arg_names:
                prepare_name_def("arg" + str(i), new_name.upper())
                i += 1
        elif opt in ("-n", "--names"):
            input_command += " -n\'" + arg + "\'"
            user_dict = json.loads(arg)
            print("//  use name dictionary:", user_dict)
            for orig_name, new_name in user_dict.items():
                prepare_name_def(orig_name, new_name.upper())
    if len(args) > 0:
        print("command does not use arguments without options: ", args)
        return

    # Handle stdout for checks
    orig_stdout = sys.stdout
    tmpfile = 'tmp.m2fc'
    if check:
        # When checking, redirect stdout to file.
        print('>> gen test file in "', tmpfile, '"')
        sys.stdout = open(tmpfile, 'w')

    # Compute def-use colors, and print in check mode.
    lines = []
    def_set = []  # Current set of defined variable in current parallel set.
    # Associate lines with a color; same color==same parallel set.
    line_color = []
    curr_color = 0  # Color identifying the current parallel set.
    for line in sys.stdin:
        l = line.rstrip()
        process_def_use_chains(l)
        if check:
            print(l)
        lines.append(l)
    # Add one to avoid checking out of bound accesses.
    line_color.append(curr_color+1)

    # Process the input.
    curr_parallel_color = -1
    print("\n// " + input_command)
    for i, line in enumerate(lines):
        process_line(i, line)

    # Complete the work when checking.
    if check:
        sys.stdout = orig_stdout
        print('>> done writing print file, now check')
        res = subprocess.run(['FileCheck', '--input-file='+tmpfile,
                              tmpfile], capture_output=True, text=True).stderr
        print(res)
        print('>> check completed (empty line is success)')


if __name__ == "__main__":
    main(sys.argv[1:])
