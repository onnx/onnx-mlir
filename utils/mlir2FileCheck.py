#!/usr/local/bin/python3

import sys
import getopt
import fileinput
import re
import json
import subprocess

####################################################################
# Usage.

def print_usage():
    print('Translate mlir format from stdin to a suitable FileCheck format.')
    print('mlir2FileCheck [-cdh] [-a <arg arrays>] [-d <dict>]')
    print('  -a,--args <array>: Rename function arguments using a json array,')
    print('                     such as \'["A", "B", "C"] for 1st, 2nd, & 3rd args.\'.')
    print('  -c,--check       : Run FileCheck on output, to verify that the ouput is good.')
    print('  -d,-debug        : Rename for easier debugging of code, disable FileCheck format.')
    print('  -h,--help        : Print help.')
    print('  -n,--names <dict>: Rename variables using a json dictionary')
    print('                     such as \'{"cst": "ZERO"}\' to rename cst to ZERO.')
    sys.exit()

    
####################################################################
# Handling of names: global dictionaries.

name_dict = {}  # orig_name -> new_name (with ref count)
refcount_dict = {}  # new_name (without ref count) -> definition count

# Prepare name for def, for mapping given by the user.
def prepare_name_def(orig_name, new_name):
    if orig_name in name_dict.keys():
        print("multiple definitions of original name:", orig_name)
        exit()
    name_dict[orig_name] = new_name

# Process name and make it a legit def for FileCheck.
def def_string(name, num):
    if debug:
        return name + "_" + num
    return "[[" + name + "_:%.+]]" + num

# Record an original name/number, with a suggested new name. Append_str is
# added at the end (to match the string looked for and to be substituted).
def record_name_def(orig_name, orig_num, new_name, append_str, num_for_zero):
    # original name: what we found in the text
    # new_name: if something was prepared for it, use the prepared name. But
    # that is not the case, use the new name.
    if orig_name in name_dict.keys():
        # When prepare, we add "orig_name -> new_name".
        if new_name in refcount_dict.keys():
            print("multiple definitions of original name:", orig_name)
            exit()
        # A name is already associated, but without a refcount; it was a prepare.
        # We will use the pre-associated name.
        new_name = name_dict[orig_name]
        num_for_zero = 0
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


####################################################################
# process a line

# Process the substitution for a whole line for the given pattern.
def process_name(new_line, pattern, default_name, append_str, num_for_zero):
    definitions = pattern.findall(new_line)
    for d in definitions:
        (name, num) = d
        x = use_name(name, num, append_str)
        y = record_name_def(name, num, default_name, append_str, num_for_zero)
        new_line = new_line.replace(x, y)
    return new_line

# Drive the processing of the current line.
def process_line(line):
    # print("Process:", line)
    def_arg_pat = re.compile(r'%([a-zA-Z0-9][a-zA-Z0-9_\-]*)():')
    def_qual_pat = re.compile(r'%([a-zA-Z0-9][a-zA-Z0-9_\-]*)(:\d+)?\s+=')
    def_pat = re.compile(r'%([a-zA-Z0-9][a-zA-Z0-9_\-]*)()\s+=')
    def_op_pat = re.compile(r'=\s+(?:\w+\.)?(\w+)')
    use_qual_pat = re.compile(r'%([a-zA-Z0-9][a-zA-Z0-9_\-]*)(:\d+)?')
    # Keep only function related text, which start by def with a space
    if re.match(r'\s+', line) is None:
        return
    new_line = line
    # Process defintions, with custom handlng for func, iterates, define_loops,
    # alloc, dim.
    if re.match(r'\s+func', line) is not None:
        new_line = process_name(new_line, def_arg_pat, "PARAM", ":", 1)
    elif re.match(r'\s+krnl\.iterate', line) is not None:
        new_line = process_name(new_line, def_pat, "I", " =", 1)
    elif re.search(r'krnl\.define_loops', line) is not None:
        new_line = process_name(new_line, def_qual_pat, "LOOP", " =", 1)
    elif re.search(r'=\s+alloc', line) is not None:
        new_line = process_name(new_line, def_pat, "RES", " =", 0)
    elif re.search(r'=\s+dim', line) is not None:
        new_line = process_name(new_line, def_pat, "DIM", " =", 1)
    elif re.search(r'(\w+\.)load\s+', line) is not None:
        res = re.search(r'load\s+%([a-zA-Z0-9][a-zA-Z0-9_\-]*)', line)
        mem = res.group(1)
        if mem in name_dict.keys():
            mem = name_dict[mem]
        new_line = process_name(new_line, def_qual_pat, "LOAD_"+mem+"_MEM", " =", 0)
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
            y = record_name_def(name, num, "VAR_"+name, " =", 0)
            new_line = new_line.replace(x, y)
    # Process uses.
    uses = use_qual_pat.findall(new_line)
    for u in uses:
        (name, num) = u
        x = use_name(name, num, "")
        y = translate_use_name(name, num, "")
        new_line = new_line.replace(x, y)
        # print("use ", x, " to ", y, " using u", u)
    
    if debug:
        print(new_line)
    else:
      # Avoid [[[ and ]]] and others text that are reserved by FileCheck.
      new_line = re.sub(r'\[\[\[', '{{.}}[[', new_line) # get rid of [[[
      new_line = re.sub(r'\]\]\]', ']]{{.}}', new_line) # get rid of ]]]
      new_line = re.sub(r'\[\[\s*(\d)', '{{.}}[\g<1>', new_line) # change [[1 -> *[1
      new_line = re.sub(r'\[\[\s*-\s*(\d)', '{{.}}[-\g<1>', new_line) # change [[-1 -> *[-1
      new_line = re.sub(r'(\d)\s*\]\]', '\g<1>]{{.}}', new_line) # change a]] -> 1]*
      if re.match(r'\s+func', line) is not None:
          # Split function line into 2 lines.
          new_line = re.sub(
            r'(\s+)(func\s+@[\w]+)\s*(\(.*)', r'//CHECK-LABEL:\1\2\n//CHECK-SAME: \1\3', new_line)
          print(new_line)
      else:
          print("//CHECK:      ", new_line)

          
####################################################################
# main

def main(argv):
    global debug, check
    debug = 0
    check = 0
    try:
        opts, args = getopt.getopt(argv, "hdca:n:", ["help", "debug", "check", "args=", "dict="])
    except getopt.GetoptError:
        print_usage()
    for opt, arg in opts:
        if opt in ('-h', "--help"):
            print_usage()
        elif opt in ('-d', "--debug"):
            if check:
                print("cannot use check and debug at the same time");
                return
            debug = 1
        elif opt in ('-c', "--check"):
            if debug:
                print("cannot use check and debug at the same time");
                return
            check = 1
        elif opt in ("-a", "--args"):
            arg_names = json.loads(arg)
            print("//  use arg names:", arg_names)
            i = 0
            for new_name in arg_names:
                prepare_name_def("arg" + str(i), new_name.upper())
                i += 1
        elif opt in ("-n", "--names"):
            user_dict = json.loads(arg)
            print("//  use name dictionary:", user_dict)
            for orig_name, new_name in user_dict.items():
                prepare_name_def(orig_name, new_name.upper())

    # Normal run, process each line in turn.
    if check == 0:
        for line in sys.stdin:
            process_line(line.rstrip())
        return
    
    # Copy input, process output, then pipe it all to FileCheck.
    orig_stdout = sys.stdout
    tmpfile = 'tmp.m2fc'
    print('>> gen test file in "', tmpfile, '"') 
    sys.stdout = open(tmpfile, 'w')
    lines = []
    # Print the original.
    for line in sys.stdin:
        l = line.rstrip()
        print(l)
        lines.append(l)
    # Process the input
    for line in lines:
        process_line(line)
    sys.stdout = orig_stdout
    print('>> done writing print file, now check')
    res = subprocess.run(['FileCheck', '--input-file='+tmpfile, tmpfile], capture_output=True, text=True).stderr
    print(res)
    print('>> check completed (empty line is success)')
    


if __name__ == "__main__":
    main(sys.argv[1:])
