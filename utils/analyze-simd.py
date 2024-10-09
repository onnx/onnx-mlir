#!/usr/bin/python3

#!/usr/local/bin/python3

##################### analyze-simd.py ##########################################
#
# Copyright 2023 The IBM Research Authors.
#
################################################################################
# Analyze SIMD in a .s file

import sys
import getopt
import re
import io
import subprocess
from pathlib import Path

################################################################################
# Usage.


def dprint(msg):
    sys.stderr.write(msg + "\n")


def print_usage(msg=""):
    dprint("")
    if msg:
        dprint("ERROR: " + msg + "\n")
    dprint("analyze-simd [-t <arch>] (-a|-c|-m|-o)+ [-n num] [-f pattern] [-dhlp] file")
    dprint("  Utility to analyze and print SIMD code located in functions")
    dprint("")
    dprint("Pattern:")
    dprint("  -t | --target <arch>: set op names to the given <arch>.")
    dprint("")
    dprint("  -a | --all:      all patterns below.")
    dprint("  -c | --compute:  search for vector compute (add/mul) patterns.")
    dprint("  -m | --mem:      search for vector vector memory patterns.")
    dprint("  -o | --overhead: search for vector overhead patterns.")
    dprint("")
    dprint("  -f | --function pattern: investigate only functions whose name match")
    dprint('                           the regexp pattern (default "^main_graph$").')
    dprint(
        "  -n | --num num:          investigate only that one occurrence num(default all)"
    )
    dprint("                           (default all: -1).")
    dprint("")
    dprint("  -d | --details: print detailed op stats.")
    dprint("  -p | --print:   print code of basic block with SIMD.")
    dprint("  -l | --listing: list all the of the code, regardless of SIMD.")
    dprint("  -h | --help:    help")
    dprint("")
    dprint("Typical command (print code and analysis for andy simd in main_graph):")
    dprint("  analyze-simd -cmop <file>")
    dprint("")
    sys.exit()


################################################################################
# Globals.

debug = 0  # 1 for emitting stats, 2 for basic block detection
print_code = False
print_listing = False
print_details = False
fct_match_str = ""
op_dict = {}
aggr_dict = {}
op_name = {}
bb_boundary = {}

################################################################################
# SIMD op patterns for given architecture.


def define_arch_op_names(arch):
    global op_name
    dprint("# use " + arch + " target arch")
    if arch == "z":
        op_name["vload"] = "(vl|[vw]fi)"
        op_name["vload-splat"] = "vlrep"
        op_name["vstore"] = "vst"
        # perm | merge | select | shift left | replicate | permute | gen mask | gen mask
        op_name["vshuffle"] = "(vperm|vsel|vmr|vsl|vrep|vpdi|vgm|vzero)"
        op_name["vfma"] = "vfma"
        op_name["vmul"] = "vfm.b"
        op_name["vdiv"] = "vfd"
        # vector conversion between formats (NNPA <-> fp, FP <-> int, int <-> int)
        op_name["vconv"] = (
            "(vclfnh|vclfnl|vcfn|vcrnf|vcnf|vclgd|vclfeb|vclgdb|vpkh|vpkf|vpkg)"
        )
        # add | sub| max | min | compare
        op_name["vadd"] = "([vw]fa|[vw]fs|[vw]fmax|[vw]fmin|[vw]f[ck][eh])"
        op_name["load"] = "lg"
        op_name["store"] = "stg"
    elif arch == "x86":  # generic x86
        op_name["vload"] = "(v?mov[au]p[sd]|mov(h|hl|lh|l)ps)"
        op_name["vload-splat"] = "nothingtosee"
        op_name["vstore"] = (
            "v?movntp[sd]"  # non temporal... other store may be just mov too
        )
        # perm | merge | shift left | replicate | permute | gen mask | gen mask
        op_name["vshuffle"] = "(v?shufp[sd]|v?unpck[lh]p)"
        op_name["vfma"] = "v?fmadd[123]+p[ds]"
        op_name["vmul"] = "v?mulp[ds]"
        op_name["vdiv"] = "v?divp[sd]"
        op_name["vconv"] = ""
        # add | sub| max | min | compare | and
        op_name["vadd"] = (
            "(v?addp[ds]|v?subp[ds]|v?maxp[ds]|v?min[dp]|cmp..p[sd]|andp|andnp|orp|xorp|pand|pandn|por|pxor)"
        )
        op_name["load"] = "mov"
        op_name["store"] = "mov"
    else:
        print_usage("unknown arch (z or x86 at this time)")


################################################################################
# Dictionary support.


def inc_op_dict(op):
    global op_dict
    if op in op_dict:
        op_dict[op] += 1
    else:
        op_dict[op] = 1


def get_op_dict(op):
    global op_dict
    if op in op_dict:
        return str(op_dict[op])
    return "0"


def inc_aggr_dict(op):
    global aggr_dict
    if op in aggr_dict:
        aggr_dict[op] += 1
    else:
        aggr_dict[op] = 1


def get_aggr_dict(op):
    global aggr_dict
    if op in aggr_dict:
        return str(aggr_dict[op])
    return "0"


################################################################################
# Characterize ops.


def characterize_op(line):
    # SIMD
    if re.match(r".*\s" + op_name["vload"], line):
        inc_op_dict("vload")
        inc_aggr_dict("vmem")
        inc_aggr_dict("vec")
        return "vload"
    if re.match(r".*\s" + op_name["vload-splat"], line):
        inc_op_dict("vload-splat")
        inc_aggr_dict("vmem")
        inc_aggr_dict("vec")
        return "vload-splat"
    if re.match(r".*\s" + op_name["vstore"], line):
        inc_op_dict("vstore")
        inc_aggr_dict("vmem")
        inc_aggr_dict("vec")
        return "vstore"
    if re.match(r".*\s" + op_name["vshuffle"], line):
        inc_op_dict("vshuffle")
        inc_aggr_dict("voverhead")
        inc_aggr_dict("vec")
        return "vshuffle"
    if re.match(r".*\s" + op_name["vmul"], line):
        inc_op_dict("vmul")
        inc_aggr_dict("vcompute")
        inc_aggr_dict("vec")
        return "vmul"
    if re.match(r".*\s" + op_name["vdiv"], line):
        inc_op_dict("vdiv")
        inc_aggr_dict("vcompute")
        inc_aggr_dict("vec")
        return "vdiv"
    if re.match(r".*\s" + op_name["vadd"], line):
        inc_op_dict("vadd")
        inc_aggr_dict("vcompute")
        inc_aggr_dict("vec")
        return "vadd"
    if re.match(r".*\s" + op_name["vfma"], line):
        inc_op_dict("vfma")
        inc_aggr_dict("vcompute")
        inc_aggr_dict("vec")
        return "vfma"
    if re.match(r".*\s" + op_name["vconv"], line):
        inc_op_dict("vconv")
        inc_aggr_dict("vec")
        return "vconv"
    # Scalar
    if re.match(r".*\s" + op_name["load"], line):
        inc_op_dict("load")
        inc_aggr_dict("mem")
        inc_aggr_dict("scalar")
        return "load"
    if re.match(r".*\s" + op_name["store"], line):
        inc_op_dict("store")
        inc_aggr_dict("mem")
        inc_aggr_dict("scalar")
        return "store"
    # else
    inc_aggr_dict("other")
    return "other"


def characterize_ops(buffer, reset=True):
    global op_dict, aggr_dict, op_name

    if reset:
        op_dict = {}
        aggr_dict = {}
    b = io.StringIO(buffer)
    for line in b:
        l = line.rstrip()
        characterize_op(l)


def print_characterization(details=False):
    vcompute_vec = float(get_aggr_dict("vcompute"))
    vec = float(get_aggr_dict("vec"))
    tot = (
        float(get_aggr_dict("vec"))
        + float(get_aggr_dict("scalar"))
        + float(get_aggr_dict("other"))
    )
    if vec > 0.0:
        vcompute_vec = vcompute_vec / vec
    vec_tot = vec
    if tot > 0.0:
        vec_tot = vec_tot / tot
    print(
        "# vector ops, "
        + get_aggr_dict("vec")
        + ", compute, "
        + get_aggr_dict("vcompute")
        + ", conversion, "
        + get_op_dict("vconv")
        + ", mem, "
        + get_aggr_dict("vmem")
        + ", overhead, "
        + get_aggr_dict("voverhead")
        + ", vcompute/vec, {:.2f}".format(vcompute_vec)
    )
    print("# scalar, " + get_aggr_dict("scalar") + ", mem, " + get_aggr_dict("mem"))
    print("# others, " + get_aggr_dict("other") + ", vec/tot, {:.2f}".format(vec_tot))
    if details:
        print("# details:")
        for f in op_name.keys():
            print("#   " + f + ", " + get_op_dict(f))


################################################################################
# Scan file for a basic block boundaries.


def scan_basic_blocks(filename):
    global bb_boundary

    last_line_id = (
        ""  # When seen end of function, mark last like as basic block boundary.
    )
    next_line_is_bb = False  # New function and new branch signal that
    # next line is start of a basic block.
    curr_fct_name = "unknown-function"  # Mark bb_boundary by function name.

    for line in open(filename, "r"):
        l = line.rstrip()
        if debug == 2:
            dprint(l)
        if re.match(r"^$", l):
            # Empty line, signal end of function.
            if last_line_id:
                if debug == 2:
                    dprint("bb due to new function " + last_line_id)
                bb_boundary[last_line_id] = curr_fct_name
                last_line_id = ""
        elif re.search(r"<\S+>:", l):
            # Has a new function.
            m = re.search(r"<(\S+)>:", l)
            curr_fct_name = m.group(1)
            next_line_is_bb = True
            if debug == 2:
                dprint("new function named " + curr_fct_name)
        elif re.match(r"\s+([0-9a-fA-F]+):\s", l):
            # Has a line of code.
            m = re.match(r"\s+([0-9a-fA-F]+):\s", l)
            curr_line_id = m.group(1)
            if next_line_is_bb:
                # Have a line, and next line was set as head of a new basic block.
                if debug == 2:
                    dprint("bb due to next line " + curr_line_id)
                bb_boundary[curr_line_id] = curr_fct_name
                next_line_is_bb = False
            # Now also test if we have a branch.
            if re.search(r"j.*<.*\+0x[0-9a-fA-F]+>", l):
                # Have a branch: next line and target should signal new basic blocks.
                if debug == 2:
                    dprint("branch on line " + curr_line_id)
                next_line_is_bb = True
                mm = re.match(
                    r"\s+[0-9a-fA-f]+:(\s+[0-9a-fA-f]{2})+\s+\w*j\w*\s+([0-9a-fA-f]+)",
                    l,
                )
                if mm:
                    goto_line_id = mm.group(2)
                    bb_boundary[goto_line_id] = curr_fct_name
                    if debug == 2:
                        dprint("bb due to branch target " + goto_line_id)
            last_line_id = curr_line_id


################################################################################
# Scan file for a basic block with the given pattern.
# When ID is -1; list the blocks that have it.
# When ID >=0, list the block with that occurrence.


def print_lines(lines, print_it=False):
    output = ""
    for ll in lines:
        # For some reasons, some .s have line like these:
        #     235a:       00 00
        # we would like to skip them.
        if re.match(r"\s*[0-9a-fA-F]+:[\s0]*$", ll) is None:
            output += ll + "\n"
    if print_it:
        print(output, end="")
    return output


def scan_for_simd(filename, pattern, id):
    global bb_boundary, fct_match_str, print_code, print_details, print_listing
    print('# Search pattern "' + pattern + '" in file "' + filename + '", id', id)
    match_str = r".*\s(" + pattern + ")"
    hasPattern = False
    lines = []
    count = -1

    for line in open(filename, "r"):
        l = line.rstrip()
        if debug == 1:
            dprint(l)
        # Get line number, to see if we have a basic block.
        match_line = re.match(r"\s+([0-9a-fA-F]+):\s", l)
        if match_line:
            # Has a line of code.
            curr_line_id = match_line.group(1)
            if curr_line_id in bb_boundary:
                # Has a basic block boundary.
                if debug == 1:
                    dprint(
                        "has basic block at line "
                        + curr_line_id
                        + " in function "
                        + bb_boundary[curr_line_id]
                    )
                match_fct = re.match(fct_match_str, bb_boundary[curr_line_id])
                if match_fct and hasPattern:
                    if id < 0 or id == count:
                        print("# Block #" + str(count) + " with SIMD")
                        output = print_lines(lines, print_code)
                        characterize_ops(output)
                        print_characterization(print_details)
                        if not print_listing and not print_code:
                            print("")
                        if id >= 0:
                            return output
                else:
                    print_lines(lines, (print_code and match_fct) or print_listing)
                lines = []
                hasPattern = False
        # Analyze line for SIMD.
        if re.match(match_str, l):
            # Has SIMD.
            if not hasPattern:
                hasPattern = True
                count += 1
            if debug == 1:
                dprint("Match pattern, number " + str(count) + ": " + l)
            offset = max(1, 70 - len(line))
            op_type = characterize_op(l)
            lines.append(l + (" " * offset) + "<<<<==== " + op_type)
        else:
            lines.append(l)

    print_lines(lines, print_listing)
    return ""


################################################################################
# Main.


def add_pattern(pattern, new_pattern):
    if not pattern:
        return new_pattern
    return pattern + "|" + new_pattern


def main(argv):
    global fct_match_str, print_code, print_details, print_listing, op_name
    pattern = ""
    num = -1  # All.
    arch = "z"

    try:
        opts, args = getopt.gnu_getopt(
            argv,
            "t:acmodhln:pf:",
            [
                "target=",
                "all",
                "compute",
                "mem",
                "overhead",
                "details",
                "help",
                "listing",
                "num=",
                "print",
                "function=",
            ],
        )
    except getopt.GetoptError:
        dprint("Error: unknown options")
        print_usage()
    for opt, arg in opts:
        if opt in ("-t", "--target"):
            # This option must come before any patterns to search for.
            arch = arg
            if pattern:
                print_usage("Arch option must come before search options")
        elif opt in ("-a", "--all"):
            if not pattern:
                define_arch_op_names(arch)
            # Compute
            pattern = add_pattern(pattern, op_name["vfma"])
            pattern = add_pattern(pattern, op_name["vadd"])
            pattern = add_pattern(pattern, op_name["vmul"])
            pattern = add_pattern(pattern, op_name["vdiv"])
            pattern = add_pattern(pattern, op_name["vconv"])
            # Mem
            pattern = add_pattern(pattern, op_name["vload"])
            pattern = add_pattern(pattern, op_name["vload-splat"])
            pattern = add_pattern(pattern, op_name["vstore"])
            # Overhead
            pattern = add_pattern(pattern, op_name["vshuffle"])
        elif opt in ("-c", "--compute"):
            if not pattern:
                define_arch_op_names(arch)
            pattern = add_pattern(pattern, op_name["vfma"])
            pattern = add_pattern(pattern, op_name["vadd"])
            pattern = add_pattern(pattern, op_name["vmul"])
            pattern = add_pattern(pattern, op_name["vdiv"])
            pattern = add_pattern(pattern, op_name["vconv"])
        elif opt in ("-m", "--mem"):
            if not pattern:
                define_arch_op_names(arch)
            pattern = add_pattern(pattern, op_name["vload"])
            pattern = add_pattern(pattern, op_name["vload-splat"])
            pattern = add_pattern(pattern, op_name["vstore"])
        elif opt in ("-o", "--overhead"):
            if not pattern:
                define_arch_op_names(arch)
            pattern = add_pattern(pattern, op_name["vshuffle"])
        elif opt in ("-n", "--num"):
            num = int(arg)
        elif opt in ("-f", "--function"):
            fct_match_str = arg
        elif opt in ("-d", "--details"):
            print_details = True
        elif opt in ("-l", "--listing"):
            print_code = True
            print_listing = True
        elif opt in ("-p", "--print"):
            print_code = True
        elif opt in ("-h", "--help"):
            print_usage()
        else:
            print_usage("Unknown option")
    if not pattern:
        print_usage("Expect at least a pattern to search")
    if len(args) != 1:
        # All commands after the file name seems to be added here!!!
        print_usage("Need an single input file as last option: ", args, ".")
    filename = args[0]

    name_stub = Path(filename).stem
    if not fct_match_str:
        fct_match_str = r"^main_graph_" + name_stub + "$"
        print(
            '# search default function: main_graph with default tag "'
            + fct_match_str
            + '"'
        )

    match_binary = re.match(r"(.*)\.so$", filename)
    if match_binary:
        asm_filename = match_binary.group(1) + ".s"
        cmd = "objdump -S ./" + filename + " > ./" + asm_filename
        dprint("# generate asm file with: " + cmd)
        ret = subprocess.call(cmd, shell=True)
        filename = asm_filename

    scan_basic_blocks(filename)
    buff = scan_for_simd(filename, pattern, num)
    return


if __name__ == "__main__":
    main(sys.argv[1:])
