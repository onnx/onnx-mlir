#!/usr/bin/python3

# SPDX-License-Identifier: Apache-2.0

##################### make-report.py ########################################
#
# Copyright 2023 The IBM Research Authors.
#
################################################################################
#
# This file scan -opt-report=* and process it
#
# For patterns, see src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.cpp, 
# impl::onnxToKrnlParallelReport(...) and impl::onnxToKrnlSimdReport(...)
#
################################################################################

import sys
import getopt
import re
import numpy as np

def print_usage(msg = ""):
    if msg:
        print("Error:", msg, "\n")
    print("make-report.py -[svh] [-c <compile_log>] [-r <run_log>] [-l <num>] [-p <op regexp>]")
    print("")
    print("Usage: Report statistics on compiler and runtime characteristics of onnx ops.")
    print("")
    print("Compile-time statistics are collected from a `onnx-mlir` compiler output")
    print("with the `--opt-report` option equal to `Simd` or other supported sub-options.")
    print("")
    print("Runtime statistics are collected from the runtime output of a model compiled.")
    print("with the `--profile-ir` option equal to `Onnx` or other supported sub-options.")
    print("")
    print("When both compile time and runtime statistics are provided at the same time,")
    print("it will correlate the performance metrics with data gathered at compile time.")
    print("")
    print("Additional help.")
    print("  If you need more specific info on individual success/failure, run ")
    print("  `onnx-mlir --debug-only=lowering-to-krnl` and look at the compiler output.")
    print("  Use `-l 3` to correlate the node name printed here with compiler output.")
    print("")
    print("Parameters:")
    print("  -c/--compile <file_log>: File name containing the compile time statistics.")
    print("  -r/--runtime <file_log>: File name containing the runtime statistics.")
    print("  -l/--level <num>:    Print statistics:")
    print("                       0: Just count successful/unsuccessful ops.")
    print("                       1: Also count reasons for success/failure.")
    print("                       2: Also list metrics.")
    print("                       3: Also list node name.")
    print("  -f/--focus <regexp>: Focus only on ops that match the regexp pattern.")
    print("  -s/supported:        Focus only on ops that are supported. Namely, the report")
    print("                       will skip ops for which compile-time statistics list")
    print("                       the 'unsupported' keyword in its printout.")
    print("                       For SIMD/parallel statistics, this include all ops that")
    print("                       have currently no support for it.")
    print("  -u/--unit <str>:     Time in second ('s'), millisecond ('ms') or microsecond ('us).")
    print("  -v/--verbose:        Run in verbose mode (see error and warnings).")
    print("  -h/--help:           Print usage.")
    print("")
    exit(1)

################################################################################
# Global info.

# For statistic info.
op_count_dict = {} # op -> count
op_detail_count_dict = {} # op -> {dictionary of detailed pattern -> count}
op_time_dict = {} # op -> cumulative time
op_detail_time_dict = {} # op -> {dictionary of detailed pattern -> cumulative time}

# For timing info
node_time_dict = {}  # op + node_name -> time statistic

focus_on_op_with_pattern = r'.*'
spurious_node_name_count = 0
error_missing_time = 0
supported_only = False
has_timing = False
verbose = False
report_level = 0 # 0: none; 1: details; 2: extra info; 3: plus node names
time_unit = 1 # seconds

# Basic pattern for reports: "==" <stat name> "==," <op name> "," <node name> ","
def common_report_str(stat_name):
    return r'^==' + stat_name + r'-REPORT==,\s*([0-9a-zA-Z\.\-]+)\s*,\s*([^,]*),\s*(.*)'

# ==SIMD-REPORT==, ..., <explanations>, <VL>, <simd-trip-count>
simd_stat_message = "SIMD vector length (in elements), SIMD loop trip count (-1 is runtime), message"

# ==PERF-REPORT==, ..., "before" | "after", time since last call, absolute time
perf_stat_message = "(after|before), time for op(s), time since start(s)"

################################################################################
# # Support.

# To record time, use op name and node name to better disambiguate.
def timing_dict_key(op, node_name):
    p = re.match(r'(.*)-(simd|par)', op)
    if p:
        op = p[1]
    return op + "__" + node_name

# Add num to dict[key]
def add_to_dict_entry(dict, key, num):
    if key in dict:
        return dict[key] + num
    # First visit, entry does not exist.
    return num

# Dict1 is a dictionary of dictionaries. First locate the secondary directory,
# dict1[key1], and then add num to the key2 entry of that secondary dictionary.
def add_to_dict2_entry(dict1, key1, key2, num):
    if key1 in dict1:
        # Retrieve dict of dict.
        dict2 = dict1[key1]
        # Increment entry for key2.
        dict2[key2] = add_to_dict_entry(dict2, key2, num)
        return dict2
    # First visit, secondary dict is empty.
    return { key2 : num}

def append_to_dict_entry(dict, key, num):
    if key in dict:
        return np.append(dict[key], num)
    # First visit, entry does not exist.
    return np.array([num])

def append_to_dict2_entry(dict1, key1, key2, num):
    if key1 in dict1:
        # Retrieve dict of dict.
        dict2 = dict1[key1]
        # Increment entry for key2.
        dict2[key2] = append_to_dict_entry(dict2, key2, num)
        return dict2
    # First visit, secondary dict is empty.
    return { key2 : np.array([num])}


def record_pattern(op, node_name, detail_key):
    global op_count_dict, op_detail_count_dict
    global op_time_dict, op_detail_time_dict
    global node_time_dict
    global verbose, report_level, has_timing, error_missing_time

    # Update statistic summaries
    op_count_dict[op] = add_to_dict_entry(op_count_dict, op, 1)
    if report_level > 0:
        op_detail_count_dict[op] = add_to_dict2_entry(
            op_detail_count_dict, op, detail_key, 1)

    # Has timing for this node?
    if not has_timing:
        return
    # Process timing.
    timing_key = timing_dict_key(op, node_name)
    if not timing_key in node_time_dict:
        error_missing_time += 1
        if verbose:
            print("> timing key", timing_key, "with no times found in the performance data.")
        return
    # Update timing summaries
    time = node_time_dict[timing_key]
    op_time_dict[op] = append_to_dict_entry(op_time_dict, op, time)
    if report_level > 0:
        op_detail_time_dict[op] = append_to_dict2_entry(
            op_detail_time_dict, op, detail_key, time)


################################################################################
# Parse line (generic).


def parse_line(line, report_str, is_perf_stat):
    global focus_on_op_with_pattern, supported_only
    global verbose, spurious_node_name_count

    p = re.match(report_str, line)
    if p is None:
        return (False, "", "", "")
    # Have a line of relevant info, extract op, op name, and stat details.
    op = p[1]
    node_name = p[2]
    details = p[3]
    # If we process supported op only, search for "unsupported" in details.
    if supported_only and re.search(r'unsupported', details) is not None:
        return (False, "", "", "")
    # If we process perf, we don't care about the "before"
    if is_perf_stat and re.search(r'before', details) is not None:
        return (False, "", "", "")

    # Check if we have an op that we focus on; if not skip.
    f = re.match(focus_on_op_with_pattern, op)
    if f is None:
        return (False, "", "", "")
    # Have a perfect match.

    if False:
        # Issues due to runtime constants having issues.
        new_node_name = node_name
        # Spurious appending of the last node_name.
        if parse_line.last_node_name:
            i0 = re.match(r'(.+)'+parse_line.last_node_name, node_name)
            if i0:
                new_node_name = i0[1]
                spurious_node_name_count += 1
                if verbose:
                    print("Cut last node_name:\n  old:", node_name,
                          "\n  cut:", parse_line.last_node_name,
                          "\n  new:", new_node_name)
        parse_line.last_node_name = node_name
        # Repeating node name.
        i1 = re.match(r'(.+)'+op, node_name)
        if i1:
            new_node_name = i1[1]
            spurious_node_name_count += 1
            if verbose:
                print("Cut op name:\n  old:", node_name,
                      "\n  cut:", op,"\n  new:", new_node_name)
        # Use new node name.
        node_name = new_node_name
    return (True, op, node_name, details)

parse_line.last_node_name = ""

################################################################################
# Parse file for statistics.

def parse_file_for_stat(file_name, stat_name):
    global report_level

    try:
        file = open(file_name, 'r')
    except OSError:
        print_usage("Could not open file `"+file_name+"`")

    report_str = common_report_str(stat_name)
    is_perf_stat = re.match(r'PERF', stat_name)
    for line in file:
        # Parse line.
        (has_stat, op, node_name, details) = parse_line(line.rstrip(),
            report_str, is_perf_stat)
        if not has_stat:
            continue
        # Use stat.
        secondary_key = ""
        detail_array = details.split(",")
        if report_level == 1:
            # Use only first element in secondary key.
            secondary_key = detail_array[0]
        elif report_level == 2:
            # Use all details in secondary key.
            secondary_key = details
        elif report_level == 3:
            # Use node name in secondary key.
            secondary_key = node_name + ", " + details

        record_pattern(op, node_name, secondary_key)

################################################################################
# Parse file for performance

def parse_file_for_perf(file_name, stat_name):
    global node_time_dict
    global spurious_node_name_count, verbose, has_timing

    try:
        file = open(file_name, 'r')
    except OSError:
        print_usage("Could not open file `"+file_name+"`")

    report_str = common_report_str(stat_name)
    time_stat_dict = {} # op+op_name -> numpy array of times
    last_node_name = ""
    for line in file:
        # Parse line.
        (has_stat, op, node_name, details) = parse_line(line.rstrip(),
            report_str, True)
        if not has_stat:
            continue
        # Keep only after times.
        detail_array = details.split(",")
        key = timing_dict_key(op, node_name)
        time_stat_dict[key] = append_to_dict_entry(time_stat_dict,
            key, float(detail_array[1]))

    # Normally, an <op>-<node-name> pair should be seen only once in a run,
    # except for loops. So we take here the sum of all the times.
    # This approach would not work well if we had performance for multiple
    # runs.
    # TODO: If wanted to average/min/max over multiple runs, we would have
    # need to pull this inside of the loop above, summing at the end of
    # a run, and then taking min/max/average of the times gathered for each
    # run.
    for node in time_stat_dict:
        node_time_dict[node] = np.sum(time_stat_dict[node])
    has_timing = True

################################################################################
# make report

def make_report(stat_message):
    global op_count_dict, op_detail_count_dict
    global op_time_dict, op_detail_time_dict
    global report_level, supported_only, verbose, spurious_node_name_count
    global has_timing, time_unit, error_missing_time

    num_desc = "num"
    if has_timing:
        if time_unit == 1:
            num_desc += ", cumulative time (s)"
        elif time_unit == 1000:
            num_desc += ", cumulative time (ms)"
        elif time_unit == 1000*1000:
            num_desc += ", cumulative time (us)"
    print("Statistic legend:")
    if report_level < 2:
        print("  op-name:", num_desc)
    elif report_level == 2:
        print("   " + num_desc + ":", stat_message, "\n")
    elif report_level == 3:
        print("   " + num_desc + ": node-name, ", stat_message, "\n")
    print("")
    if supported_only:
        print("Statistics start (ignore unsupported ops).")
    else:
        print("Statistics start (all ops).")
    for op in sorted(op_count_dict):
        count_time_str = str(op_count_dict[op])
        if op in op_time_dict:
            time = np.sum(op_time_dict[op])
            count_time_str += ", {:.7f}".format(time * time_unit)
        print("  " + op + ", " + count_time_str)
        if report_level:
            det_dict = op_detail_count_dict[op]
            det_time_dict = {}
            if op in op_detail_time_dict:
                det_time_dict = op_detail_time_dict[op]
            for det_key in sorted(det_dict):
                if det_dict[det_key] == op_count_dict[op]:
                    count_time_str = "*"
                else:
                    count_time_str = str(det_dict[det_key])
                if det_key in det_time_dict:
                    time = np.sum(det_time_dict[det_key])
                    count_time_str += ", {:.7f}".format(time * time_unit)
                print("    ", count_time_str, ":", det_key)
    print("Statistics end.")

    # Report spurious node name if any.
    if spurious_node_name_count:
        if error_missing_time:
            print("> Spurious node name were detected.")
            print("> Timing information was missing for some of the nodes.")
        else:
            print("> Spurious node name were detected and fixed.")
        print("> Run with `-v` for detailed list of fixes and errors.")
    elif error_missing_time:
        print("> Timing information was missing for some of the nodes.")
        print("> Run with `-v` for detailed list of errors.")


################################################################################
# Main.

def main(argv):
    global report_level, focus_on_op_with_pattern, supported_only, time_unit, verbose

    compile_file_name = ""
    runtime_file_name = ""
    try:
        opts, args = getopt.getopt(
            argv, "c:f:hl:r:su:v",
            ["compile=", "focus=", "help", "level=", "runtime=", "supported", "unit=", "verbose"])
    except getopt.GetoptError:
        print_usage("Failure to parse inputs")
    for opt, arg in opts:
        if opt in ('-c', "--compile"):
            compile_file_name = arg
        elif opt in ('-f', "--focus"):
            focus_on_op_with_pattern = arg
            if report_level == 0:
                report_level = 1
        elif opt in ('-h', "--help"):
            print_usage()
        elif opt in ('-l', "--level"):
            report_level = int(arg)
            if (report_level<0 or report_level > 3):
                print_usage("detail levels are 0, 1, 2, or 3")
        elif opt in ('-r', "--runtime"):
            runtime_file_name = arg
        elif opt in ('-s', "--supported"):
            supported_only = True
        elif opt in ('-u', "--unit"):
            if re.match(r'\s*s\s*', arg):
                time_unit = 1
            elif re.match(r'\s*ms\s*', arg):
                time_unit = 1000
            elif re.match(r'\s*us\s*', arg):
                time_unit = 1000*1000
            else:
                print_usage("time units are 's', 'ms', or 'us'")
        elif opt in ('-v', "--verbose"):
            verbose = True

    if compile_file_name and runtime_file_name:
        parse_file_for_perf(runtime_file_name, "PERF")
        parse_file_for_stat(compile_file_name, "SIMD")
        make_report(simd_stat_message)
    elif compile_file_name:
        parse_file_for_stat(compile_file_name, "SIMD")
        make_report(simd_stat_message)
    elif runtime_file_name:
        parse_file_for_perf(runtime_file_name, "PERF")
        parse_file_for_stat(runtime_file_name, "PERF")
        make_report(perf_stat_message)
    else:
        print_usage("Command requires an input file name (compile/runtime or both).\n")


if __name__ == "__main__":
    main(sys.argv[1:])
