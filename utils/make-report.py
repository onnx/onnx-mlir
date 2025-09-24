#!/usr/bin/python3

# SPDX-License-Identifier: Apache-2.0

##################### make-report.py ########################################
#
# Copyright 2023 The IBM Research Authors.
#
#############################################################################
#
# This file scan --opt-report=*, --profile-ir, --instrument-onnx-signature
# and process it.
#
# For patterns, see src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.cpp,
# impl::onnxToKrnlParallelReport(...) and impl::onnxToKrnlSimdReport(...)
#
# Format for stats:
# '=='<STAT>'==,' <qualified op name> ',' <node name> ',' <secondary key>
#     {',' <info>}
#
# where <qualified op name> may have a '-simd' or '-par' suffix to indicate
# the success of that pass.
#
# Format of perf:
# '==PERF==,' <op name> ',' <node name> ',' <secondary key> ','
#     <elapsed time in float> ',' <absolute time in f>
#
#############################################################################

import sys
import getopt
import re
import numpy as np


def print_usage(msg=""):
    if msg:
        print("Error:", msg, "\n")
    print(
        """
Usage: Report statistics on compiler and runtime characteristics of ONNX ops.
make-report.py -[vh] [-c <compile_log>] [-r <run_log>] [-l <num>]
      [-p <plot_file_name][-s <stats>] [--sort <val>] [--supported] [-u <val>]
      [-f <op regexp>] [-w <num>]
          
Compile-time statistics are collected from a `onnx-mlir` compiler output
with the `--opt-report` option equal to `Simd` or other supported sub-options.

Runtime statistics are collected from the runtime output of a model compiled.
with the `--profile-ir` option equal to `Onnx` or other supported sub-options.
  
When both compile time and runtime statistics are provided at the same time,
it will correlate the performance metrics with data gathered at compile time.
 
Additional help.
  If you need more specific info on individual success/failure, run
  `onnx-mlir --debug-only=lowering-to-krnl` and look at the compiler output.
  Use `-l 3` to correlate the node name printed here with compiler output.

Parameters on inputs:
  -c/--compile <file_log>: File name containing the compile-time statistics
                           or runtime signature statistics. Default is same
                           as runtime file.
  -r/--runtime <file_log>: File name containing the runtime time statistics.

Parameters to focus analysis:
  -f/--focus <regexp>: Focus only on ops that match the regexp pattern.
  -m/--min <num>:      Focus on operations with at least <num>% of exec time.
  --supported:         Focus only on ops that are supported. Namely, the report
                       will skip ops for which compile-time statistics list
                       the 'unsupported' keyword in its printout.
                       For SIMD/parallel statistics, this include all ops that
                       have currently no support for it.
  -w/--warmup <num>:   If multiple runtime statistics are given, ignore the first
                       <num> stats. Default is zero.

Parameters on what to print:
  -s/--stats <name>:   Print specific statistics:
                       simd: Print simd optimization stats.
                             Default if a compile time file is given.
                       par:  Print parallel optimization stats.
                       sig:  Print signatures of op.
                       perf: Print runtime execution time of ops.
                             Default if no compile time file is given.
  -l/--level <num>:    Print detailed level of statistics:
                       0: Just count successful/unsuccessful ops (default).
                       1: Also count reasons for success/failure.
                       2: Also list metrics.
                       3: Also list node name.
  -p/--plot <name>:    Print a "<name>.jpg" plot of the output.

Parameters on how to print:
  -u/--unit <str>:     Time in second ('s', default), millisecond ('ms') or
                       microsecond ('us).
  --sort <str>:        Sort output by op 'name', occurrence 'num' or `time`.

Help:
  -v/--verbose:        Run in verbose mode (see error and warnings).
  -h/--help:           Print usage.
    """
    )
    exit(1)


################################################################################
# Global info.

# For statistic info.
op_count_dict = {}  # op -> count
op_detail_count_dict = {}  # op -> {dictionary of detailed pattern -> count}
op_time_dict = {}  # op -> cumulative time
op_detail_time_dict = {}  # op -> {dictionary of detailed pattern -> cumulative time}

# For timing info
node_time_dict = {}  # op + node_name -> time statistic
node_time_used = {}  # op + node_name -> 1 if used; not present if unused.
tot_time = 0  # total time.

focus_on_op_with_pattern = r".*"
error_missing_time = 0
supported_only = False
has_timing = False
verbose = False
sorting_preference = ""
report_level = 0  # 0: none; 1: details; 2: extra info; 3: plus node names.
time_unit = 1  # seconds.
min_percent_reporting = 0.0  # percentage.

# For plot info
plot_names = np.array([])
plot_values = np.array([])
plot_x_axis = "time"


# Basic pattern for reports: "==" <stat name> "==," <op name> "," <node name> ","
def common_report_str(stat_name):
    return r"^==" + stat_name + r"-REPORT==,\s*([0-9a-zA-Z\.\-]+)\s*,\s*([^,]*),\s*(.*)"


def match_start_report(line):
    return re.match(r"==START-REPORT==", line)


# ==SIMD-REPORT==, ..., <explanations>, <VL>, <simd-trip-count>
simd_legend = (
    "message, SIMD vector length (in elements), SIMD loop trip count (-1 is runtime)"
)
sig_legend = "comma separated list of shapes with inputs followed by results"

# ==PAR-REPORT==, onnx.Mul-par, Mul_28, elementwise variadic not simdized, 0, -1
par_legend = "message, loop nest (0 outermost), loop trip count (-1 runtime)"

# ==PERF-REPORT==, ..., "before" | "after", time since last call, absolute time
perf_legend = "(after|before), time for op(s), time since start(s)"

################################################################################
# # Support.


# To record time, use op name and node name to better disambiguate.
def get_timing_key(op, node_name):
    p = re.match(r"(.*)-(simd|par)", op)
    if p:
        op = p[1]
    return op + "_=_" + node_name


def get_op_node_from_timing_key(timing_key):
    p = re.match(r"(.*)_=_(.*)", timing_key)
    assert p is not None
    return (p[1], p[2])


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
    return {key2: num}


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
    return {key2: np.array([num])}


def record_pattern(op, node_name, detail_key):
    global op_count_dict, op_detail_count_dict
    global op_time_dict, op_detail_time_dict
    global node_time_dict, node_time_used
    global verbose, report_level, has_timing, error_missing_time

    # Update statistic summaries
    op_count_dict[op] = add_to_dict_entry(op_count_dict, op, 1)
    if report_level > 0:
        op_detail_count_dict[op] = add_to_dict2_entry(
            op_detail_count_dict, op, detail_key, 1
        )

    # Has timing for this node?
    if not has_timing:
        return
    # Process timing.
    timing_key = get_timing_key(op, node_name)
    if not timing_key in node_time_dict:
        error_missing_time += 1
        if verbose:
            print(
                "> timing key",
                timing_key,
                "with no times found in the performance data.",
            )
        return
    # Update timing summaries
    time = node_time_dict[timing_key]
    op_time_dict[op] = append_to_dict_entry(op_time_dict, op, time)
    if report_level > 0:
        op_detail_time_dict[op] = append_to_dict2_entry(
            op_detail_time_dict, op, detail_key, time
        )
    # Record timing key as used.
    node_time_used[timing_key] = 1


################################################################################
# Parse line (generic).


def parse_line(line, report_str, is_perf_stat):
    global focus_on_op_with_pattern, supported_only
    global verbose

    p = re.match(report_str, line)
    if p is None:
        return (False, "", "", "")
    # Have a line of relevant info, extract op, op name, and stat details.
    op = p[1]
    node_name = p[2]
    details = p[3]
    # If we process supported op only, search for "unsupported" in details.
    if supported_only and re.search(r"unsupported", details) is not None:
        return (False, "", "", "")
    # If we process perf, we don't care about the "before"
    if is_perf_stat and re.search(r"before", details) is not None:
        return (False, "", "", "")

    # Check if we have an op that we focus on; if not skip.
    f = re.search(focus_on_op_with_pattern, op)
    if f is None:
        return (False, "", "", "")
    # Have a perfect match.
    return (True, op, node_name, details)


################################################################################
# Parse file for statistics.


def get_secondary_key(node_name, details):
    global report_level

    detail_array = details.split(",")
    if report_level == 1:
        # Use only first element in secondary key.
        return detail_array[0]
    if report_level == 2:
        # Use all details in secondary key.
        return details
    if report_level == 3:
        # Use node name in secondary key.
        return node_name + ", " + details
    return ""


def parse_file_for_stat(file_name, stat_name):
    global node_time_dict, node_time_used
    global report_level, has_timing

    try:
        file = open(file_name, "r")
    except OSError:
        print_usage("Could not open file `" + file_name + "`")

    report_str = common_report_str(stat_name)
    is_perf_stat = re.match(r"PERF", stat_name)
    start_count = 0
    for line in file:
        line = line.rstrip()

        # Only gather stats for the first set of measurement.
        if match_start_report(line):
            start_count += 1
            if start_count >= 2:
                break

        # Parse line.
        (has_stat, op, node_name, details) = parse_line(
            line.rstrip(), report_str, is_perf_stat
        )
        if not has_stat:
            continue
        # Use stat.
        secondary_key = get_secondary_key(node_name, details)
        record_pattern(op, node_name, secondary_key)

    # Continue processing if has timing.
    if not has_timing:
        return
    for timing_key in node_time_dict:
        if timing_key in node_time_used:
            # has seen it
            continue
        (op, node_name) = get_op_node_from_timing_key(timing_key)
        secondary_key = get_secondary_key(node_name, "")
        record_pattern(op, node_name, secondary_key)


################################################################################
# Parse file for performance


def parse_file_for_perf(file_name, stat_name, warmup_num=0):
    global node_time_dict, tot_time
    global verbose, has_timing

    try:
        file = open(file_name, "r")
    except OSError:
        print_usage("Could not open file `" + file_name + "`")

    report_str = common_report_str(stat_name)

    # Skip lines until the first start report
    has_start = False
    start_count = 0
    for line in file:
        line = line.rstrip()
        if match_start_report(line):
            has_start = True
            break  # On to the first measurement.
    if not has_start:
        # Failed to have any timing info, stop.
        return

    time_stat_dict_all_meas = {}  # op+op_name -> numpy array of times
    while has_start:
        time_stat_dict = {}  # op+op_name -> numpy array of times
        has_start = False
        start_count += 1
        meas_tot_time = 0
        for line in file:
            line = line.rstrip()
            # New measurement set?
            if match_start_report(line):
                has_start = True
                break  # On to the next measurement.

            # Parse line.
            (has_stat, op, node_name, details) = parse_line(line, report_str, True)
            if not has_stat:
                continue
            # Keep only after times.
            detail_array = details.split(",")
            key = get_timing_key(op, node_name)
            time = float(detail_array[1])
            meas_tot_time += time
            time_stat_dict[key] = append_to_dict_entry(time_stat_dict, key, time)

        # Has reach the end or a new START_REPORT, summarize this.

        # Normally, an <op>-<node-name> pair should be seen only once in a run,
        # except for loops and some other circumstances (e.g. multiple dim op for
        # a given original onnx op). Because in any case, the report will be done
        # on visiting N time a op-nodename combination that has N instances, and
        # thus adding N times the value from "node_time_dict[node]", we must take
        # here the average. We loose distinguishing the variability of the timing
        # of the same op with same op name, but this is ok. Taking the sum is just
        # wrong, as we would add N times the sum of the N time measurements.
        for node in time_stat_dict:
            time = np.average(time_stat_dict[node])
            time_stat_dict_all_meas[node] = append_to_dict_entry(
                time_stat_dict_all_meas, node, time
            )
        # Add a "tot time" entry.
        time_stat_dict_all_meas["tot_time"] = append_to_dict_entry(
            time_stat_dict_all_meas, "tot_time", meas_tot_time
        )

    # Encountered now all the measurements, has no more.
    meas_num = start_count - warmup_num
    assert meas_num > 0, "expected at least one set of measurement after warmups"
    discard_num = int(meas_num / 4)
    print(
        "Gather stats from",
        start_count,
        "measurement sets with",
        warmup_num,
        "warmup; keep inner",
        meas_num - 2 * discard_num,
        "experiment(s)",
    )
    for node in time_stat_dict_all_meas:
        # Verify that we have the right number of measurements.
        time_array = time_stat_dict_all_meas[node][warmup_num:]
        assert time_array.size == meas_num
        if discard_num > 0 and meas_num - 2 * discard_num > 0:
            time_array = np.sort(time_array)
            time_array = time_array[discard_num:-discard_num]
        node_time_dict[node] = np.average(time_array)
    # Success.
    has_timing = True
    tot_time = node_time_dict["tot_time"]
    del node_time_dict["tot_time"]


################################################################################
# make report


def get_percent(n, d):
    if d == 0.0:
        return 0.0
    return n * 100 / d


def get_sorting_key(count, name, time):
    global sorting_preference
    if sorting_preference == "num":
        return -count
    if sorting_preference == "name":
        return name
    return -time


def make_report(stat_message):
    global op_count_dict, op_detail_count_dict
    global op_time_dict, op_detail_time_dict, tot_time
    global has_timing, time_unit, error_missing_time
    global report_level, supported_only, verbose, min_percent_reporting
    global sorting_preference
    global plot_names, plot_values, plot_x_axis

    # Gather statistics in a dictionary so that we may sort the entries.
    sorted_output = {}
    sorted_plot_output = {}
    for op in op_count_dict:
        count = op_count_dict[op]
        count_time_str = str(count)
        time = 0
        plot_name_val = []
        if op in op_time_dict:
            time = np.sum(op_time_dict[op])
            count_time_str += ", {:.7f}".format(time * time_unit / count)
            count_time_str += ", {:.7f}".format(time * time_unit)
            count_time_str += ", {:.1f}%".format(get_percent(time, tot_time))
            # Keep plot_name -> plot_val in sorted_plot_output.
            plot_name = "{} ({})".format(op, count)
            plot_val = time * time_unit
            plot_name_val = [plot_name, plot_val]
        if get_percent(time, tot_time) < min_percent_reporting:
            continue
        output = "  " + op + ", " + count_time_str
        if report_level:
            det_dict = op_detail_count_dict[op]
            det_time_dict = {}
            sorted_det_output = {}
            if op in op_detail_time_dict:
                det_time_dict = op_detail_time_dict[op]
            for det_key in det_dict:
                det_count = det_dict[det_key]
                if det_count == count:
                    count_time_str = "*"
                else:
                    count_time_str = str(det_count)
                det_time = 0
                if det_key in det_time_dict:
                    det_time = np.sum(det_time_dict[det_key])
                    count_time_str += ", {:.7f}".format(
                        det_time * time_unit / det_count
                    )
                    count_time_str += ", {:.7f}".format(det_time * time_unit)
                    count_time_str += ", {:.1f}%".format(get_percent(det_time, time))

                det_output = "\n    " + count_time_str + ", " + det_key
                det_output_key = get_sorting_key(det_count, det_key, det_time)
                if det_output_key in sorted_det_output:
                    sorted_det_output[det_output_key] += det_output
                else:
                    sorted_det_output[det_output_key] = det_output
            for key in sorted(sorted_det_output):
                output += sorted_det_output[key]

        # Add output to sorted_output.
        output_key = get_sorting_key(count, op, time)
        if output_key in sorted_output:
            sorted_output[output_key] += "\n" + output
        else:
            sorted_output[output_key] = output
        # Add plot name/val tuple to sorted_plot_output.
        if len(plot_name_val) == 2:
            if output_key in sorted_plot_output:
                curr_list = sorted_plot_output[output_key]
                curr_list.append(plot_name_val)
                sorted_plot_output[output_key] = curr_list
            else:
                sorted_plot_output[output_key] = [plot_name_val]

    # Print legend and stats.
    num_desc = "num"
    if has_timing:
        if time_unit == 1:
            unit_str = "s"
        elif time_unit == 1000:
            unit_str = "ms"
        elif time_unit == 1000 * 1000:
            unit_str = "us"
        plot_x_axis = "execution time ({}) out of total time of {:.2f}{}".format(
            unit_str, tot_time * time_unit, unit_str
        )
        unit_str = "(" + unit_str + ")"
        num_desc += ", average time " + unit_str
        num_desc += ", cumulative time " + unit_str
        num_desc += ", percent of total "
    print("Statistic legend:")
    if report_level < 2:
        print("  op-name:", num_desc)
    elif report_level == 2:
        print("   " + num_desc + ",", stat_message, "\n")
    elif report_level == 3:
        print("   " + num_desc + ", node-name, ", stat_message, "\n")
    print("")
    stat_details = ""
    if supported_only:
        stat_details = " supported ops"
    else:
        stat_details = " all ops"
    if min_percent_reporting > 0:
        stat_details += ", " + str(min_percent_reporting) + "%+ exec time"
    stat_details += " ordered_by " + sorting_preference
    if has_timing:
        stat_details += ", tot_time,  {:.7f}".format(tot_time * time_unit)
    print("Statistics start" + stat_details)
    for key in sorted(sorted_output):
        print(sorted_output[key])
        if key in sorted_plot_output:
            for t in sorted_plot_output[key]:
                plot_names = np.append(plot_names, t[0])
                plot_values = np.append(plot_values, t[1])
    print("Statistics end" + stat_details)

    # Report spurious node name if any.
    if error_missing_time:
        print("> Timing information was missing for some of the nodes.")
        print("> Run with `-v` for detailed list of errors.")


################################################################################
# Print plot.


def output_plot(runtime_file_name, plot_file_name):
    global plot_names, plot_values, plot_x_axis
    if len(plot_names) == 0:
        print("\n> No info to plot, skip")
        return
    try:
        import matplotlib.pyplot as plt

        # Create the horizontal bar graph.
        fig, ax = plt.subplots()
        bars = ax.barh(np.flip(plot_names), np.flip(plot_values), color="blue")

        # Adding the data values next to the bars.
        for bar in bars:
            width = bar.get_width()
            text = " {:.1f}".format(width)
            ax.text(width, bar.get_y() + bar.get_height() / 2, text, va="center")

        # Setting the axes limits to make room for annotations.
        ax.set_xlim(0, max(plot_values) * 1.2)

        # Adjust layout to make room for the table:
        # plt.subplots_adjust(left=0.1)
        plt.xlabel(plot_x_axis)
        plt.title("execution time summary for " + runtime_file_name)
        output_file_name = "{}.jpg".format(plot_file_name)
        plt.savefig(output_file_name, bbox_inches="tight")
        print('\n> output plot printed in "{}"'.format(output_file_name))
    except ImportError:
        print("\n> Could not import mathplotlib, please add if you want to plot.")


################################################################################
# Main.


def main(argv):
    global report_level, focus_on_op_with_pattern, supported_only, time_unit
    global min_percent_reporting, verbose
    global sorting_preference

    compile_file_name = ""
    runtime_file_name = ""
    plot_file_name = ""
    make_stats = ""
    make_legend = ""
    warmup_num = 0
    try:
        opts, args = getopt.getopt(
            argv,
            "c:f:hl:m:p:r:s:u:vw:",
            [
                "compile=",
                "focus=",
                "help",
                "level=",
                "min=",
                "plot=",
                "runtime=",
                "stats=",
                "sort=",
                "supported",
                "unit=",
                "verbose",
                "warmup=",
            ],
        )
    except getopt.GetoptError:
        print_usage("Failure to parse inputs")
    for opt, arg in opts:
        if opt in ("-c", "--compile"):
            compile_file_name = arg
        elif opt in ("-f", "--focus"):
            focus_on_op_with_pattern = arg
            if report_level == 0:
                report_level = 1
        elif opt in ("-h", "--help"):
            print_usage()
        elif opt in ("-l", "--level"):
            report_level = int(arg)
            if report_level < 0 or report_level > 3:
                print_usage("detail levels are 0, 1, 2, or 3")
        elif opt in ("-m", "--min"):
            min_percent_reporting = float(arg)
        elif opt in ("-p", "--plot"):
            plot_file_name = arg
        elif opt in ("-r", "--runtime"):
            runtime_file_name = arg
        elif opt in ("-s", "--stats"):
            if re.match(r"\s*par\s*", arg):
                make_stats = "PAR"
                make_legend = par_legend
            elif re.match(r"\s*perf\s*", arg):
                make_stats = "PERF"
                make_legend = perf_legend
            elif re.match(r"\s*sig?\s*", arg):
                make_stats = "SIG"
                make_legend = sig_legend
            elif re.match(r"\s*simd\s*", arg):
                make_stats = "SIMD"
                make_legend = simd_legend
            else:
                print_usage("statistics options are 'par', 'perf', 'sig', or 'simd'")
        elif opt in ("--supported"):
            supported_only = True
        elif opt in ("--sort"):
            if re.match(r"\s*name\s*", arg):
                sorting_preference = "name"
            elif re.match(r"\s*num\s*", arg):
                sorting_preference = "num"
            elif re.match(r"\s*time\s*", arg):
                sorting_preference = "time"
            else:
                print_usage("sorting options are 'name', 'num', or 'time'")
        elif opt in ("-u", "--unit"):
            if re.match(r"\s*s\s*", arg):
                time_unit = 1
            elif re.match(r"\s*ms\s*", arg):
                time_unit = 1000
            elif re.match(r"\s*us\s*", arg):
                time_unit = 1000 * 1000
            else:
                print_usage("time units are 's', 'ms', or 'us'")
        elif opt in ("-v", "--verbose"):
            verbose = True
        elif opt in ("-w", "--warmup"):
            warmup_num = int(arg)

    # Default stats.
    if not make_stats:
        if compile_file_name:
            # Default for compile analysis.
            make_stats = "SIMD"
            make_legend = simd_legend
        else:
            # Default for perf only (no compile)
            make_stats = "PERF"
            make_legend = perf_legend
    print("Analyse", make_stats)
    # Default compile file for sig
    if make_stats != "PERF" and not compile_file_name:
        compile_file_name = runtime_file_name
    # Default sorting preference.
    if not sorting_preference:
        if runtime_file_name:
            sorting_preference = "time"
        else:
            sorting_preference = "name"
    if compile_file_name and runtime_file_name:
        parse_file_for_perf(runtime_file_name, "PERF", warmup_num)
        parse_file_for_stat(compile_file_name, make_stats)
        print(
            'Report using runtime file "'
            + runtime_file_name
            + '" and compile file "'
            + compile_file_name
            + '"'
        )
        make_report(make_legend)
    elif compile_file_name:
        parse_file_for_stat(compile_file_name, make_stats)
        print('Report using compile file "' + compile_file_name + '"')
        make_report(make_legend)
    elif runtime_file_name:
        parse_file_for_perf(runtime_file_name, "PERF", warmup_num)
        parse_file_for_stat(runtime_file_name, "PERF")
        print('Report using runtime file "' + runtime_file_name + '"')
        make_report(make_legend)
    else:
        print_usage("Command requires an input file name (compile/runtime or both).\n")

    if plot_file_name:
        output_plot(runtime_file_name, plot_file_name)


if __name__ == "__main__":
    main(sys.argv[1:])
