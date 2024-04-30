#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

##################### make-timechart.py #######################################
#
# Copyright 2019-2023 The IBM Research Authors.
#
################################################################################
#
# Generate a timechart graph from an intstrumentation file.
#
################################################################################
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import numpy as np
import argparse
import sys

default_xscale = -1.0  # auto scale
default_xscales = [0.01, 0.02, 0.05, 0.10, 0.20, 0.25, 0.50, 1.0, 2.0, 2.5, 5.0, 10.0]
default_number_of_lines = -1  # auto scale
max_number_of_lines = 20
default_start_time = 0.0
default_period = 200.0
max_number_of_legend = 18
number_of_xticks = 5

elapsed_time_min = -1.0
elapsed_time_max = -1.0
epsilon = 1.0e-10

op_time_dic = {}
op_count_dic = {}
op_index_dic = {}
op_tbl = []
time_tbl = []
label_tbl = []
colormap_tbl = []
handle_tbl = []


# Read instrumentation file
def read_inst_file(inst_file, iteration, data_start_time, data_period):
    global elapsed_time_min, elapsed_time_max
    data_end_time = data_start_time + data_period
    inst_data_tbl = []
    before_elapsed_time_dic = {}
    curr_iteration = 0
    with open(inst_file) as f:
        for line in f:
            columns = line.split(",")
            first_column = columns[0].strip()
            if first_column == "==START-REPORT==":
                if iteration > curr_iteration:
                    break
                if iteration < curr_iteration:
                    op_time_dic.clear()
                    op_count_dic.clear()
                    before_elapsed_time_dic.clear()
                    inst_data_tbl.clear()
                    elapsed_time_min = -1.0
                    elapsed_time_max = -1.0
                curr_iteration += 1
            if first_column != "==PERF-REPORT==":
                continue
            if len(columns) != 6:
                print("SKIP incomplete line [{}]".format(line), file=sys.stderr)
                continue
            op = columns[1].strip()
            node = columns[2].strip()
            before_after = columns[3].strip()
            time = float(columns[4].strip())
            elapsed_time = float(columns[5].strip())
            if elapsed_time_min < -1.0:  # reset
                elapsed_time_min = elapsed_time
                elapsed_time_max = elapsed_time
            elapsed_time_max = max(elapsed_time_max, elapsed_time)
            key = op + ":" + node
            if before_after == "before":
                before_elapsed_time_dic[key] = elapsed_time
                if not op in op_time_dic:
                    op_time_dic[op] = 0.0
                    op_count_dic[op] = 0
                continue
            # before_after == "after"
            if (
                key in before_elapsed_time_dic
                and before_elapsed_time_dic[key] <= data_end_time
                and elapsed_time >= data_start_time
            ):
                inst_data_tbl.append((key, before_elapsed_time_dic[key], elapsed_time))
                op_time_dic[op] += elapsed_time - before_elapsed_time_dic[key]
                op_count_dic[op] += 1
            # else:
            #    print("WARNING: no corresponding line for [{}]".format(line), file=sys.stderr)
    return inst_data_tbl


def write_line_for_op(op, start, end, ax, idx, xscale, number_of_lines):
    line_height = number_of_lines * 0.02
    color = colormap_tbl[idx]
    handle = None
    while start + epsilon < end:
        y, x = divmod(start + epsilon, xscale)
        next_start = min(end, (y + 1) * xscale)
        handle = ax.barh(
            left=x,
            y=number_of_lines - 1 - y,
            width=next_start - start,
            height=line_height,
            align="center",
            color=color,
            label=op,
            alpha=1.0,
            linewidth=0.4,
            ec="k",
        )
        start = next_start
    if handle:
        handle_tbl[idx] = handle
    return


def generate_timechart(
    inst_data_tbl,
    graph,
    xscale,
    number_of_lines,
    data_start_time,
    data_period,
):
    data_end_time = data_start_time + data_period
    frame_start_time = int(data_start_time / xscale) * xscale
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.3, right=0.95)
    # generate colormap
    op_time_list = sorted(op_time_dic.items(), key=lambda x: x[1], reverse=True)
    # color_table = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    total_time = sum(list(map(lambda x: x[1], op_time_list)))
    for idx, op_time in enumerate(op_time_list):
        op, time = op_time
        op_index_dic[op] = idx
        op_tbl.append(op)
        time_tbl.append(time)
        # colormap_tbl.append(color_table[idx % len(color_table)])
        colormap_tbl.append(matplotlib.cm.tab20(idx % 20))
        handle_tbl.append(None)
        label_tbl.append(
            "{}: {:.3f}s / {}".format(op[:13], op_time_dic[op], op_count_dic[op])
        )
        if args.print_summary:
            print("{} {:.6f} {}".format(op, op_time_dic[op], op_count_dic[op]))
    # plot inst_data
    used_op_idx_list = []
    for inst_data in inst_data_tbl:
        key, start, end = inst_data
        start = max(start, data_start_time) - frame_start_time
        end = min(end, data_end_time) - frame_start_time
        op = key[: key.find(":")]
        idx = op_index_dic[op]
        write_line_for_op(op, start, end, ax, idx, xscale, number_of_lines)

    # ax.set_xlabel("Elapsed Time (sec)")
    ax.set_xlim(0.0, xscale)
    ax.xaxis.set_major_formatter("{x:+.3f}")
    yticks = range(number_of_lines)
    yticks_labels = list(
        map(
            lambda x: "{:.02f}".format(
                frame_start_time + (number_of_lines - 1 - x) * xscale
            ),
            yticks,
        )
    )
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks_labels)
    ax.set_ylim(-0.7, number_of_lines)
    ax.set_ylabel("Elapsed Time (sec)")
    title = args.title if args.title else "Time Chart [{}]".format(args.instrumentation)
    ax.set_title(title)
    handles = []
    labels = []
    for idx, hdl in enumerate(handle_tbl):
        if hdl:
            handles.append(handle_tbl[idx])
            labels.append(label_tbl[idx])
    ax.legend(
        handles=handles[:max_number_of_legend],
        labels=labels[:max_number_of_legend],
        handler_map={tuple: HandlerTuple(ndivide=None)},
        frameon=False,
        loc="lower center",
        fontsize="x-small",
        ncol=3,
        bbox_to_anchor=(0.5, -0.5),
    )
    # handler_map={tuple: HandlerTuple(ndivide=None)}, ncol=1, colmnspacing=1
    plt.grid(True)
    plt.savefig(graph)


parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--instrumentation",
    type=str,
    default="",
    help="Path to instrumentation file to be read",
)
parser.add_argument(
    "--print-summary",
    action="store_true",
    help="Flag to printout operation time summary (default=False)",
)
parser.add_argument(
    "-g",
    "--graph",
    type=str,
    default="",
    help="Path to graph file to be generated",
)
parser.add_argument(
    "-t",
    "--title",
    type=str,
    default="",
    help="Title of the graph",
)
parser.add_argument(
    "--start-time",
    type=float,
    default=default_start_time,
    help="Start time (sec) (default={})".format(default_start_time),
)
parser.add_argument(
    "--period",
    type=float,
    default=default_period,
    help="Period (sec) (default={})".format(default_period),
),
parser.add_argument(
    "--xscale",
    type=float,
    default=default_xscale,
    help="X-axis scale (default={})".format(default_xscale),
)
parser.add_argument(
    "--number-of-lines",
    type=int,
    default=default_number_of_lines,
    help="Number of timechart lines (default={})".format(default_number_of_lines),
)
parser.add_argument(
    "--iteration",
    type=int,
    default=-1,
    help="Iteration number starting from 0 (default=last)",
)

args = parser.parse_args()
if not args.instrumentation:
    print("error: no instrumentation file, use argument --inst")
    print(parser.format_usage())
    exit(1)
if not args.graph:
    print("error: no graph file, use argument --graph")
    print(parser.format_usage())
    exit(1)


def main():
    inst_data_tbl = read_inst_file(
        args.instrumentation, args.iteration, args.start_time, args.period
    )
    xscale = args.xscale
    if args.xscale < 0:  # set xscale with auto scale
        for scale in default_xscales:
            if scale * max_number_of_lines >= elapsed_time_max:
                break
        xscale = scale
    number_of_lines = args.number_of_lines
    if number_of_lines < 0:  # set number_of_lines with auto scale
        number_of_lines = min(
            int((elapsed_time_max + xscale - epsilon) / xscale), max_number_of_lines
        )
    generate_timechart(
        inst_data_tbl,
        args.graph,
        xscale,
        number_of_lines,
        args.start_time,
        args.period,
    )


if __name__ == "__main__":
    main()
