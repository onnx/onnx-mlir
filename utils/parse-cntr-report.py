#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

##################### parse-cntr-report.py ####################################
#
# Copyright 2026 The IBM Research Authors.
#
################################################################################
#
# Summarize ==CNTR-REPORT== lines from an onnx-mlir profiling log. Each line
# has the form:
#
#   ==CNTR-REPORT==, <op name>, <node name>, before|after, <counter name>, <value>
#
# "before" and "after" lines for the same op/node/counter bracket one op
# invocation; this script pairs them up (in the order they appear) and sums
# (after - before) across all invocations, per (op name, node name, counter).
#
# The log may contain multiple runs, each starting with a "==START-REPORT=="
# line. The first -w/--warmup runs are dropped; the remaining runs are each
# summarized independently and then averaged cell-by-cell.
#
# Output is one row per (op name, node name), with one column per counter,
# followed by two summary rows:
#   sum   - the sum, per counter, of every (op, node) row above it (after
#           averaging across the kept runs).
#   total - per counter, the last "after" value seen in a run minus the
#           first "before" value seen in that run, averaged across the kept
#           runs. This is the net change over a full run regardless of which
#           op/node it's attributed to, so comparing it against "sum" is a
#           useful sanity check (e.g. a counter logged twice per op will
#           make "sum" roughly double "total").
#
# With -p/--percent, each (op, node) row is shown as a percentage of its
# column's "sum" value instead of the raw counter value (one decimal place).
# The "sum" and "total" rows still show raw values.
#
################################################################################

import argparse
import csv
import sys
from collections import defaultdict, deque

MARKER_CNTR = "==CNTR-REPORT=="
MARKER_START = "==START-REPORT=="


def split_runs(path):
    """Split a profile log into runs, each delimited by a "==START-REPORT=="
    line. Returns a list of runs, each a list of (lineno, line) pairs for
    lines belonging to that run (the START-REPORT line itself is not
    included). If the file has no START-REPORT markers, the whole file is
    treated as a single run.
    """
    runs = []
    current = None
    with open(path) as f:
        for lineno, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if line.strip().startswith(MARKER_START):
                current = []
                runs.append(current)
                continue
            if current is None:
                current = []
                runs.append(current)
            current.append((lineno, line))
    return runs


def parse_run(path, lines, run_label):
    """Parse the ==CNTR-REPORT== lines of a single run.

    Returns a tuple:
      accumulated    - dict[(op_name, node_name, counter_name)] -> summed
                        (after - before) across all invocations in this run.
      node_order     - list of unique (op_name, node_name) tuples, in the
                        order they first appear in this run.
      counter_order  - list of unique counter names, in the order they first
                        appear in this run.
      total          - dict[counter_name] -> (last "after" seen in this run)
                        minus (first "before" seen in this run).

    Pairing of before/after is FIFO per (op_name, node_name, counter_name):
    each "before" value is queued, and the next "after" for that same key is
    diffed against the oldest still-queued "before". This keeps repeated
    invocations of the same op/node, and repeated counters within a single
    before/after block, matched to the right pair.
    """
    pending_before = defaultdict(deque)
    accumulated = defaultdict(int)
    node_order = []
    seen_nodes = set()
    counter_order = []
    seen_counters = set()
    first_before = {}
    last_after = {}

    for lineno, line in lines:
        line = line.strip()
        if not line.startswith(MARKER_CNTR):
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6:
            print(
                f"warning: {path}:{lineno} ({run_label}): expected 6 fields, "
                f"got {len(fields)}; skipping",
                file=sys.stderr,
            )
            continue
        _, op_name, node_name, phase, counter_name, value = fields
        node_key = (op_name, node_name)
        if node_key not in seen_nodes:
            seen_nodes.add(node_key)
            node_order.append(node_key)
        if counter_name not in seen_counters:
            seen_counters.add(counter_name)
            counter_order.append(counter_name)

        key = (op_name, node_name, counter_name)
        try:
            value = int(value)
        except ValueError:
            value = float(value)

        if phase == "before":
            pending_before[key].append(value)
            if counter_name not in first_before:
                first_before[counter_name] = value
        elif phase == "after":
            if not pending_before[key]:
                print(
                    f"warning: {path}:{lineno} ({run_label}): 'after' with no "
                    f"matching 'before' for {key}; skipping",
                    file=sys.stderr,
                )
                continue
            before_value = pending_before[key].popleft()
            accumulated[key] += value - before_value
            last_after[counter_name] = value
        else:
            print(
                f"warning: {path}:{lineno} ({run_label}): unknown phase "
                f"{phase!r}; skipping",
                file=sys.stderr,
            )

    total = {
        counter_name: last_after[counter_name] - first_before[counter_name]
        for counter_name in counter_order
        if counter_name in first_before and counter_name in last_after
    }
    return accumulated, node_order, counter_order, total


def combine_runs(parsed_runs):
    """Average a list of per-run (accumulated, node_order, counter_order,
    total) tuples into a single (accumulated, node_order, counter_order,
    total) tuple, cell-by-cell.

    A run that never saw a given (op, node, counter) contributes 0 for that
    cell (rather than being excluded), since a missing entry means that op
    simply didn't fire in that run. A run missing a counter's total (no
    before/after pair anywhere) is excluded only from that counter's total
    average.
    """
    n_runs = len(parsed_runs)
    node_order = []
    seen_nodes = set()
    counter_order = []
    seen_counters = set()
    accum_sum = defaultdict(float)
    total_sum = defaultdict(float)
    total_count = defaultdict(int)

    for accumulated, nodes, counters, total in parsed_runs:
        for node in nodes:
            if node not in seen_nodes:
                seen_nodes.add(node)
                node_order.append(node)
        for counter_name in counters:
            if counter_name not in seen_counters:
                seen_counters.add(counter_name)
                counter_order.append(counter_name)
        for key, value in accumulated.items():
            accum_sum[key] += value
        for counter_name, value in total.items():
            total_sum[counter_name] += value
            total_count[counter_name] += 1

    accumulated = {key: value / n_runs for key, value in accum_sum.items()}
    total = {
        counter_name: total_sum[counter_name] / total_count[counter_name]
        for counter_name in counter_order
        if total_count[counter_name] > 0
    }
    return accumulated, node_order, counter_order, total


def parse_cntr_report(path, warmup=0):
    """Parse a profile log, dropping the first `warmup` runs (each run
    delimited by a "==START-REPORT==" line) and averaging the rest.

    Returns the same shape as parse_run: (accumulated, node_order,
    counter_order, total), averaged across the kept runs.
    """
    runs = split_runs(path)
    if warmup >= len(runs):
        sys.exit(
            f"Error: --warmup {warmup} >= number of runs ({len(runs)}) found "
            f"in {path}"
        )
    kept_runs = runs[warmup:]
    parsed_runs = [
        parse_run(path, lines, f"run {warmup + i + 1}")
        for i, lines in enumerate(kept_runs)
    ]
    return combine_runs(parsed_runs)


def _fmt_number(value):
    """Format a possibly-averaged number: whole numbers print bare, others
    get 2 decimal places."""
    if value == int(value):
        return str(int(value))
    return f"{value:.2f}"


def build_table(accumulated, node_order, counter_order, total, percent=False):
    """Return (headers, rows) where each row is
    [op_name, node_name, value_for_counter_1, ...], followed by a 'sum' row
    (column-wise sum over all (op, node) rows) and a 'total' row (per
    counter, averaged last-after-minus-first-before across the kept runs).

    When percent is True, each (op, node) row's counter cells are rendered
    as a percentage of that column's sum (one decimal place) instead of the
    raw value; the 'sum' and 'total' rows are still raw values.
    """
    headers = ["op_name", "node_name"] + counter_order

    sums = {
        counter_name: sum(
            accumulated.get((op_name, node_name, counter_name), 0)
            for op_name, node_name in node_order
        )
        for counter_name in counter_order
    }

    rows = []
    for op_name, node_name in node_order:
        row = [op_name, node_name]
        for counter_name in counter_order:
            value = accumulated.get((op_name, node_name, counter_name), 0)
            if percent:
                col_sum = sums[counter_name]
                pct = (value / col_sum * 100) if col_sum else 0.0
                row.append(f"{pct:.1f}%")
            else:
                row.append(_fmt_number(value))
        rows.append(row)

    sum_row = ["sum", ""] + [_fmt_number(sums[c]) for c in counter_order]
    rows.append(sum_row)

    total_row = ["total", ""]
    for counter_name in counter_order:
        if counter_name in total:
            total_row.append(_fmt_number(total[counter_name]))
        else:
            total_row.append("")
    rows.append(total_row)

    return headers, rows


def main():
    parser = argparse.ArgumentParser(
        description="Summarize ==CNTR-REPORT== perf counters from an "
        "onnx-mlir profiling log into a table of (op, node) x counter "
        "totals, plus 'sum' and 'total' summary rows. Handles logs with "
        "multiple ==START-REPORT== runs, skipping warmup runs and "
        "averaging the rest."
    )
    parser.add_argument("logfile", help="path to the profile log file")
    parser.add_argument(
        "-w",
        "--warmup",
        type=int,
        default=0,
        help="number of leading runs (each delimited by ==START-REPORT==) "
        "to treat as warmup and exclude from the average (default: 0)",
    )
    parser.add_argument(
        "-p",
        "--percent",
        action="store_true",
        help="show each (op, node) row's counter values as a percentage of "
        "that column's sum, instead of the raw value",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="print output as CSV instead of an aligned table",
    )
    args = parser.parse_args()

    accumulated, node_order, counter_order, total = parse_cntr_report(
        args.logfile, warmup=args.warmup
    )
    if not accumulated:
        print(f"No ==CNTR-REPORT== entries found in {args.logfile}", file=sys.stderr)
        sys.exit(1)

    headers, rows = build_table(
        accumulated, node_order, counter_order, total, percent=args.percent
    )

    if args.csv:
        writer = csv.writer(sys.stdout)
        writer.writerow(headers)
        writer.writerows(rows)
        return

    widths = [
        max(len(headers[i]), max(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]
    fmt = (
        "  ".join(f"{{:<{w}}}" for w in widths[:2])
        + "  "
        + "  ".join(f"{{:>{w}}}" for w in widths[2:])
    )
    print(fmt.format(*headers))
    print(fmt.format(*("-" * w for w in widths)))
    for row in rows:
        print(fmt.format(*row))


if __name__ == "__main__":
    main()
