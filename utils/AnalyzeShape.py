#!/usr/bin/env python3
"""Fold onnx.DimGroup info into tensor types as a #onnx.dg<> encoding attribute.

Useful for debugging shape inference: dynamic dimensions ("?") that the compiler
has proven are equal show up as separate onnx.DimGroup ops elsewhere in the file,
making it hard to tell at a glance which dims are related. This tool inlines that
info directly into each tensor type, e.g. two dims sharing a group turn from
tensor<?x?xf32> into tensor<d0xu0xf32, #onnx.dg<["d0", "u0"]>>, so it's clear
right where the type is used that those dims are the same size.

Runs the onnx-dim-analysis pass on <input.mlir> to annotate dynamic dimensions
with onnx.DimGroup ops, saving that intermediate as <input-name>-dg.mlir, then
folds the group info into a #onnx.dg<> encoding attribute and writes the result
to <input-name>-annotated.mlir.

To generate a suitable input.mlir:
  1) Run onnx-mlir with --EmitONNXIR or --EmitZHighIR to produce a .mlir file.
  2) Pass that file to AnalyzeShape.py.

Locating the onnx-mlir-opt binary:
  Set ONNX_MLIR_HOME to the onnx-mlir build dir (e.g. path-to-onnx-mlir/build/Debug,
  the parent folder containing bin/onnx-mlir-opt), or make sure onnx-mlir-opt is on PATH.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys

from mlir_log_utils import (
    clean_line,
    VALUE,
    Annotator,
    strip_suffix,
    sanitize_ident,
    find_matching_close,
    split_top_level,
    handle_func_signature,
    handle_generic_op,
    handle_return,
    handle_pretty_result,
    apply_comments,
)

DIMGROUP_RE = re.compile(
    r'^\s*"onnx\.DimGroup"\((' + VALUE + r")\)\s*<\{axis\s*=\s*(-?\d+)\s*:\s*si64,"
    r"\s*group_id\s*=\s*(-?\d+)\s*:\s*si64\}>\s*:\s*\([^()]*\)\s*->\s*\(\)\s*$"
)
DIM_PARAMS_ARG_RE = re.compile(
    r"("
    + VALUE
    + r")(?:/\d+)?\s*:\s*tensor<[^<>]*>\s*\{[^{}]*onnx\.dim_params\s*=\s*\"([^\"]*)\""
)
RETURN_VALUES_RE = re.compile(
    r"onnx\.Return\s+(" + VALUE + r"(?:\s*,\s*" + VALUE + r")*)"
)


def parse_dimgroups(lines):
    axis_to_group = {}
    for line in lines:
        m = DIMGROUP_RE.match(line)
        if m:
            axis_to_group[(strip_suffix(m.group(1)), int(m.group(2)))] = int(m.group(3))
    return axis_to_group


def parse_dim_params(lines):
    dim_params = {}
    for line in lines:
        for m in DIM_PARAMS_ARG_RE.finditer(line):
            value = strip_suffix(m.group(1))
            for entry in m.group(2).split(","):
                entry = entry.strip()
                if ":" not in entry:
                    continue
                axis_str, name = entry.split(":", 1)
                axis_str, name = axis_str.strip(), name.strip()
                if axis_str.lstrip("-").isdigit() and name:
                    dim_params[(value, int(axis_str))] = sanitize_ident(name)
    return dim_params


def build_name_map(axis_to_group, dim_params):
    name_of_group = {}
    for (value, axis), group in axis_to_group.items():
        name = dim_params.get((value, axis))
        if name and group not in name_of_group:
            name_of_group[group] = name
    return name_of_group


def parse_return_values(lines):
    for line in lines:
        m = RETURN_VALUES_RE.search(line)
        if m:
            return [strip_suffix(v.strip()) for v in split_top_level(m.group(1), ",")]
    return []


def total_elements_from_shape(dims):
    total = 1
    for d in dims:
        if not re.fullmatch(r"-?\d+", d):
            return None
        total *= int(d)
    return total


def elements_after(line, pos):
    m = re.match(r"\s*:\s*tensor<([^<>]*)>", line[pos:])
    if not m:
        return None
    dims = m.group(1).split("x")[:-1]
    return total_elements_from_shape(dims)


def elide_long_constants(line):
    out = []
    i, n = 0, len(line)
    while i < n:
        idx_bracket = line.find("dense<[", i)
        idx_hex = line.find('dense<"0x', i)
        candidates = [v for v in (idx_bracket, idx_hex) if v != -1]
        if not candidates:
            out.append(line[i:])
            break
        idx = min(candidates)
        out.append(line[i:idx])
        if idx == idx_bracket:
            open_idx = idx + len("dense<")
            close_idx = find_matching_close(line, open_idx, "[", "]")
            if close_idx == -1:
                out.append(line[idx:])
                break
            gt_match = re.match(r"\s*>", line[close_idx + 1 :])
            if not gt_match:
                out.append(line[idx : close_idx + 1])
                i = close_idx + 1
                continue
            gt_idx = close_idx + 1 + gt_match.end()
            literal_inner = line[open_idx + 1 : close_idx]
            count = (
                0
                if literal_inner.strip() == ""
                else len(split_top_level(literal_inner, ","))
            )
            elements = elements_after(line, gt_idx)
            elide = elements > 20 if elements is not None else count > 20
        else:
            quote_start = idx + len("dense<")
            close_quote = line.find('"', quote_start + 1)
            if close_quote == -1:
                out.append(line[idx:])
                break
            gt_match = re.match(r"\s*>", line[close_quote + 1 :])
            if not gt_match:
                out.append(line[idx : close_quote + 1])
                i = close_quote + 1
                continue
            gt_idx = close_quote + 1 + gt_match.end()
            hex_str = line[quote_start + 1 : close_quote]
            hex_digits = len(hex_str) - 2 if hex_str.startswith("0x") else len(hex_str)
            elements = elements_after(line, gt_idx)
            elide = elements > 20 if elements is not None else hex_digits > 40
        out.append("dense_resource<__elided__>" if elide else line[idx:gt_idx])
        i = gt_idx
    return "".join(out)


def find_onnx_mlir_opt():
    exe_name = "onnx-mlir-opt.exe" if sys.platform == "win32" else "onnx-mlir-opt"

    onnx_mlir_home = os.environ.get("ONNX_MLIR_HOME")
    if onnx_mlir_home:
        candidate = os.path.join(onnx_mlir_home, "bin", exe_name)
        if os.path.isfile(candidate):
            return candidate

    path = shutil.which(exe_name)
    if path:
        return path

    raise RuntimeError(
        "Cannot find onnx-mlir-opt binary. Please either:\n"
        " 1) Set environment variable ONNX_MLIR_HOME to the path to onnx-mlir\n"
        "    (e.g., path-to-onnx-mlir/build/Debug, the parent folder containing bin, lib, etc)\n"
        " 2) Add onnx-mlir-opt to your PATH"
    )


def uses_zhigh_dialect(input_path):
    with open(input_path) as f:
        return re.search(r"\bzhigh\.", f.read()) is not None


def run_dim_group_annotation_pass(input_path, dg_path):
    """Run the onnx-dim-analysis pass, which annotates dynamic dimensions with
    onnx.DimGroup ops, and save the result to dg_path."""
    onnx_mlir_opt = find_onnx_mlir_opt()
    cmd = [onnx_mlir_opt, input_path, "--onnx-dim-analysis"]
    if uses_zhigh_dialect(input_path):
        cmd += ["--march=z17", "--maccel=NNPA"]
    print("// started onnx-mlir-opt dependence analysis", file=sys.stderr)
    with open(dg_path, "w") as out:
        subprocess.run(cmd, stdout=out, check=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input",
        metavar="input.mlir",
        help="mlir file to investigate (an isolated onnx/zhigh pass)",
    )
    parser.add_argument(
        "-c",
        "--comments",
        action="store_true",
        help="Add useful comments to the annotated listing: use-count for each def; "
        "values for scalar constant (same transform as IsolatePass.py -c).",
    )
    parser.add_argument(
        "-w",
        "--wrap",
        action="store_true",
        help="Reformat long func signatures / ops / returns across multiple indented "
        "lines when they exceed the line-length limit. Off by default: without this "
        "flag, annotated lines are kept as single lines no matter how long.",
    )
    args = parser.parse_args()

    base, _ = os.path.splitext(args.input)
    dg_path = base + "-dg.mlir"
    annotated_path = base + "-annotated.mlir"

    run_dim_group_annotation_pass(args.input, dg_path)
    print(f"Wrote {dg_path}", file=sys.stderr)

    with open(dg_path) as f:
        lines = f.read().splitlines()

    axis_to_group = parse_dimgroups(lines)
    dim_params = parse_dim_params(lines)
    name_of_group = build_name_map(axis_to_group, dim_params)
    return_values = parse_return_values(lines)
    ann = Annotator(axis_to_group, name_of_group)

    print("// started annotation", file=sys.stderr)
    out_lines = [
        "// -----// IR Dump After (anonymous namespace)::ONNXDimAnalysisPass (onnx-dim-analysis) //----- //"
    ]
    for line in lines:
        if DIMGROUP_RE.match(line):
            continue
        line = clean_line(elide_long_constants(line))
        new_line = (
            handle_func_signature(line, ann, return_values, wrap=args.wrap)
            or handle_generic_op(line, ann, wrap=args.wrap)
            or handle_return(line, ann, wrap=args.wrap)
            or handle_pretty_result(line, ann, wrap=args.wrap)
            or line
        )
        out_lines.append(new_line)

    text = "\n".join(out_lines) + "\n"
    if args.comments:
        text = apply_comments(text)

    with open(annotated_path, "w") as f:
        f.write(text)
    print(f"Wrote {annotated_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
