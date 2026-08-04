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

from mlir_log_utils import clean_line

VALUE = r"%[\w.]+"

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
FUNC_HEAD_RE = re.compile(r"func\.func\s+@[\w]+\s*\(")
GENERIC_OP_HEAD_RE = re.compile(
    r"^(\s*(?:" + VALUE + r"(?:\s*,\s*" + VALUE + r')*\s*=\s*)?)"[\w.]+"\('
)
RETURN_RE = re.compile(
    r"^(\s*onnx\.Return\s+)(" + VALUE + r"(?:\s*,\s*" + VALUE + r")*)(\s*:\s*)(.+?)\s*$"
)
PRETTY_RESULT_LHS_RE = re.compile(
    r"^(\s*(?:" + VALUE + r"(?:\s*,\s*" + VALUE + r")*\s*=\s*)?)(.*?):\s*$"
)


def tensor_type_span(text, start=0):
    """Find `tensor<...>` in text starting at/after start, matching nested `<>` (e.g. zhigh
    layout encodings). Returns (start_idx, open_idx, close_idx) of the outer tensor<...>, or
    None if not found."""
    idx = text.find("tensor<", start)
    if idx == -1:
        return None
    open_idx = idx + len("tensor")
    close_idx = find_matching_close(text, open_idx, "<", ">")
    if close_idx == -1:
        return None
    return idx, open_idx, close_idx


def strip_suffix(name):
    return re.sub(r"/\d+$", "", name)


def sanitize_ident(name):
    s = re.sub(r"[^0-9a-zA-Z_]+", "_", name.strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "dim"


def find_matching_close(s, open_idx, open_ch, close_ch):
    depth = 0
    i = open_idx
    n = len(s)
    in_str = False
    while i < n:
        c = s[i]
        if in_str:
            if c == "\\":
                i += 1
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == open_ch:
            depth += 1
        elif c == close_ch:
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def split_top_level(s, sep=","):
    parts = []
    depth = 0
    in_str = False
    start = 0
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if in_str:
            if c == "\\":
                i += 1
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c == "-" and i + 1 < n and s[i + 1] == ">":
            i += 1  # skip the "->" arrow so its '>' isn't read as a closing bracket
        elif c in "([{<":
            depth += 1
        elif c in ")]}>":
            depth -= 1
        elif c == sep and depth == 0:
            parts.append(s[start:i])
            start = i + 1
        i += 1
    parts.append(s[start:])
    return parts


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


class Annotator:
    def __init__(self, axis_to_group, name_of_group):
        self.axis_to_group = axis_to_group
        self.name_of_group = name_of_group
        self.group_renumber = {}
        self.next_id = 0

    def annotate(self, inner, value):
        if value is None:
            return None
        parts = inner.split("x")
        if len(parts) < 2:
            return None
        dims = parts[:-1]
        if not dims or any(d == "*" for d in dims):
            return None
        tokens = []
        changed = False
        for axis, d in enumerate(dims):
            if d != "?":
                tokens.append(d)
                continue
            group = self.axis_to_group.get((value, axis))
            if group is None:
                tokens.append("?")
                continue
            first_seen = group not in self.group_renumber
            if first_seen:
                self.group_renumber[group] = self.next_id
                self.next_id += 1
            name = self.name_of_group.get(group)
            if name:
                tokens.append(name)
            else:
                tokens.append(
                    f"d{self.group_renumber[group]}"
                    if first_seen
                    else f"u{self.group_renumber[group]}"
                )
            changed = True
        if not changed:
            return None
        dg = "#onnx.dg<[" + ", ".join(f'"{t}"' for t in tokens) + "]>"
        return f"tensor<{inner}, {dg}>"

    def annotate_str(self, type_text, value):
        t = type_text.strip()
        span = tensor_type_span(t)
        if not span or span[0] != 0 or span[2] != len(t) - 1:
            return type_text
        _, open_idx, close_idx = span
        replacement = self.annotate(t[open_idx + 1 : close_idx], value)
        return replacement if replacement else t


def unwrap_parens(text):
    t = text.strip()
    if (
        t.startswith("(")
        and t.endswith(")")
        and find_matching_close(t, 0, "(", ")") == len(t) - 1
    ):
        return t[1:-1], True
    return t, False


MAX_LINE = 120
INDENT_UNIT = "    "


def indent_of(line):
    return re.match(r"^\s*", line).group()


def wrap_items(items, base_indent):
    inner_indent = base_indent + INDENT_UNIT
    pieces = [it + ("," if i < len(items) - 1 else "") for i, it in enumerate(items)]

    lines = []
    current = []
    current_len = len(inner_indent)
    for piece in pieces:
        extra = len(piece) + (1 if current else 0)
        if current and current_len + extra > MAX_LINE:
            lines.append(inner_indent + " ".join(current))
            current = []
            current_len = len(inner_indent)
            extra = len(piece)
        current.append(piece)
        current_len += extra
    if current:
        lines.append(inner_indent + " ".join(current))
    return "\n".join(lines)


def handle_func_signature(line, ann, return_values):
    m = FUNC_HEAD_RE.search(line)
    if not m:
        return None
    open_idx = m.end() - 1
    close_idx = find_matching_close(line, open_idx, "(", ")")
    if close_idx == -1:
        return None
    rest = line[close_idx + 1 :]
    if "->" not in rest or not rest.rstrip().endswith("{"):
        return None
    arrow_idx = rest.index("->")
    ret_section = rest[arrow_idx + 2 :]
    brace_pos = ret_section.rfind("{")
    ret_types_text, trailing = ret_section[:brace_pos], ret_section[brace_pos:]

    args_text = line[open_idx + 1 : close_idx]
    arg_items = [
        process_arg_chunk(chunk, ann).strip()
        for chunk in split_top_level(args_text, ",")
    ]

    inner, wrapped = unwrap_parens(ret_types_text)
    ret_type_list = split_top_level(inner, ",") if inner.strip() else []
    ret_items = [
        ann.annotate_str(
            t, return_values[i] if i < len(return_values) else None
        ).strip()
        for i, t in enumerate(ret_type_list)
    ]
    new_ret_text = ", ".join(ret_items)
    if wrapped:
        new_ret_text = f"({new_ret_text})"

    oneline = (
        line[: open_idx + 1]
        + ", ".join(arg_items)
        + ") -> "
        + new_ret_text
        + " "
        + trailing
    )
    if len(oneline) <= MAX_LINE:
        return oneline

    base_indent = indent_of(line)
    parts = [line[: open_idx + 1].rstrip()]
    parts.append(wrap_items(arg_items, base_indent))
    if wrapped:
        parts.append(f"{base_indent}) -> (")
        parts.append(wrap_items(ret_items, base_indent))
        parts.append(f"{base_indent}) {trailing}")
    else:
        parts.append(f"{base_indent}) -> {new_ret_text} {trailing}")
    return "\n".join(parts)


def process_arg_chunk(chunk, ann):
    m = re.match(r"^(\s*)(" + VALUE + r")(?:/\d+)?(\s*:\s*)", chunk)
    if not m:
        return chunk
    lead, name, colon = m.groups()
    rest = chunk[m.end() :]
    span = tensor_type_span(rest)
    if not span or span[0] != 0:
        return chunk
    _, open_idx, close_idx = span
    typetext = rest[: close_idx + 1]
    tail = rest[close_idx + 1 :]
    replacement = ann.annotate(rest[open_idx + 1 : close_idx], name)
    return f"{lead}{name}{colon}{replacement if replacement else typetext}{tail}"


def handle_generic_op(line, ann):
    m = GENERIC_OP_HEAD_RE.match(line)
    if not m:
        return None
    open_idx = m.end() - 1
    close_idx = find_matching_close(line, open_idx, "(", ")")
    if close_idx == -1:
        return None
    operand_text = line[open_idx + 1 : close_idx]
    operand_names = (
        [strip_suffix(x.strip()) for x in split_top_level(operand_text, ",")]
        if operand_text.strip()
        else []
    )

    rest = line[close_idx + 1 :]
    sig = re.search(r":\s*\(([^()]*)\)\s*->\s*(.+?)\s*$", rest)
    if not sig:
        return None

    op_type_list = split_top_level(sig.group(1), ",") if sig.group(1).strip() else []
    operand_items = [
        ann.annotate_str(
            t, operand_names[i] if i < len(operand_names) else None
        ).strip()
        for i, t in enumerate(op_type_list)
    ]

    lhs_names = []
    lhs_part = m.group(1)
    if "=" in lhs_part:
        names_str = lhs_part[: lhs_part.rindex("=")]
        lhs_names = [
            strip_suffix(x.strip())
            for x in split_top_level(names_str, ",")
            if x.strip()
        ]

    res_inner, wrapped = unwrap_parens(sig.group(2))
    result_list = split_top_level(res_inner, ",") if res_inner.strip() else []
    result_items = [
        ann.annotate_str(t, lhs_names[i] if i < len(lhs_names) else None).strip()
        for i, t in enumerate(result_list)
    ]
    new_result_text = ", ".join(result_items)
    if wrapped:
        new_result_text = f"({new_result_text})"

    abs_start = close_idx + 1 + sig.start()
    abs_end = close_idx + 1 + sig.end()
    new_sig = f": ({', '.join(operand_items)}) -> {new_result_text}"
    oneline = line[:abs_start] + new_sig + line[abs_end:]
    if len(oneline) <= MAX_LINE:
        return oneline

    base_indent = indent_of(line)
    parts = [f"{line[:abs_start].rstrip()} : ("]
    parts.append(wrap_items(operand_items, base_indent))
    if wrapped:
        parts.append(f"{base_indent}) -> (")
        parts.append(wrap_items(result_items, base_indent))
        parts.append(f"{base_indent})")
    else:
        parts.append(f"{base_indent}) -> {new_result_text}")
    return "\n".join(parts)


def handle_return(line, ann):
    m = RETURN_RE.match(line)
    if not m:
        return None
    lead, vals_text, colon, types_text = m.groups()
    values = [strip_suffix(v.strip()) for v in split_top_level(vals_text, ",")]
    types = split_top_level(types_text, ",")
    new_types = [
        ann.annotate_str(t, values[i] if i < len(values) else None).strip()
        for i, t in enumerate(types)
    ]
    oneline = lead + vals_text + colon + ", ".join(new_types)
    if len(oneline) <= MAX_LINE:
        return oneline

    base_indent = indent_of(line)
    head = (lead + vals_text + colon).rstrip()
    return f"{head}\n{wrap_items(new_types, base_indent)}"


def handle_pretty_result(line, ann):
    if "->" in line:
        return None
    stripped = line.rstrip()
    idx = stripped.rfind("tensor<")
    if idx == -1:
        return None
    open_idx = idx + len("tensor")
    close_idx = find_matching_close(stripped, open_idx, "<", ">")
    if close_idx == -1 or close_idx != len(stripped) - 1:
        return None
    lm = PRETTY_RESULT_LHS_RE.match(stripped[:idx])
    if not lm:
        return None
    lhs_part = lm.group(1)
    lhs_names = []
    if "=" in lhs_part:
        names_str = lhs_part[: lhs_part.rindex("=")]
        lhs_names = [
            strip_suffix(x.strip())
            for x in split_top_level(names_str, ",")
            if x.strip()
        ]
    value = lhs_names[0] if lhs_names else None
    inner = stripped[open_idx + 1 : close_idx]
    replacement = ann.annotate(inner, value)
    if not replacement:
        return None
    oneline = stripped[:idx] + replacement
    if len(oneline) <= MAX_LINE:
        return oneline

    base_indent = indent_of(line)
    head = stripped[:idx].rstrip()
    return f"{head}\n{base_indent}{INDENT_UNIT}{replacement}"


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
            handle_func_signature(line, ann, return_values)
            or handle_generic_op(line, ann)
            or handle_return(line, ann)
            or handle_pretty_result(line, ann)
            or line
        )
        out_lines.append(new_line)

    with open(annotated_path, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"Wrote {annotated_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
