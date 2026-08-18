#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

####################### mlir_log_utils.py #######################################
#
# Copyright 2026 The IBM Research Authors.
#
################################################################################
#
# Shared helpers for cleaning up and annotating onnx-mlir IR dumps, used by
# onnx-mlir-truncate.py, IsolatePass.py, and AnalyzeShape.py:
#  - normalizing unparseable elided-constant markers, and eliding over-long
#    quoted string literals (e.g. llvm.mlir.global byte dumps after lowering)
#  - annotating SSA defs with use-counts and inlining scalar-constant values
#    (the "-c"/"--comments" transform)
#  - reformatting long func signatures / ops / returns across multiple
#    indented lines (the "-w"/"--wrap" transform)
#
################################################################################

import re
from collections import Counter

# Quoted string literals longer than this (in raw/escaped characters) are
# assumed to be constant data dumps (e.g. llvm.mlir.global byte strings) and
# get replaced by a short placeholder. For ops like llvm.mlir.global, the type
# is inferred from the string's length, so shrinking the string in place keeps
# the line self-consistent and parseable.
LONG_STRING_THRESHOLD = 300


def fix_elided_dense(line):
    # onnx-mlir's DisposableElementsAttr sometimes prints elided large constants
    # as "dense<__elided__>", which (unlike "dense_resource<__elided__>") has no
    # parser rule and cannot be read back in. Normalize it to the parseable form.
    return line.replace("dense<__elided__>", "dense_resource<__elided__>")


def elide_long_quoted_strings(line, threshold=LONG_STRING_THRESHOLD):
    out = []
    i, n = 0, len(line)
    while i < n:
        q = line.find('"', i)
        if q == -1:
            out.append(line[i:])
            break
        out.append(line[i:q])
        j = q + 1
        while j < n and line[j] != '"':
            if line[j] == "\\":
                j += 1
            j += 1
        if j >= n:
            # Unterminated string on this line; leave it as-is.
            out.append(line[q:])
            break
        content_len = j - (q + 1)
        if content_len > threshold:
            out.append(f'"<elided, was {content_len} chars>"')
        else:
            out.append(line[q : j + 1])
        i = j + 1
    return "".join(out)


def clean_line(line):
    """Normalize elided-constant markers and shrink over-long quoted strings."""
    return elide_long_quoted_strings(fix_elided_dense(line))


###########################################################
# Annotate each def by its number of uses ("-c"/"--comments" transform, part 1)

# Matches SSA values like %0, %res_12, %xyz.$tmp
SSA_RE = re.compile(r"%[A-Za-z0-9_.$]+")


def split_lhs_rhs_strip_comment(line: str):
    """Return (lhs, rhs) after stripping '//' comments.
    If '=' not present, lhs == '' and rhs == (comment-stripped line).
    """
    # Drop '//' comments
    line = line.split("//", 1)[0]
    if "=" not in line:
        return "", line
    lhs, rhs = line.split("=", 1)
    return lhs, rhs


def split_comment(line: str):
    """Split a line into (code, comment) where comment includes the leading '//' if present."""
    if "//" in line:
        code, comment = line.split("//", 1)
        return code, "//" + comment
    return line, ""


def collect_outputs(mlir_text: str):
    """Collect names defined on the LHS of '=' (i.e., operation results)."""
    outputs = set()
    for line in mlir_text.splitlines():
        lhs, _ = split_lhs_rhs_strip_comment(line)
        if lhs:
            for name in SSA_RE.findall(lhs):
                outputs.add(name)
    return outputs


def count_output_uses(mlir_text: str):
    """Return a dict { '%name': use_count } for every SSA defined as '%x = ...'.

    Counts only RHS occurrences (and lines without '=') so definitions aren't counted.
    Includes outputs with 0 uses.
    """
    outputs = collect_outputs(mlir_text)
    counts = Counter({name: 0 for name in outputs})
    for line in mlir_text.splitlines():
        lhs, rhs = split_lhs_rhs_strip_comment(line)
        # Count only the RHS (or the whole line if there's no '=')
        scan = rhs  # rhs already equals full line when lhs == ''
        for name in SSA_RE.findall(scan):
            if name in outputs:
                counts[name] += 1
    return dict(counts)


def annotate_mlir_with_use_counts(mlir_text: str, use_counts: dict[str, int]) -> str:
    """
    Return a new MLIR text where each SSA result defined on a line is annotated as '%x/N'
    on the LHS of '=', where N is the use count from `use_counts`.

    Non-definition lines are left unchanged. Comments are preserved.
    """
    out_lines = []
    for line in mlir_text.splitlines():
        code, comment = split_comment(line)
        if "=" not in code:
            # Not a definition line—pass through unchanged
            out_lines.append(line)
            continue
        lhs, rhs = code.split("=", 1)

        # Replace each SSA name in the LHS with '%name/N'
        def _add_count(m: re.Match) -> str:
            name = m.group(0)
            return f"{name}/{use_counts.get(name, 0)}"

        annotated_lhs = SSA_RE.sub(_add_count, lhs)
        # Reassemble, preserving spacing and comments
        new_line = f"{annotated_lhs}={rhs}{comment}"
        out_lines.append(new_line)
    # Preserve trailing newline behavior similar to input
    return "\n".join(out_lines) + ("\n" if mlir_text.endswith("\n") else "")


###########################################################
# Annotate each use of a constant by its value ("-c"/"--comments" transform, part 2)

DENSE_PAYLOAD_RE = re.compile(r"dense<([^>]+)>\s*:\s*tensor<([^>]+)>", re.IGNORECASE)


def is_rank0_or_size1_tensor(type_inner: str) -> bool:
    """
    Given the inner part of tensor<...> (i.e., what's inside the angle brackets),
    return True if it is a rank-0 tensor (just a type like 'f32'/'i64') or a
    rank-1 size-1 tensor (e.g., '1xf32', '1xi64').
    """
    inner = type_inner.strip()
    # Rank-0: just a scalar type token, no 'x'
    if "x" not in inner and "[" not in inner and "]" not in inner:
        return True
    # Rank-1 size-1: '1x<type>'
    m = re.match(r"^\s*1x([A-Za-z0-9_<>:?$\-\[\]]+)\s*$", inner)
    return m is not None


def extract_onnx_dense_scalar(rhs: str) -> str | None:
    """
    Return the scalar-like value string if RHS contains:
      onnx.Constant ... dense<...> : tensor<...>
    and the tensor type is rank-0 or rank-1 size-1.
    Otherwise return None.
    """
    m = DENSE_PAYLOAD_RE.search(rhs)
    if not m:
        return None
    payload = m.group(1).strip()  # e.g., '64', '1.0', '[1, -1, 12, 64]'
    tensor_inner = m.group(2).strip()  # e.g., '1xi64', 'f32', '4xi64'
    if not is_rank0_or_size1_tensor(tensor_inner):
        return None
    # Reject vector/array payloads like '[1, -1, 12, 64]'
    if payload.startswith("[") and payload.endswith("]"):
        return None
    return payload


def collect_onnx_dense_scalar_constants(mlir_text: str) -> dict[str, str]:
    """
    Collect { '%name': '<value>' } for onnx.Constant with dense<...> payloads
    when the type is rank-0 or size-1 tensor.
    """
    const_vals: dict[str, str] = {}
    for line in mlir_text.splitlines():
        lhs, rhs = split_lhs_rhs_strip_comment(line)
        if not lhs:
            continue
        # Quick pre filter for 'onnx.Constant' and 'dense<'
        if "onnx.Constant" not in rhs or "dense<" not in rhs:
            continue
        value = extract_onnx_dense_scalar(rhs)
        if value is None:
            continue
        for name in SSA_RE.findall(lhs):
            const_vals[name] = value
    return const_vals


def _annotate_scalar_constant_uses_in_code_segment(
    code: str, const_vals: dict[str, str]
) -> str:
    """Append '/<value>' to each SSA *use* that is a scalar constant."""

    def repl(m: re.Match) -> str:
        name = m.group(0)
        if name in const_vals:
            return f"{name}={const_vals[name]}"
        return name

    return SSA_RE.sub(repl, code)


def annotate_constant_uses_with_values(mlir_text: str) -> str:
    """
    Annotate each *use* of a constant discovered by collect_onnx_dense_scalar_constants
    by appending '/<value>' after the SSA name.

    - LHS (definitions) are not modified.
    - RHS (and lines without '=') get '%c' -> '%c/<value>' for constants.
    - Comments are preserved.
    """
    const_vals = collect_onnx_dense_scalar_constants(mlir_text)

    out_lines = []
    for line in mlir_text.splitlines():
        code, comment = split_comment(line)

        if "=" in code:
            lhs, rhs = code.split("=", 1)
            # Do NOT change LHS (definitions); only annotate RHS uses
            annotated_rhs = _annotate_scalar_constant_uses_in_code_segment(
                rhs, const_vals
            )
            new_line = f"{lhs}={annotated_rhs}{comment}"
            out_lines.append(new_line)
        else:
            # No definition; annotate whole line for constant uses
            annotated = (
                _annotate_scalar_constant_uses_in_code_segment(code, const_vals)
                + comment
            )
            out_lines.append(annotated)

    # Preserve trailing newline behavior similar to input
    return "\n".join(out_lines) + ("\n" if mlir_text.endswith("\n") else "")


def apply_comments(mlir_text: str) -> str:
    """Annotate SSA defs with use-counts and inline scalar-constant values.

    This is the full "-c"/"--comments" transform: use-count for each def,
    plus values inlined at every use of a scalar constant.
    """
    use_counts = count_output_uses(mlir_text)
    mlir_text = annotate_mlir_with_use_counts(mlir_text, use_counts)
    mlir_text = annotate_constant_uses_with_values(mlir_text)
    return mlir_text


###########################################################
# Reformat long lines across multiple indented lines
# ("-w"/"--wrap" transform)

VALUE = r"%[\w.]+"

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

MAX_LINE = 120
INDENT_UNIT = "    "


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


def handle_func_signature(line, ann, return_values, wrap=True):
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
    if not wrap or len(oneline) <= MAX_LINE:
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


def handle_generic_op(line, ann, wrap=True):
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
    if not wrap or len(oneline) <= MAX_LINE:
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


def handle_return(line, ann, wrap=True):
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
    if not wrap or len(oneline) <= MAX_LINE:
        return oneline

    base_indent = indent_of(line)
    head = (lead + vals_text + colon).rstrip()
    return f"{head}\n{wrap_items(new_types, base_indent)}"


def handle_pretty_result(line, ann, wrap=True):
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
    if not wrap or len(oneline) <= MAX_LINE:
        return oneline

    base_indent = indent_of(line)
    head = stripped[:idx].rstrip()
    return f"{head}\n{base_indent}{INDENT_UNIT}{replacement}"


def apply_wrap(mlir_text: str, ann=None, return_values=None) -> str:
    """Reformat long func signatures / ops / returns / result types across
    multiple indented lines when they exceed MAX_LINE.

    This is the full "-w"/"--wrap" transform. Without dimension-group data
    (the default `ann`/`return_values`), this purely wraps long lines with
    no other annotation.
    """
    ann = ann if ann is not None else Annotator({}, {})
    return_values = return_values if return_values is not None else []
    out_lines = []
    for line in mlir_text.splitlines():
        new_line = (
            handle_func_signature(line, ann, return_values)
            or handle_generic_op(line, ann)
            or handle_return(line, ann)
            or handle_pretty_result(line, ann)
            or line
        )
        out_lines.append(new_line)
    return "\n".join(out_lines) + ("\n" if mlir_text.endswith("\n") else "")
