#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

####################### mlir_log_utils.py #######################################
#
# Copyright 2026 The IBM Research Authors.
#
################################################################################
#
# Shared helpers for cleaning up onnx-mlir IR dumps, used by both
# onnx-mlir-truncate.py and IsolatePass.py: normalizing unparseable
# elided-constant markers, and eliding over-long quoted string literals (e.g.
# llvm.mlir.global byte dumps after lowering).
#
################################################################################

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
