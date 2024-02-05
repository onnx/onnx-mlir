# SPDX-License-Identifier: Apache-2.0

# ===------------ doc_parser.py - Documentation Parsing Utility ------------===//
#
# Copyright 2019-2020 The IBM Research Authors.
#
# =============================================================================
#
# ===----------------------------------------------------------------------===//

import logging

from typing import List
from utils import *

from directive import Directive, generic_config_parser

logger = logging.getLogger("doc-check")


def parse_code_section_delimiter(ctx):
    assert ctx.doc_file_ext() == ".md"
    if not ctx.doc_file.next_non_empty_line().strip().startswith("```"):
        raise ValueError("Did not parse a code section delimiter")


def try_parse_and_handle_directive(ctx):
    from directive_impl import same_as_file
    from directive_impl import file_same_as_stdout

    try:
        line = ctx.doc_file.next_non_empty_line()
    except RuntimeError as e:
        # Do not raise exception when next non-empty line
        # does not exist. Instead, return failure.
        if str(e) != "Enf of file.":
            raise
        return failure()

    # Register all directives.
    all_directives: List[Directive] = [
        Directive(
            same_as_file.ext_to_patterns,
            [generic_config_parser, same_as_file.parse],
            same_as_file.handle,
        ),
        Directive(
            file_same_as_stdout.ext_to_patterns,
            [generic_config_parser],
            file_same_as_stdout.handle,
        ),
    ]

    for directive in all_directives:
        directive_config = []
        if succeeded(
            directive.try_parse_directive(line, ctx.doc_file_ext(), directive_config)
        ):
            directive.handle(directive_config.pop(), ctx)
            return success(directive_config)

    return failure()
