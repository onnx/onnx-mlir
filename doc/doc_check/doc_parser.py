# ===------------ doc_parser.py - Documentation Parsing Utility ------------===//
#
# Copyright 2019-2020 The IBM Research Authors.
#
# =============================================================================
#
# ===----------------------------------------------------------------------===//

from typing import List
from utils import *

from directive import Directive, generic_config_parser


def parse_code_section_delimiter(ctx):
    assert ctx.doc_file_ext() == ".md"
    if not ctx.doc_file.next_non_empty_line().strip().startswith("```"):
        raise ValueError("Did not parse a code section delimiter")


def try_parse_and_handle_directive(ctx):
    from directive_impl import same_as_file

    # Register all directives.
    all_directives: List[Directive] = [
        Directive(same_as_file.ext_to_patterns, [generic_config_parser, same_as_file.parse], same_as_file.handle)
    ]

    for directive in all_directives:
        directive_config = []
        if succeeded(directive.try_parse_directive(ctx, directive_config)):
            directive.handle(directive_config.pop(), ctx)
            return success(directive_config)

    return failure()
