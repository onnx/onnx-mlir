# ===----------------- directive.py - Directive Base Class ----------------===//
#
# Copyright 2019-2020 The IBM Research Authors.
#
# =============================================================================
#
# ===----------------------------------------------------------------------===//

import re
import ast
from typing import List, Dict, Callable, Any, Pattern, Tuple

from doc_parser import failure, success, succeeded
from utils import DocCheckerCtx

DirectiveConfigList = List[Dict[str, Any]]
ConfigParseResult = Tuple[str, Dict[str, Any]]


class Directive(object):
    """"""
    def __init__(self, ext_to_regexes: Dict[str, str],
                 config_parsers: List[Callable[[str, DirectiveConfigList],
                                               ConfigParseResult]],
                 handler: Callable[[Dict[str, Any], DocCheckerCtx], None]):
        """
        :param ext_to_regexes: specify a regex expression to match the directive (for each file extension type).
        :param config_parsers: specify a list of parsers to parse configuration. They will be invoked in order until one indicates parsing is successful.
        :param handler: a function to perform the invariance check specified by the directive.
        """
        self.ext_to_patterns: Dict[str, Pattern] = {}
        for ext, pattern in ext_to_regexes.items():
            self.ext_to_patterns[ext] = re.compile(pattern)

        self.config_parsers: List[Callable[[str, DirectiveConfigList],
                                           ConfigParseResult]] = config_parsers
        self.handler = handler

    def try_parse_directive(
            self, ctx: DocCheckerCtx,
            directive_config: DirectiveConfigList) -> Tuple[str, Any]:
        """
        :param ctx: parser context.
        :param directive_config: a list used to output parsed directive configuration.
        :return: parse result.
        """
        line = ctx.doc_file.next_non_empty_line()
        matches = self.ext_to_patterns[ctx.doc_file_ext()].findall(line)
        if len(matches) > 1:
            raise ValueError("more than one directives in a line")

        match = matches[0] if len(matches) else None
        if match:
            for parser in self.config_parsers:
                if succeeded(parser(match, directive_config)):
                    return success()

            raise ValueError("Failed to parse configuration.")
        else:
            return failure()

    def handle(self, config, ctx):
        self.handler(config, ctx)


def generic_config_parser(
        match: str, directive_config: DirectiveConfigList) -> Tuple[str, Any]:
    """
    Generic configuration parser.
    Will return success if and only if configuration is specified as a python dictionary literal.

    @param match: the content from which to parse the directive configuration.
    @param directive_config: a list to output the parsed directive_config.
    @return: parsing result.
    """
    try:
        directive_config.append(ast.literal_eval(match))
        return success()
    except (SyntaxError, ValueError):
        # If literal_eval failed, return parsing failure.
        return failure()
