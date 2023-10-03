# SPDX-License-Identifier: Apache-2.0

# ===-------------------------- utils.py - Utility ------------------------===//
#
# Copyright 2019-2020 The IBM Research Authors.
#
# =============================================================================
#
# ===----------------------------------------------------------------------===//

import logging

logger = logging.getLogger("doc-check")

from doc_parser import *
from utils import *


def parse(line, directive_configs):
    directive_configs.append({"ref": line})
    return success()


def handle(config, ctx):
    logger.debug("Handling a same-as-file directive with config {}".format(config))
    ref_file_path = config["ref"]
    doc_file = ctx.doc_file
    parse_code_section_delimiter(ctx)
    with WrappedFile(
        open(os.path.join(ctx.root_dir, ref_file_path), encoding="utf-8")
    ) as ref_file:
        doc_file.skip_lines(config.get("skip-doc", 0))
        ref_file.skip_lines(config.get("skip-ref", 0))

        while not ref_file.eof():
            ref_line = ref_file.readline().rstrip("\r\n")
            doc_line = doc_file.readline().rstrip("\r\n")
            loc = (doc_file.f.name, doc_file.line, ref_file_path, ref_file.line)
            loc_info = "\ndoc file {}, line no. {}. ref file {}, line no. {}.".format(
                *loc
            )
            if doc_file.eof():
                raise ValueError(
                    "Check failed because doc file is "
                    "shorter than reference file." + loc_info
                )

            if ref_line != doc_line:
                doc_line_info = "\nDoc line      : {}".format(doc_line)
                ref_line_info = "\nReference line: {}".format(ref_line)
                raise ValueError(
                    "Check failed because doc file content is not "
                    "the same as that of reference file."
                    + doc_line_info
                    + ref_line_info
                )

    parse_code_section_delimiter(ctx)


ext_to_patterns = {".md": "\\[same-as-file\\]: <> \\(([^)]*)\\)"}
