# SPDX-License-Identifier: Apache-2.0

# ===-------------------- check.py - Documentation Checker ----------------===//
#
# Copyright 2019-2020 The IBM Research Authors.
#
# =============================================================================
#
# ===----------------------------------------------------------------------===//

import argparse
import os, sys

from itertools import chain
from pathlib import Path

from utils import setup_logger, DocCheckerCtx

logger = setup_logger("doc-check")

# Make common utilities visible by adding them to system paths.
doc_check_base_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(doc_check_base_dir)

from doc_parser import try_parse_and_handle_directive

parser = argparse.ArgumentParser()
parser.add_argument(
    "root_dir", help="directory in which to look for documentation to operate on"
)

parser.add_argument(
    "--exclude_dirs",
    nargs="+",
    help="a set of directories to exclude, with path specified relative to root_dir",
    default=[],
)


def main(root_dir, exclude_dirs):
    for i, exclude_dir in enumerate(exclude_dirs):
        exclude_dirs[i] = os.path.normpath(os.path.join(root_dir, exclude_dir))

    ctx = DocCheckerCtx(root_dir)
    for doc_file in chain(Path(root_dir).rglob("*.md"), Path(root_dir).rglob("*.dc")):
        doc_file = os.path.normpath(doc_file)
        # Skip, if doc file is in directories to be excluded.
        if any([str(doc_file).startswith(exclude_dir) for exclude_dir in exclude_dirs]):
            continue

        logger.info("Checking {}...".format(doc_file))
        with ctx.open_doc(doc_file) as markdown_file:
            while not markdown_file.eof():
                try_parse_and_handle_directive(ctx)


if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
