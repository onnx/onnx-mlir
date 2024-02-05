# SPDX-License-Identifier: Apache-2.0

# ===------- file_same_as_stdout.py - File Same as stdout Directive -------===//
#
# Copyright 2019-2020 The IBM Research Authors.
#
# =============================================================================
#
# Verifies that a file is the same as stdout of some command execution.
#
# ===----------------------------------------------------------------------===//

import logging
import subprocess
import difflib
import sys

logger = logging.getLogger("doc-check")

from doc_parser import *
from utils import *


def handle(config, ctx):
    logger.debug(
        "Handling a file-same-as-stdout directive with config {}".format(config)
    )

    # Read in file content.
    file = config["file"]
    with open(os.path.join(ctx.root_dir, file), encoding="utf-8") as f:
        file_content = f.read()

    # Execute command and retrieve output.
    cmd = config["cmd"]
    cmd_stdout = subprocess.run(
        cmd, stdout=subprocess.PIPE, cwd=ctx.root_dir
    ).stdout.decode("utf-8")

    # Compute diff.
    diff = difflib.unified_diff(
        file_content.splitlines(keepends=True),
        cmd_stdout.splitlines(keepends=True),
        fromfile=file,
        tofile="$({})".format(" ".join(cmd)),
    )
    diff = list(diff)

    # If diff is non-trivial, raise error and display diff.
    if len(diff):
        print("The following diff is detected:")
        sys.stdout.writelines(diff)
        raise ValueError("Check file-same-as-stdout failed")


ext_to_patterns = {".dc": "file-same-as-stdout\\((.*)\\)"}
