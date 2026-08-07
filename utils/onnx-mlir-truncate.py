#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

####################### onnx-mlir-truncate.py ##################################
#
# Copyright 2026 The IBM Research Authors.
#
################################################################################
#
# This script runs onnx-mlir and cleans up its IR dump: it does not truncate
# any line by character count (that used to mangle func/onnx.Return signatures
# and left unbalanced quotes behind). Instead it targets the actual sources of
# useless bulk: unparseable elided-constant markers, and raw constant-data
# string literals (e.g. llvm.mlir.global byte dumps after lowering), which get
# replaced with a short placeholder.
#
################################################################################

# Usage
#
# All but last arguments are mlir arguments; last one is log file name.

import datetime
import subprocess
import sys

from mlir_log_utils import clean_line as process_line


def main():
    args = sys.argv[1:]
    if not args:
        sys.exit("Usage: onnx-mlir-truncate.py <onnx-mlir arguments...> <log-file>")
    onnx_mlir_args, log_path = args[:-1], args[-1]

    with open(log_path, "w") as log:

        def tee(text):
            print(text)
            log.write(text + "\n")

        tee(f"Command on {datetime.datetime.now()}")
        tee("onnx-mlir " + " ".join(onnx_mlir_args))
        tee("")

        cmd = [
            "onnx-mlir",
            *onnx_mlir_args,
            "--mlir-elide-elementsattrs-if-larger=20",
            "--mlir-elide-resource-strings-if-larger=20",
        ]
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        for line in proc.stdout:
            tee(process_line(line.rstrip("\n")))
        sys.exit(proc.wait())


if __name__ == "__main__":
    main()
