#!/usr/bin/env python3

############# PyOMCompile.py #####################################################
#
# Copyright 2026 The IBM Research Authors.
#
################################################################################
# Common class `PyOMCompile` called by python scripts
################################################################################
import numpy as np

try:
    from PyOMCompileC import (
        OMCompileSession,
    )
except ImportError:
    raise ImportError(
        "Looks like you did not build the PyOMCompileC target, build it by "
        "running `make PyOMCompileC`. You may need to set ONNX_MLIR_HOME "
        "to `onnx-mlir/build/Debug` since `make PyOMCompileC` outputs to "
        "`build/Debug` by default."
    )
