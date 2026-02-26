#!/usr/bin/env python3

############# PyCompileAndRuntime.py #######################################
#
# Copyright 2021-2024 The IBM Research Authors.
#
################################################################################
# Common class `PyOMRuntime` called by python scripts
################################################################################
import numpy as np

try:
    from PyCompileC import (
        OMCompileSession,
    )
except ImportError:
    raise ImportError(
        "Looks like you did not build the PyCompileC target, build it by running `make PyCompileC`. "
        "You may need to set ONNX_MLIR_HOME to `onnx-mlir/build/Debug` since `make PyCompileC` "
        "outputs to `build/Debug` by default."
    )
