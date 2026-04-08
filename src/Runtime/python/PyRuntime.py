#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

############# PyOMRuntime.py #######################################
#
# Copyright 2021-2024 The IBM Research Authors.
#
################################################################################
# commom class `PyOMRuntime` called by python scripts
################################################################################
import numpy as np
import sys
import os
import importlib
import importlib.util

try:
    from PyRuntimeC import OMExecutionSession as OMExecutionSession_
except ImportError:
    raise ImportError(
        "Failed in loading the prebuilt PyRuntimeC*.so for your system."
        "Here is how to fix this issue:"
        "1. Make sure to build PyRuntimeC target to match your architecture and python version."
        "2. Add the path of PyRuntimeC, in build/Debug/lib by default, to your os path"
    )


class OMExecutionSession(OMExecutionSession_):

    def run(self, inputs):
        # Prepare arguments to call session.run
        pyrun_inputs = []
        pyrun_shapes = []
        pyrun_strides = []
        for inp in inputs:
            pyrun_inputs.append(inp.ravel())
            pyrun_shapes.append(np.array(inp.shape, dtype=np.int64))
            pyrun_strides.append(np.array(inp.strides, dtype=np.int64))
        return super(OMExecutionSession, self)._runImplementation(
            pyrun_inputs,
            pyrun_shapes,
            pyrun_strides,
            False,  # No Signal handler
            False,  # No forced copy of output data.
        )

    def runDebug(self, inputs, with_signal_handler=False, force_output_data_copy=False):
        # Prepare arguments to call session.run
        pyrun_inputs = []
        pyrun_shapes = []
        pyrun_strides = []
        for inp in inputs:
            pyrun_inputs.append(inp.ravel())
            pyrun_shapes.append(np.array(inp.shape, dtype=np.int64))
            pyrun_strides.append(np.array(inp.strides, dtype=np.int64))
        return super(OMExecutionSession, self)._runImplementation(
            pyrun_inputs,
            pyrun_shapes,
            pyrun_strides,
            with_signal_handler,
            force_output_data_copy,
        )
