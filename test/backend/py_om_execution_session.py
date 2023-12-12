#!/usr/bin/env python3

##################### common.py ################################################
#
# Copyright 2021-2022 The IBM Research Authors.
#
################################################################################
# commom class `PyOMExecutionSession` called by python scripts
################################################################################
import numpy as np

try:
    from PyRuntime import OMExecutionSession
except ImportError:
    raise ImportError(
        "Looks like you did not build the PyRuntime target, build it by running `make PyRuntime`."
        "You may need to set ONNX_MLIR_HOME to `onnx-mlir/build/Debug` since `make PyRuntime` outputs to `build/Debug` by default"
    )


class PyOMExecutionSession(OMExecutionSession):
    def run(self, inputs):
        # Prepare arguments to call sess.run
        pyrun_inputs = []
        pyrun_shapes = []
        pyrun_strides = []
        for inp in inputs:
            pyrun_inputs.append(inp.flatten())
            pyrun_shapes.append(np.array(inp.shape, dtype=np.int64))
            pyrun_strides.append(np.array(inp.strides, dtype=np.int64))
        return super(PyOMExecutionSession, self).run(
            pyrun_inputs, pyrun_shapes, pyrun_strides
        )
