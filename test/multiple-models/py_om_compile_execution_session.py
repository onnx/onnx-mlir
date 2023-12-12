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
    from PyCompileRuntime import OMCompileExecutionSession
except ImportError:
    raise ImportError(
        "Looks like you did not build the PyCompileRuntime target, build it by running `make PyCompileRuntime`."
        "You may need to set ONNX_MLIR_HOME to `onnx-mlir/build/Debug` since `make PyCompileRuntime` outputs to `build/Debug` by default"
    )


class PyOMCompileExecutionSession(OMCompileExecutionSession):
    def run(self, inputs):
        # Prepare arguments to call sess.run
        pyrun_inputs = []
        pyrun_shapes = []
        pyrun_strides = []
        for inp in inputs:
            pyrun_inputs.append(inp.flatten())
            pyrun_shapes.append(np.array(inp.shape, dtype=np.int64))
            pyrun_strides.append(np.array(inp.strides, dtype=np.int64))
        return super(PyOMCompileExecutionSession, self).run(
            pyrun_inputs, pyrun_shapes, pyrun_strides
        )
