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
    from PyCompileAndRuntimeC import (
        OMCompileExecutionSession as OMCompileExecutionSession_,
    )
except ImportError:
    raise ImportError(
        "Looks like you did not build the PyCompileAndRuntimeC target, build it by running `make PyCompileAndRuntimeC`."
        "You may need to set ONNX_MLIR_HOME to `onnx-mlir/build/Debug` since `make PyCompileAndRuntimeC` outputs to `build/Debug` by default"
    )


class OMCompileExecutionSession(OMCompileExecutionSession_):
    def run(self, inputs):
        # Prepare arguments to call sess.run
        pyrun_inputs = []
        pyrun_shapes = []
        pyrun_strides = []
        for inp in inputs:
            pyrun_inputs.append(inp.ravel())
            pyrun_shapes.append(np.array(inp.shape, dtype=np.int64))
            pyrun_strides.append(np.array(inp.strides, dtype=np.int64))
        return super(OMCompileExecutionSession, self).run(
            pyrun_inputs, pyrun_shapes, pyrun_strides
        )
