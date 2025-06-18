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
import pkgutil

if __package__ == "onnxmlir" or __package__ == "onnxmlirtorch":
    python_version_str = str(sys.version_info[0]) + "_" + str(sys.version_info[1])
    loader = pkgutil.get_loader("onnxmlir")
    PyRuntimeC_module = os.path.join(
        os.path.dirname(loader.get_filename("onnxmlir")), "libs/" + python_version_str
    )
    if not os.path.isdir(PyRuntimeC_module):
        print(
            f"This package does not have the pre-built lib for python version of {python_version_str}. You can build the package for your env by using onnx-mlir/utils/build-pyruntime-light.sh"
        )
        exit(-1)
    sys.path.append(PyRuntimeC_module)

    try:
        from PyRuntimeC import OMExecutionSession as OMExecutionSession_
    except ImportError:
        raise ImportError("Failure to load the prebuild PyRuntimeC.*.so.")
else:
    try:
        from PyRuntimeC import OMExecutionSession as OMExecutionSession_
    except ImportError:
        raise ImportError(
            "Looks like you did not build the PyRuntimeC target, build it by running `make PyRuntimeC`."
            "You may need to set ONNX_MLIR_HOME to `onnx-mlir/build/Debug` since `make PyRuntimeC` outputs to `build/Debug` by default"
        )


class OMExecutionSession(OMExecutionSession_):
    def run(self, inputs):
        # Prepare arguments to call sess.run
        pyrun_inputs = []
        pyrun_shapes = []
        pyrun_strides = []
        for inp in inputs:
            pyrun_inputs.append(inp.ravel())
            pyrun_shapes.append(np.array(inp.shape, dtype=np.int64))
            pyrun_strides.append(np.array(inp.strides, dtype=np.int64))
        return super(OMExecutionSession, self).run(
            pyrun_inputs, pyrun_shapes, pyrun_strides
        )
