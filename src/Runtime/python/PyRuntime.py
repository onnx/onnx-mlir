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

if __package__ == "onnxmlir" or __package__ == "torch_onnxmlir":
    loader = pkgutil.get_loader(__package__)
    PyRuntimeC_module = os.path.join(
        os.path.dirname(loader.get_filename(__package__)), "libs"
    )
    sys.path.append(PyRuntimeC_module)

    try:
        from PyRuntimeC import OMExecutionSession as OMExecutionSession_
    except ImportError:
        raise ImportError(
            "Failure to load the prebuild PyRuntimeC*.so for your system."
            "The reason could be that either your system or your python version is not supported"
            "Refer to docs/BuildPyRuntimeLight.md to build the driver by yourself"
        )
else:
    try:
        from PyRuntimeC import OMExecutionSession as OMExecutionSession_
    except ImportError:
        raise ImportError(
            "Looks like you did not build the PyRuntimeC target, build it by running `make PyRuntimeC`."
            "You may need to set ONNX_MLIR_HOME to `onnx-mlir/build/Debug` since `make PyRuntimeC` outputs to `build/Debug` by default"
        )


class OMExecutionSession(OMExecutionSession_):
    # def __init__(self, model_path_or_compile_session):
    #     """
    #     Initialize OMExecutionSession.
    #     
    #     Args:
    #         model_path_or_compile_session: Either a string path to the compiled
    #             shared library, or an OMCompileSession instance. If an
    #             OMCompileSession is provided, the shared library path will be
    #             obtained from compile_session.get_output_file_name().
    #     
    #     Example:
    #         # Initialize with shared library path (string)
    #         >>> session = OMExecutionSession("model.so")
    #         
    #         # Initialize with compile session
    #         >>> from PyOMCompileC import OMCompileSession
    #         >>> compiler = OMCompileSession("model.onnx", "-O3")
    #         >>> session = OMExecutionSession(compiler)
    #     """
    #     # Check if it's a string path first
    #     if isinstance(model_path_or_compile_session, str):
    #         # It's a string path
    #         shared_lib_path = model_path_or_compile_session
    #     elif hasattr(model_path_or_compile_session, 'get_output_file_name'):
    #         # It's a compile session, get the output file name
    #         shared_lib_path = model_path_or_compile_session.get_output_file_name()
    #     else:
    #         raise TypeError(
    #             "Argument must be either a string path to the shared library "
    #             "or an OMCompileSession instance with get_output_file_name() method."
    #         )
    #     
    #     # Initialize the parent class with the shared library path
    #     super(OMExecutionSession, self).__init__(shared_lib_path)
    
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
