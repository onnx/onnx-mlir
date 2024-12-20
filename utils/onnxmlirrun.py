#!/usr/bin/python
# Copyright 2019-2023 The IBM Research Authors.

import os
import sys
import onnx
import time
import signal
import subprocess
import numpy as np
import tempfile

from onnx import numpy_helper
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
from collections import OrderedDict

# This file provide utility to compile and run onnx model with onnx-mlir,
# in an existing python env, such as Pytorch, tensorflow or SciKit learn.
# The interface is delibrately designed similar to onnxruntime to reduce user's
# burden of learning and code change.
# Lots of code is inherited from utils/RunONNXModel.py, which is a
# "standalone" python script to use onnx-mlir compiler.
# In future, this file will evolved to be part of onnx-mlir python package.
# An example:

"""
import onnxmlirrun
import numpy as np

a = np.random.rand(3, 4, 5).astype(np.float32)
b = np.random.rand(3, 4, 5).astype(np.float32)
session = onnxmlirrun.InferenceSession("test_add.onnx")
outputs = session.run(None, {"a": a, "b":b})
print(outputs)
"""


class InferenceSession:
    def __init__(self, model_path, target="cpu", **kwarg):
        self.target = target
        if "options" in kwarg:
            self.options = kwarg["options"]
        else:
            self.options = ""

        # Initialize parameters

        self.VERBOSE = os.environ.get("VERBOSE", False)
        self.input_model_path = model_path

        # name for the compiled library in temporary directory
        self.temp_lib_name = "model"

        # locate onnx-mlir compiler and its library
        if "ONNX_MLIR_HOME" in kwarg:
            self.ONNX_MLIR_HOME = kwarg["ONNX_MLIR_HOME"]
        elif not os.environ.get("ONNX_MLIR_HOME", None):
            raise RuntimeError(
                "The path to the HOME directory of onnx-mlir should be set with either"
                "keyword parameter ONNX_MLIR_HOME in the session initialization,"
                "or with environment variable ONNX_MLIR_HOME."
                "The HOME directory for onnx-mlir refers to the parent folder containing the"
                "bin, lib, etc sub-folders in which ONNX-MLIR executables and libraries can"
                "be found, typically `onnx-mlir/build/Debug`"
            )
        else:
            self.ONNX_MLIR_HOME = os.environ["ONNX_MLIR_HOME"]

        self.ONNX_MLIR_EXENAME = "onnx-mlir"
        if sys.platform == "win32":
            self.ONNX_MLIR_EXENAME = "onnx-mlir.exe"

        # Compiler package related parameters.
        # Should be changed when package is installed

        self.ONNX_MLIR = os.path.join(
            self.ONNX_MLIR_HOME, "bin", self.ONNX_MLIR_EXENAME
        )
        self.RUNTIME_DIR = os.path.join(self.ONNX_MLIR_HOME, "lib")
        sys.path.append(self.RUNTIME_DIR)
        try:
            from PyRuntime import OMExecutionSession
        except ImportError:
            raise ImportError(
                "Looks like you did not build the PyRuntime target, build it by running `make PyRuntime`.You may need to set ONNX_MLIR_HOME to `onnx-mlir/build/Debug` since `make PyRuntime` outputs to `build/Debug` by default"
            )
        # Initialize status
        self.compiled = False
        self.loaded = False

    def compile(self):
        # Prepare compiler arguments.

        self.temp_dir = tempfile.TemporaryDirectory()
        command_str = [self.ONNX_MLIR]

        # for onnxruntime, the provider flag will determine the flags
        # need more work on flags here

        command_str += [self.input_model_path]
        output_path = os.path.join(self.temp_dir.name, self.temp_lib_name)
        command_str += ["-o", output_path]
        if self.target == "zAIU":
            command_str += ["--maccel=NNPA", "-O3", "--march=z16"]
        command_str += self.options.split()

        # Compile the model.

        print("Compiling the model ...")
        start = time.perf_counter()
        (ok, msg) = self.execute_commands(command_str)
        end = time.perf_counter()
        print("compile took ", end - start, " seconds.\n")
        if not ok:
            print("Compiler Error:", msg)
            exit(1)
        self.compiled = True

    def loadSession(self):
        try:
            from PyRuntime import OMExecutionSession
        except ImportError:
            raise ImportError(
                "Looks like you did not build the PyRuntime target, build it by running `make PyRuntime`.You may need to set ONNX_MLIR_HOME to `onnx-mlir/build/Debug` since `make PyRuntime` outputs to `build/Debug` by default"
            )

        # Use the generated shared library to create an execution session.

        print("Loading the compiled model ...")
        start = time.perf_counter()
        shared_lib_path = os.path.join(self.temp_dir.name, self.temp_lib_name + ".so")
        self.sess = OMExecutionSession(shared_lib_path)
        end = time.perf_counter()
        print("load took ", end - start, " seconds.\n")
        self.loaded = True

    def run(self, unknown, runInputs):
        # The first input is from the signature of onnxruntime

        # Check whether the model is compiled

        if not self.compiled:
            self.compile()

        # Check whether  the sess is loaded

        if not self.loaded:
            self.loadSession()

        # Prepare the input

        if isinstance(runInputs, dict):
            # onnxruntime interface

            inputs = list(runInputs.values())
        elif isinstance(runInputs, list):
            inputs = runInputs
        elif type(runInputs).__module__ == np.__name__:
            inputs = [runInputs]
        else:
            msg = "Inputs have to be a dictionary or list."
            print(msg)
            exit(1)

        # Should we check the elements in inputs are np.array?

        print("Running inference ...")
        start = time.perf_counter()
        outs = self.sess.run(inputs)
        end = time.perf_counter()
        print("inference took ", end - start, " seconds.\n")

        return outs

    def execute_commands(self, cmds):
        if self.VERBOSE:
            print(cmds)
        out = subprocess.Popen(cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (stdout, stderr) = out.communicate()
        msg = stderr.decode("utf-8") + stdout.decode("utf-8")
        if out.returncode == -signal.SIGSEGV:
            return (False, "Segfault")
        if out.returncode != 0:
            return (False, msg)
        return (True, msg)
