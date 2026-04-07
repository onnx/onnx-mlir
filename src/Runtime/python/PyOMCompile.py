#!/usr/bin/env python3

############# PyOMCompile.py #####################################################
#
# Copyright 2026 The IBM Research Authors.
#
################################################################################
# Common class `PyOMCompile` called by python scripts
#
# Use this indirection to call PyOMCompileC in case we need to add some
# specific python processing. None is needed at this time.
################################################################################
import numpy as np

try:
    from PyOMCompileC import OMCompile as OMCompile_

except ImportError:
    raise ImportError(
        "Looks like you did not build the PyOMCompileC target, build it by\n"
        "running `make PyOMCompileC`. You may need to set PYTHONPATH to\n"
        "include `onnx-mlir/build/Debug/lib` since `make PyOMCompileC` outputs\n"
        "to `build/Debug` by default (or Release if building Release)."
    )


class OMCompile(OMCompile_):
    def __init__(
        self,
        input_model_path,
        flags,
        compiler_path="",
        log_file_name="",
        reuse_compiled_model=False,
    ):
        if __package__ and not compiler_path:
            from . import compiler_path as compiler_path_in_package

            super().__init__(
                input_model_path,
                flags,
                compiler_path_in_package,
                log_file_name,
                reuse_compiled_model,
            )
        else:
            super().__init__(
                input_model_path,
                flags,
                compiler_path,
                log_file_name,
                reuse_compiled_model,
            )
