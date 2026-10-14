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
        compiler_path="",
        compiler_image="",
        auto_pull=True,
        engine="auto",
        cache=None,
        verbose=False,
    ):

        self.cache = cache
        if compiler_image:
            super().__init__(compiler_image, compiler_path, engine, auto_pull, verbose)
        else:
            if not compiler_path:
                # Import the package for standalone compile
                try:
                    import OMPyCompile

                    compiler_path = OMPyCompile.get_compiler_path()
                except ImportError as e:
                    # No standalone compiler, use local PATH
                    pass
            super().__init__(compiler_path, verbose)

    def compile(
        self,
        model_path,
        flags,
        output_path="",
        compiler_path="",
        log_file_name="",
        reuse_compiled_model=False,
    ):
        if self.cache:
            # Check cache policy to decide whether a cached .so can be used
            exit(-1)
        return super().compile(
            model_path,
            flags,
            output_path,
            compiler_path,
            log_file_name,
            reuse_compiled_model,
        )
