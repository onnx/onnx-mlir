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
        compile_policy="local",
        compiler_image="",
        auto_pull=True,
        engine="auto",
        cache=None,
        verbose=False
    ):
        
        self.cache = cache
        # Check legality of the parameter combination
        if compiler_image and not compiler_path:
            print("When compiler image is used, the compiler_path has to be provided")
            exit(-1)
        if compile_policy == "standalone" and (compiler_image or compiler_path):
            print("Choose exactly one compiler")
            exit(-1)

        if compile_policy == "standalone":
            # Import the package for standalone compile
            try:
                import OMPyCompile
            except ImportError as e:
                print(f"Error: {e.msg}")
                print(f"Module name: {e.name}")
                print("Please install package for standalone compiler")
                exit(-1)

            super().__init__(
                OMPyCompile.get_compiler_path(),
                verbose
            )

        elif compiler_image:
            super().__init__(
                compiler_image,
                compiler_path,
                engine,
                auto_pull,
                verbose
            )
        else:
            super().__init__(
                compiler_path,
                verbose
            )

    def compile(self, model_path, flags, compiler_path="", log_file_name=""):
        if self.cache:
            # Check cache policy to decide whether a cached .so can be used
            exit(-1)
        super().compile(model_path, flags, compiler_path, log_file_name)
        if self.cache:
            # Update cache
            exit(-1)

