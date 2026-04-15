# SPDX-License-Identifier: Apache-2.0

import os
import sys
import importlib.util

spec = importlib.util.find_spec(__package__)
if spec is None or spec.loader is None:
    raise ImportError(
        "This module must be imported as part of a package, not run directly."
    )
loader = spec.loader
compiler_path = os.path.join(
    os.path.dirname(loader.get_filename(__package__)), "bin/onnx-mlir"
)
compiler_lib = os.path.join(os.path.dirname(loader.get_filename(__package__)), "lib")
sys.path.append(compiler_lib)

from .PyOMCompile import OMCompile
