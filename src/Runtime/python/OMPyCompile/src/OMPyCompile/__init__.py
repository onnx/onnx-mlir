# SPDX-License-Identifier: Apache-2.0

import os
import sys
import importlib.util

spec = importlib.util.find_spec(__package__)
loader = spec.loader
compiler_bin = os.path.join(
    os.path.dirname(loader.get_filename(__package__)), "bin"
)
compiler_path = compiler_bin + "/onnx-mlir"
compiler_lib = os.path.join(
    os.path.dirname(loader.get_filename(__package__)), "lib"
)
sys.path.append(compiler_lib)

from .PyOMCompile import OMCompile
