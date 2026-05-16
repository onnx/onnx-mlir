# SPDX-License-Identifier: Apache-2.0

import os
import sys
import importlib.util

spec = importlib.util.find_spec(__package__)
loader = spec.loader
PyRuntimeC_module = os.path.join(
    os.path.dirname(loader.get_filename(__package__)), "libs"
)
sys.path.append(PyRuntimeC_module)

from .PyRuntime import OMExecutionSession as InferenceSession
from .utils import (
    parse_args,
    run_model_with_input_output_files,
    run_model_with_input_output_arrays,
    compare_result,
)

from .CompileWithContainer import CompileWithContainer
from .CompileDriver import (
    CompileWithStandalone,
    CompileWithLocal,
    compile,
)

__all__ = [
    "InferenceSession",
    "parse_args",
    "run_model_with_input_output_files",
    "run_model_with_input_output_arrays",
    "compare_result",
    "CompileWithContainer",
]
