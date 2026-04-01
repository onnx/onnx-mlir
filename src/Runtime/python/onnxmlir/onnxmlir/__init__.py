import os
import sys
import importlib.util

spec = importlib.util.find_spec(__package__)
loader = spec.loader
PyRuntimeC_module = os.path.join(
    os.path.dirname(loader.get_filename(__package__)), "libs"
)
sys.path.append(PyRuntimeC_module)

from .onnxmlirdocker import InferenceSession
