# IBM Confidential
# © Copyright IBM Corp. 2025

import numpy as np
import sys
import os
import importlib
#import pkgutil
#loader = pkgutil.get_loader(__package__)
import importlib.util
# Replace: loader = pkgutil.get_loader('my_module')
spec = importlib.util.find_spec(__package__)
loader = spec.loader
PyRuntimeC_module = os.path.join(
    os.path.dirname(loader.get_filename(__package__)), "libs"
)
sys.path.append(PyRuntimeC_module)

try:
    from PyRuntimeC import OMExecutionSession as OMExecutionSession_
except ImportError:
    raise ImportError(
        "Failure to load the prebuilt PyRuntimeC*.so for your system."
        "The reason could be that either your system or your python version is not supported"
        "Refer to README.md to for solution"
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
