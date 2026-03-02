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
    from PyOMCompileC import (
        OMCompileSession,
    )
except ImportError:
    raise ImportError(
        "Looks like you did not build the PyOMCompileC target, build it by "
        "running `make PyOMCompileC`. You may need to set PYTHONPATH to the "
        "location of PyOMCompileC (possibly in Debug/lib or Release/lib)."
    )
