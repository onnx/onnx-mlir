# SPDX-License-Identifier: Apache-2.0

##################### __init__.py *******#######################################
#
# Copyright 2025-2026 The IBM Research Authors.
#
################################################################################
#
#
################################################################################

from .register import *
from .explain import explain_context
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("torch_onnxmlir")
except PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = ["onnxmlir_backend", "explain_context"]
