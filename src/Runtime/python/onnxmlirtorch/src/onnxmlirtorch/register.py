# SPDX-License-Identifier: Apache-2.0

##################### register.py *******#######################################
#
# Copyright 2025 The IBM Research Authors.
#
################################################################################
#
# This file registers onnxmlir_backend with pytorch.
#
################################################################################

import torch
from torch._dynamo import register_backend
from torch._dynamo.backends.registry import lookup_backend
from torch._dynamo.exc import InvalidBackend

# Freeze the model so that parameters (weights and biases) in
# the forward function's arguments become constants in GraphModule.
# Alternative way is setting TORCHDYNAMO_PREPARE_FREEZING=1
torch._dynamo.config.prepare_freezing = 1

# Exporting with dynamic shapes.
torch._dynamo.config.assume_static_by_default = False

from .backend import (
    onnxmlir_backend,
)

_BACKENDS = {
    "onnxmlir": onnxmlir_backend,
}

for name in _BACKENDS:
    compiler_fn = None
    try:
        compiler_fn = lookup_backend(name)
    except InvalidBackend:
        pass

    # This is only necessary if the package was not installed. Lookup
    # failed so we will register manually.
    if compiler_fn is None:
        register_backend(compiler_fn=_BACKENDS[name], name=name)
