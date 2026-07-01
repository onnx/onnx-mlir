# SPDX-License-Identifier: Apache-2.0

##################### test_relu.py #############################################
#
# Copyright 2026 The IBM Research Authors.
#
################################################################################

import unittest
import logging

import numpy as np
import torch
import torch.nn as nn
import torch_onnxmlir
from utils import TorchOMTestCase

logger = logging.basicConfig(level=logging.INFO)

const_N = 10
const_M = 10


class MyModule(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.nn.functional.relu(x)


model = MyModule()
model.eval()

compiled_model = torch.compile(
    model,
    backend="onnxmlir",
    options={
        "compile_options": "-O3",
        "compiler_path": "/workdir/onnx-mlir/build/Debug/bin/onnx-mlir",
    },
)


class TestRelu(TorchOMTestCase):

    def test_relu(self):
        torch_onnxmlir.config.cache_dir = self.TMP_DIR
        x = torch.randn(const_N, const_M)
        with self.assertLogs(logger) as cm:
            with torch.no_grad():
                y = model(x)
                y_compiled = compiled_model(x)
            assert np.array_equal(y, y_compiled)
        self.assertCompile("\n".join(cm.output))
