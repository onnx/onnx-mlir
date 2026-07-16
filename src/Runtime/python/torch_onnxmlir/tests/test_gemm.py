# SPDX-License-Identifier: Apache-2.0

##################### test_gemm.py #############################################
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
from utils import TorchOMTestCase, COMPILER_IMAGE_NAME, COMPILER_PATH

logger = logging.basicConfig(level=logging.INFO)

const_N = 10
const_M = 10


class MyModule(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(const_N, const_M)

    def forward(self, x):
        return self.lin(x)


model = MyModule()
model.eval()

compiled_model = torch.compile(
    model,
    backend="onnxmlir",
    options={
        "compiler_image_name": COMPILER_IMAGE_NAME,
        "compiler_path": COMPILER_PATH,
        "compile_options": "-O3",
    },
)


class TestGemm(TorchOMTestCase):

    def test_gemm(self):
        x = torch.randn(const_N, const_M)
        with self.assertLogs(logger) as cm:
            with torch.no_grad():
                y = model(x)
                y_compiled = compiled_model(x)
            np.testing.assert_allclose(
                y.contiguous().numpy(),
                y_compiled.contiguous().numpy(),
                rtol=1e-5,
                atol=1e-5,
            )
        self.assertCompile(cm.output)
