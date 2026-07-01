# SPDX-License-Identifier: Apache-2.0

##################### test_add.py ##############################################
#
# Copyright 2026 The IBM Research Authors.
#
################################################################################

import tempfile
import shutil
from pathlib import Path
import unittest
import logging

import numpy as np
import torch
import torch.nn as nn
import torch_onnxmlir

from utils import TorchOMTestCase

TMP_DIR = Path(tempfile.mkdtemp())
torch_onnxmlir.config.cache_dir = TMP_DIR


def tearDownModule():
    shutil.rmtree(TMP_DIR)


class AddModel(nn.Module):

    def __init__(self):
        super(AddModel, self).__init__()

    def forward(self, x, y):
        return x + y  # Element-wise addition


model = AddModel()
model = torch.compile(
    model,
    backend="onnxmlir",
    options={
        "compile_options": "-O3",
        "compiler_path": "/workdir/onnx-mlir/build/Debug/bin/onnx-mlir",
    },
)

logger = logging.basicConfig(level=logging.INFO)  # Or INFO, WARNING, etc.


class TestAdd(TorchOMTestCase):
    def test_add(self):
        with self.assertLogs(logger) as cm:
            x = torch.randn(2, 3)
            y = torch.randn(2, 3)
            z = model(x, y)
            assert np.array_equal(z, x + y)
        self.assertCompile("\n".join(cm.output))
