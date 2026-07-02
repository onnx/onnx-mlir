# SPDX-License-Identifier: Apache-2.0

##################### test_sessioncache.py #####################################
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


class AddModel(nn.Module):

    def __init__(self):
        super(AddModel, self).__init__()

    def forward(self, x, y):
        return x + y  # Element-wise addition


model = AddModel()
model.eval()

model = torch.compile(
    model,
    backend="onnxmlir",
    options={
        "compiler_image_name": COMPILER_IMAGE_NAME,
        "compiler_path": COMPILER_PATH,
        "compile_options": "-O3",
    },
)

logger = logging.basicConfig(level=logging.INFO)  # Or INFO, WARNING, etc.


class TestSessionCache(TorchOMTestCase):

    def test_cache(self):
        torch_onnxmlir.config.cache_dir = self.TMP_DIR

        # First inference.
        with self.assertLogs(logger) as cm:
            print("\n1st inference: should compile")
            x = torch.randn(2, 3)
            y = torch.randn(2, 3)
            with torch.no_grad():
                z = model(x, y)
            assert np.array_equal(z, x + y)
        self.assertCompile(cm.output)

        # Second inference.
        with self.assertLogs(logger) as cm:
            print("\n2nd inference: should reuse")
            x = torch.randn(3, 3)
            y = torch.randn(3, 3)
            with torch.no_grad():
                z = model(x, y)
            assert np.array_equal(z, x + y)
        self.assertInCache(cm.output)

        # Third inference.
        with self.assertLogs(logger) as cm:
            print("\n3rd inference: should compile")
            x = torch.randn(5)
            y = torch.randn(5)
            with torch.no_grad():
                z = model(x, y)
            assert np.array_equal(z, x + y)
        self.assertCompile(cm.output)

        # Forth inference.
        with self.assertLogs(logger) as cm:
            print("\n4th inference: should reuse")
            x = torch.randn(2, 5)
            y = torch.randn(2, 5)
            with torch.no_grad():
                z = model(x, y)
            assert np.array_equal(z, x + y)
        self.assertInCache(cm.output)

        # Fifth inference.
        with self.assertLogs(logger) as cm:
            print("\n5th inference: should reuse")
            x = torch.randn(7)
            y = torch.randn(7)
            with torch.no_grad():
                z = model(x, y)
            assert np.array_equal(z, x + y)
        self.assertInCache(cm.output)
