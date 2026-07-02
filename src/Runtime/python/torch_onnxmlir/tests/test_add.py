# SPDX-License-Identifier: Apache-2.0

##################### test_add.py ##############################################
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
        return x + y


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


class TestAdd(TorchOMTestCase):
    def test_add(self):
        torch_onnxmlir.config.cache_dir = self.TMP_DIR
        with self.assertLogs(logger) as cm:
            x = torch.randn(2, 3)
            y = torch.randn(2, 3)
            with torch.no_grad():
                z = model(x, y)
            assert np.array_equal(z, x + y)
        self.assertCompile(cm.output)
