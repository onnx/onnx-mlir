# SPDX-License-Identifier: Apache-2.0

##################### test_sessioncache.py #####################################
#
# Copyright 2026 The IBM Research Authors.
#
################################################################################

import unittest
import tempfile
import shutil
from pathlib import Path
import torch_onnxmlir

COMPILER_IMAGE_NAME = None
COMPILER_PATH = "/workdir/onnx-mlir/build/Debug/bin/onnx-mlir"


class TorchOMTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.TMP_DIR = Path(tempfile.mkdtemp())
        torch_onnxmlir.config.cache_dir = cls.TMP_DIR

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.TMP_DIR, ignore_errors=True)

    def assertCompile(self, all_logs):
        joined_logs = "\n".join(all_logs)
        self.assertIn("Export the pytorch model to ONNX", joined_logs)
        self.assertIn("Compile the onnx model", joined_logs)
        self.assertNotIn("Switch to the eager mode", joined_logs)

    def assertInCache(self, all_logs):
        joined_logs = "\n".join(all_logs)
        self.assertIn("Found the model in the cache. No recompilation.", joined_logs)
        self.assertNotIn("Export the pytorch model to ONNX", joined_logs)
        self.assertNotIn("Compile the onnx model", joined_logs)
        self.assertNotIn("Switch to the eager mode", joined_logs)

    def assertNumCompile(self, all_logs, num_compile):
        count = 0
        for line in all_logs:
            if "Compile the onnx model" in line:
                count += 1
        assert count == num_compile

    def assertNoEagerMode(self, all_logs):
        joined_logs = "\n".join(all_logs)
        self.assertNotIn("Switch to the eager mode", joined_logs)
