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


class TorchOMTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.TMP_DIR = Path(tempfile.mkdtemp())

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.TMP_DIR, ignore_errors=True)

    def assertCompile(self, all_logs):
        self.assertIn("Export the pytorch model to ONNX", all_logs)
        self.assertIn("Compile the onnx model", all_logs)
        self.assertNotIn("Switch to the eager mode", all_logs)

    def assertInCache(self, all_logs):
        self.assertIn("Found the model in the cache. No recompilation.", all_logs)
        self.assertNotIn("Export the pytorch model to ONNX", all_logs)
        self.assertNotIn("Compile the onnx model", all_logs)
        self.assertNotIn("Switch to the eager mode", all_logs)
