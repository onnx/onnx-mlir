# SPDX-License-Identifier: Apache-2.0

##################### test_sessioncache.py #####################################
#
# Copyright 2026 The IBM Research Authors.
#
################################################################################

import unittest

class TorchOMTestCase(unittest.TestCase):
    def assertCompile(self, all_logs):
        self.assertIn("Export the pytorch model to ONNX", all_logs)
        self.assertIn("Compile the onnx model", all_logs)
        self.assertNotIn("Switch to the eager mode", all_logs)

    def assertInCache(self, all_logs):
        self.assertIn("Found the model in the cache. No recompilation.", all_logs)
        self.assertNotIn("Export the pytorch model to ONNX", all_logs)
        self.assertNotIn("Compile the onnx model", all_logs)
        self.assertNotIn("Switch to the eager mode", all_logs)

