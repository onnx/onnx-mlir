# SPDX-License-Identifier: Apache-2.0

# ===------- test-same-as-file.py - Test for same-as-file directive -------===//
#
# Copyright 2019-2020 The IBM Research Authors.
#
# =============================================================================
#
# ===----------------------------------------------------------------------===//

import unittest
import os
import sys

# Make common utilities visible by adding them to system paths.
test_dir = os.path.dirname(os.path.realpath(__file__))
doc_check_base_dir = os.path.abspath(os.path.join(test_dir, os.pardir))
print(doc_check_base_dir)
sys.path.append(doc_check_base_dir)

import check


class TestStringMethods(unittest.TestCase):
    def test_basic(self):
        check.main("./file-same-as-stdout/success/", [])

    def test_failure(self):
        with self.assertRaises(ValueError) as context:
            check.main("./file-same-as-stdout/failure/", [])
        self.assertTrue("Check file-same-as-stdout failed" in str(context.exception))


if __name__ == "__main__":
    unittest.main()
