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
        check.main("./same-as-file/simple/", [])

    def test_different(self):
        with self.assertRaises(ValueError) as context:
            check.main("./same-as-file/error-doc-different-from-ref/", [])
        self.assertTrue(
            "Check failed because doc file content is not the same as that of reference file."
            in str(context.exception)
        )

    def test_doc_shorter_than_ref(self):
        # check.main('./same-as-file/error-doc-shorter-than-ref/', [])
        with self.assertRaises(ValueError) as context:
            check.main("./same-as-file/error-doc-shorter-than-ref/", [])
        self.assertTrue(
            "Check failed because doc file is shorter than reference file."
            in str(context.exception)
        )

    def test_skip_doc_ref(self):
        check.main("./same-as-file/skip-doc-ref/", [])


if __name__ == "__main__":
    unittest.main()
