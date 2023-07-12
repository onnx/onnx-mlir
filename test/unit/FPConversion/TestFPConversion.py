# SPDX-License-Identifier: Apache-2.0

from onnx import helper
import unittest

class TestCase(unittest.TestCase):

    def test_zero(self):
        zero = helper.float32_to_float8e5m2(0.0)
        self.assertEqual(zero, 0)
        negZero = helper.float32_to_float8e5m2(-0.0)
        self.assertEqual(negZero, 0x80)

    def test_inf(self):
        inf = helper.float32_to_float8e5m2(float('inf'))
        self.assertEqual(inf, 0x7B)
        negInf = helper.float32_to_float8e5m2(-float('inf'))
        self.assertEqual(negInf, 0xFB)

if __name__ == '__main__':
    unittest.main()
