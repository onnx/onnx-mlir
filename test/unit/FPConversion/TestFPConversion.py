# SPDX-License-Identifier: Apache-2.0

from onnx import helper
import unittest

class TestF8E5M2(unittest.TestCase):

    @staticmethod
    def toF8(f, saturate=True):
        return helper.float32_to_float8e5m2(f, saturate=saturate)

    def assertF8Equal(self, f, i, j=None):
        if j is None:
            j = i
        self.assertEqual(self.toF8(f), i)
        self.assertEqual(self.toF8(f, saturate=False), j)

    def test_zero(self):
        self.assertF8Equal(0.0, 0)
        self.assertF8Equal(-0.0, 0x80)

    def test_max(self):
        self.assertF8Equal(57344.0, 0x7B)
        self.assertF8Equal(-57344.0, 0xFB)

    def test_inf(self):
        self.assertF8Equal(float('inf'), 0x7B, 0x7C)
        self.assertF8Equal(-float('inf'), 0xFB, 0XFC)

    def test_nan(self):
        self.assertF8Equal(float('nan'), 0x7F)

class TestF8E5M2FNUZ(unittest.TestCase):

    @staticmethod
    def toF8(f, saturate=True):
        return helper.float32_to_float8e5m2(f, fn=True, uz=True, saturate=saturate)

    def assertF8Equal(self, f, i, j=None):
        if j is None:
            j = i
        self.assertEqual(self.toF8(f), i)
        self.assertEqual(self.toF8(f, saturate=False), j)

    def test_zero(self):
        self.assertF8Equal(0.0, 0)
        self.assertF8Equal(-0.0, 0)

    def test_max(self):
        self.assertF8Equal(57344.0, 0x7F)
        self.assertF8Equal(-57344.0, 0xFF)

    def test_inf(self):
        self.assertF8Equal(float('inf'), 0x7F, 0x80)
        self.assertF8Equal(-float('inf'), 0xFF, 0x80)

    def test_nan(self):
        self.assertF8Equal(float('nan'), 0x80)

if __name__ == '__main__':
    unittest.main()
