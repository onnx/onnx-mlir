# SPDX-License-Identifier: Apache-2.0

from onnx import helper
import unittest

class TestF8E5M2(unittest.TestCase):

    @staticmethod
    def toF8(f):
        return helper.float32_to_float8e5m2(f)

    def assertF8Equal(self, f, i):
        self.assertEqual(self.toF8(f), i)

    def test_zero(self):
        self.assertF8Equal(0.0, 0)
        self.assertF8Equal(-0.0, 0x80)

    def test_inf(self):
        self.assertF8Equal(float('inf'), 0x7B)
        self.assertF8Equal(-float('inf'), 0xFB)

    def test_nan(self):
        self.assertF8Equal(float('nan'), 0x7F)

class TestF8E5M2FNUZ(unittest.TestCase):

    @staticmethod
    def toF8(f):
        return helper.float32_to_float8e5m2(f, fn=True, uz=True)

    def assertF8Equal(self, f, i):
        self.assertEqual(self.toF8(f), i)

    def test_zero(self):
        self.assertF8Equal(0.0, 0)
        self.assertF8Equal(-0.0, 0)

    def test_inf(self):
        self.assertF8Equal(float('inf'), 0x7F)
        self.assertF8Equal(-float('inf'), 0xFF)

    def test_nan(self):
        self.assertF8Equal(float('nan'), 0x80)

if __name__ == '__main__':
    unittest.main()
