# Test: no name for input and output
import numpy as np
import onnxmlir

a = np.arange(3*4*5, dtype=np.float32).reshape((3, 4, 5))
b = a + 4

sess = onnxmlir.InferenceSession("test_add.so")
r = sess.run([a, b])
print(r)
