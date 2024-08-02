# Test: no name for input and output
import numpy as np
import onnxmlir

print(dir(onnxmlir))
a = np.zeros((3,4,5), dtype=np.float32)
b = a + 4

sess = onnxmlir.inferenceSession("test_add.onnx")
r = sess.run(None, [a, b])
print(r)
