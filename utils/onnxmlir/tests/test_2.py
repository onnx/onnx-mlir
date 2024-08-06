import numpy as np
import onnxmlir

a0 = np.zeros((3, 4, 5), dtype=np.float32)
a = a0 + 2
b = a0 + 4

sess = onnxmlir.InferenceSession("test_add.onnx")
r = sess.run(["my_out"], {"input_0": a, "input_1": b})
print(r)
