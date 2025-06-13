import numpy as np
import onnxmlir

a = np.arange(3 * 4 * 5, dtype=np.float32).reshape((3, 4, 5))
b = a + 4

sess = onnxmlir.InferenceSession(
    "test_add.onnx", options='--compile-args="-O3 --parallel"'
)
r = sess.run_ort(["my_out"], {"x": a, "y": b})
print(r)
