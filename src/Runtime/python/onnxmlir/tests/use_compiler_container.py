# Test: no name for input and output
import numpy as np
import onnxmlir

a = np.arange(3 * 4 * 5, dtype=np.float32).reshape((3, 4, 5))
b = a + 4

sess = onnxmlir.InferenceSession(
    "test_add.onnx",
    compile_args="-O3 --parallel",
    container_engine="docker",
    compiler_image_name="ghcr.io/onnxmlir/onnx-mlir-dev",
    compiler_path="/workdir/onnx-mlir/build/Debug/bin/onnx-mlir",
)
# In the example above, all the compiler container related options are the default
# value. Therefore, the previous sentence can be simplified to the following one:
"""
 sess = onnxmlir.InferenceSession("test_add.onnx", compile_args="-O3 --parallel")
"""
r = sess.run([a, b])
print(r)
