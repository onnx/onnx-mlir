# Test: no name for input and output
import numpy as np
import onnxmlir

a = np.arange(3 * 4 * 5, dtype=np.float32).reshape((3, 4, 5))
b = a + 4

# When compiler_image_name is None, local compiler will be used. The compiler_path is needed to
# locate the compiler.
# Alternative implementation:  use env variable ONNX_MLIR_HOME?
# compile_args is the flags passed to onnx-mlir
sess = onnxmlir.InferenceSession(
    "test_add.onnx",
    compile_args="-O3",
    compiler_image_name=None,
    compiler_path="/workdir/onnx-mlir/build/Debug/bin/onnx-mlir",
)
r = sess.run([a, b])
print(r)
