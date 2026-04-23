import sys

print("=== import onnx diagnostic ===")
print("sys.path =", sys.path)

import onnx

print("onnx module =", onnx)
print("onnx.__file__ =", getattr(onnx, "__file__", "no __file__"))
print("onnx.__spec__ =", getattr(onnx, "__spec__", "no __spec__"))
print("has helper =", hasattr(onnx, "helper"))
print("onnx attrs sample =", sorted(name for name in dir(onnx) if name in {
    "__file__",
    "__path__",
    "__spec__",
    "helper",
    "checker",
    "TensorProto",
    "AttributeProto",
    "GraphProto",
}))

import onnx.helper

print("onnx.helper =", onnx.helper)

