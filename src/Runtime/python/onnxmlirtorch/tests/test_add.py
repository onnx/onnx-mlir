import numpy as np
import torch
import torch.nn as nn
import logging


class AddModel(nn.Module):
    def __init__(self):
        super(AddModel, self).__init__()

    def forward(self, x, y):
        return x + y  # Element-wise addition


mod = AddModel()


logging.basicConfig(level=logging.INFO)  # Or INFO, WARNING, etc.

import onnxmlirtorch

my_option = {
    "compiler_image_name": None,
    "compile_options": "-O3",
    "compiler_path": "/workdir/onnx-mlir/build/Debug/bin/onnx-mlir",
}


opt_mod = torch.compile(mod, backend="onnxmlir", options=my_option, dynamic=True)


def run_model(x, y):
    print("Doing computation, input shape:", x.shape, y.shape)
    z = opt_mod(x, y)
    assert np.array_equal(z, x + y)
    print("Verified output.")
    return z


# First inference
print("\n1st inference: should compile")
x = torch.randn(2, 3)
y = torch.randn(2, 3)
z = run_model(x, y)

# Second inference
print("\n2nd inference: should reuse")
x = torch.randn(3, 3)
y = torch.randn(3, 3)
z = run_model(x, y)

print("\n3rd inference: should compile")
x = torch.randn(5)
y = torch.randn(5)
z = run_model(x, y)

# Forth inference
print("\n4th inference: should reuse")
x = torch.randn(2, 5)
y = torch.randn(2, 5)
z = run_model(x, y)

# Fifth inference
print("\n5th inference: should reuse")
x = torch.randn(7)
y = torch.randn(7)
z = run_model(x, y)
