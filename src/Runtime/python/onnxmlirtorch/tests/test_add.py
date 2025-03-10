import numpy as np
import torch
import torch.nn as nn

class AddModel(nn.Module):
    def __init__(self):
        super(AddModel, self).__init__()
    
    def forward(self, x, y):
        return x + y  # Element-wise addition

mod = AddModel()


"""
#  PyTorch code

opt_mod = torch.compile(mod)

input1=torch.randn(2)
input2=torch.randn(2)
print(opt_mod(input1, input2))

"""


import onnxmlirtorch

# Example of using default compiler image
opt_mod = onnxmlirtorch.compile(mod)

# Example of using local compiler
#opt_mod = onnxmlirtorch.compile(mod, compiler_image_name=None, compiler_path="/gpfs/projects/s/stco/users/chentong/Projects/onnx-mlir-compiler/onnx-mlir/build-1/Debug/bin/onnx-mlir")

input1=torch.randn(2)
input2=torch.randn(2)
print(opt_mod(input1, input2))

# Test of cache
input=torch.randn(2)
output = opt_mod(input, input)
print(output)

input=torch.randn(3)
output = opt_mod(input, input)
print(output)

input=torch.randn(4)
output = opt_mod(input, input)
print(output)

input=torch.randn(5)
output = opt_mod(input, input)
print(output)

input=torch.randn(3)
output = opt_mod(input, input)
print(output)

input=torch.randn(2)
output = opt_mod(input, input)
print(output)

