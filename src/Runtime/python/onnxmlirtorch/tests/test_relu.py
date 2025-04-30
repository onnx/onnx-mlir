import numpy as np
import torch

const_N = 10
const_M = 10


class MyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(const_N, const_M)

    def forward(self, x):
        return torch.nn.functional.relu(self.lin(x))


mod = MyModule()


"""
# Torch code
opt_mod = torch.compile(mod)

#print(opt_mod(torch.randn(const_N, const_M)))
input=torch.randn(const_N, const_M)

# Be careful of the default data type of torch tensor and np tensor
#input=np.random.rand(const_N, const_M)
print(opt_mod(input))
"""


import onnxmlirtorch

# opt_mod = onnxmlirtorch.compile(mod, compiler_path="/gpfs/projects/s/stco/users/chentong/Projects/onnx-mlir-compiler/onnx-mlir/build-1/Debug/bin/onnx-mlir", compiler_image_name=None)

opt_mod = onnxmlirtorch.compile(mod)
input = torch.randn(const_N, const_M)
output = opt_mod(input)
print(output)
