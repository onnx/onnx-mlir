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

# Use the default compiler container image to compile
# --verifyInputTensors is for debug purpose
my_option = {
    "compile_options": "--verifyInputTensors",
}

opt_mod = torch.compile(mod, backend="onnxmlir", options=my_option)

# First inference
input = torch.randn(2)
output = opt_mod(input, input)
print("output: ", output)


# Second inference
input1 = torch.randn(3)
input2 = torch.randn(3)
output1 = opt_mod(input1, input2)
print("output: ", output1)

input3 = torch.randn(2)
output2 = opt_mod(input3, input3)
print("output: ", output2)
