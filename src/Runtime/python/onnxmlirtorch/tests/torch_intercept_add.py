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

opt_mod = torch.ompile(mod)

input1=torch.randn(2)

input2=torch.randn(2)

print(opt_mod(input1, input2))

"""


import onnxmlirtorch

opt_mod = onnxmlirtorch.interceptForward(mod)

# First inference
input = torch.randn(2)
output = opt_mod(input, input)
print("output: ", output)


# Second inference
input1 = torch.randn(3)
input2 = torch.randn(3)
output1 = opt_mod(input1, input2)
print("output: ", output1)

input7 = torch.randn(4)
output = opt_mod(input7, input7)
print(output)

input3 = torch.randn(2)
output2 = opt_mod(input3, input3)
print("output: ", output2)

input4 = torch.randn(5)
input44 = torch.randn(5)
output = opt_mod(input4, input4)
print(output)

input5 = torch.randn(3)
output = opt_mod(input5, input5)
print(output)

input6 = torch.randn(2)
output = opt_mod(input6, input6)
print(output)
