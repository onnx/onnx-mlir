import numpy as np
import torch
import torch.nn as nn


class AddModel(nn.Module):
    def __init__(self):
        super(AddModel, self).__init__()

    def forward(self, x, y):
        return x + y  # Element-wise addition


mod = AddModel()

import onnxmlirtorch

# Use the default compiler container image to compile
# --verifyInputTensors is for debug purpose
my_option = {
    "compile_options": "--verifyInputTensors -O3 --maccel=NNPA  --march=arch14",
}

opt_mod = torch.compile(mod, backend="onnxmlir", options=my_option)

# First inference
input1 = torch.randn(64, 1024)
input2 = torch.randn(64, 1024)
output = opt_mod(input1, input2)
print("output: ", output[0].shape)
