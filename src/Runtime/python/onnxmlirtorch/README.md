This package provides a python interface to use onnx-mlir compiler to run inference of torch model. The basic parameters of the interface are supported with options ignored. 

## Description
Let's start with a simple torch model:
```
import torch
import torch.nn as nn

class AddModel(nn.Module):
    def __init__(self):
        super(AddModel, self).__init__()
    
    def forward(self, x, y):
        return x + y  # Element-wise addition

mod = AddModel()
opt_mod = torch.compile(mod)
input1=torch.randn(2)
input2=torch.randn(2)
print(opt_mod(input1, input2))

```

With onnxmlirtorch package, the inference part can be rewritten as follows:
```
import onnxmlirtorch

opt_mod = onnxmlirtorch.compile(mod)
input1=torch.randn(2)
input2=torch.randn(2)
print(opt_mod(input1, input2))

```

## Installation


### Install from local directory
First create the source of the package.
In onnx-mlir/build, `cmake --build . --target OMCreateONNXMLIRTorchPackage`
At top of onnx-mlir: `pip3 install -e src/Runtime/python/onnxmlirtorch`

### Install from repo
After the package is uploaded to pip server, you can install with 'pip3 install onnxmlirtorch`

