This package provides an onnxmlir-based compiler backend for torch.compile().

## Usage
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

# Compile the model.
opt_mod = torch.compile(mod)

input1=torch.randn(2)
input2=torch.randn(2)
print(opt_mod(input1, input2))

```

With torch_onnxmlir package, `torch.compile()` can be rewritten as follows:
```
import torch
import torch.nn as nn
import torch_onnxmlir

class AddModel(nn.Module):
    def __init__(self):
        super(AddModel, self).__init__()
    
    def forward(self, x, y):
        return x + y  # Element-wise addition

mod = AddModel()

# Compile the model using onnxmlir backend in the torch_onnxmlir package.
om_option = {
    "compiler_image_name": None,
    "compile_options": "-O3",
    "compiler_path": "/workdir/onnx-mlir/build/Debug/bin/onnx-mlir",
}
opt_mod = torch.compile(mod, backend="onnxmlir", options=om_options)

input1=torch.randn(2)
input2=torch.randn(2)
print(opt_mod(input1, input2))

```

For more information about `torch.compile`, see its [document](https://docs.pytorch.org/docs/stable/generated/torch.compile.html).

## Installation


### Install from local directory
First create the source of the package.
In onnx-mlir/build, `cmake --build . --target OMCreateTorchONNXMLIRPackage`
At top of onnx-mlir: `pip3 install -e src/Runtime/python/torch_onnxmlir`

### Install from repo
After the package is uploaded to pip server, you can install with 'pip3 install torch_onnxmlir`

