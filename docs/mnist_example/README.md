Table of Contents
=================

   * [Train Model in PyTorch, Compile using ONNX-MLIR](#train-model-in-pytorch-compile-using-onnx-mlir)
      * [Training the Model](#training-the-model)
      * [Environment Variables Setup:](#environment-variables-setup)
      * [Compile Model](#compile-model)
      * [Write a C Driver Code](#write-a-C-driver-code)
         * [Inference Entry Point](#inference-entry-point)
         * [Feeding Inputs and Retrieving Results](#feeding-inputs-and-retrieving-results)
      * [Write a Python Driver Code](#write-a-Python-driver-code)

# Train Model in PyTorch, Compile using ONNX-MLIR

In this example, we will demonstrate training a mnist model in PyTorch and compile, run it using only C++.

## Training the Model

Make sure that dependent python packages specified in `requirements.txt` are installed.
Run the training script using the following command:
```bash
python gen_mnist_onnx.py --epochs=1 --batch-size=128 --export-onnx --save-model
```

Which basically says, train the model for 1 epoch using a batch size of 128. Such configuration encourages a speedy training process.
The flag `--export-onnx` will export the trained model to an ONNX protobuf object.
The flag `--save-model` will save a snapshot of the trained model.

The model is a simple neural network defined as such:
```python
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(14*14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.max_pool2d(x, 2)
        x = x.reshape(-1, 1*14*14)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.softmax(x, dim=1)
        return output
```

After training is complete, an onnx model named `mnist.onnx` should appear.
If you are interested in knowing how to export a pytorch model, here's the relevant code snippet:
```python
  model = Net()
  # ...
  # Train...
  # ...
  input_names = ["image"]
  output_names = ["prediction"]
  dummy_input = torch.randn(1, 1, 28, 28)
  torch.onnx.export(model,
                    dummy_input,
                    "mnist.onnx",
                    verbose=True,
                    input_names=input_names,
                    output_names=output_names)
```

Upon inspection, it should look like:

![alt text](mnist-simple.png "Simple MNIST Model")

## Environment Variables Setup:

Now we are ready to compile the model! To make it easier to invoke commands and include header files, I updated my environment variables as such:

```bash
# ONNX_MLIR_ROOT points to the root of the onnx-mlir, 
# under which the include and the build directory lies.
export ONNX_MLIR_ROOT=$(pwd)/../..
# Define the include directory where onnx-mlir runtime include files resides.
# Change only if you have a non-standard install.
export ONNX_MLIR_INCLUDE=$ONNX_MLIR_ROOT/include
# Define the bin directory where onnx-mlir binary resides. Change only if you
# have a non-standard install.
export ONNX_MLIR_BIN=$ONNX_MLIR_ROOT/build/Debug/bin

# Include ONNX-MLIR executable directories part of $PATH.
export PATH=$ONNX_MLIR_ROOT/build/Debug/bin:$PATH
```

Run these commands directly in the docs/docs/mnist_example and everything should work fine. You may also simply execute `. update_env.sh`

## Compile Model

Firstly, we invoke `onnx-mlir` to compile the trained onnx model into LLVM bitcode:
```bash
onnx-mlir -O3 mnist.onnx
```

A `mnist.so` should appear, which corresponds to the compiled model object file.

## Write a C Driver Code

To invoke the compiled model, we need to know the entry point signature with which to call into the model inference function, and based on it, engineer a C++ driver that feeds test data into this inference function and retrieve the prediction results.

### Inference Entry Point

The signature of the model inference function for all models is:
```cpp
extern "C" OMTensorList *run_main_graph(OMTensorList *);
```

I.e., all models ingests an `OMTensorList*`, and returns an `OMTensorList*`. Documentation of the data structures are found [here](https://onnx.ai/onnx-mlir), with the C interface for Tensor [here](https://onnx.ai/onnx-mlir/doxygen_html/OMTensor/_o_m_tensor_8h.html) and TensorList [here](https://onnx.ai/onnx-mlir/doxygen_html/OMTensorList/_o_m_tensor_list_8h.html).

### Feeding Inputs and Retrieving Results

To invoke the inference function, we use the following code to communicate with the compiled model inference function.

```cpp
#include <iostream>
#include <vector>

#include "OnnxMlirRuntime.h"

// Declare the inference entry point.
extern "C" OMTensorList *run_main_graph(OMTensorList *);

static float img_data[] = {...};

int main() {
  // Create an input tensor list of 1 tensor.
  int inputNum = 1;
  OMTensor **inputTensors = (OMTensor **)malloc(inputNum * sizeof(OMTensor *));
  // The first input is of tensor<1x1x28x28xf32>.
  int64_t rank = 4;
  int64_t shape[] = {1, 1, 28, 28};
  OMTensor *tensor = omTensorCreate(img_data, shape, rank, ONNX_TYPE_FLOAT);
  // Create a tensor list.
  inputTensors[0] = tensor;
  OMTensorList *tensorListIn = omTensorListCreate(inputTensors, inputNum);

  // Compute outputs.
  OMTensorList *tensorListOut = run_main_graph(tensorListIn);

  // Extract the output. The model defines one output of type tensor<1x10xf32>.
  OMTensor *y = omTensorListGetOmtByIndex(tensorListOut, 0);
  float *prediction = (float *)omTensorGetDataPtr(y);

  // Analyze the output.
  int digit = -1;
  float prob = 0.;
  for (int i = 0; i < 10; i++) {
    printf("prediction[%d] = %f\n", i, prediction[i]);
    if (prediction[i] > prob) {
      digit = i;
      prob = prediction[i];
    }
  }

  printf("The digit is %d\n", digit);
  return 0;
}
```

Now, putting everything together, we invoke g++ to compile and link together the driver code, C runtime API and the compiled model inference function:
```bash
g++ --std=c++11 -O3 mnist.cpp ./mnist.so -o mnist -I $ONNX_MLIR_INCLUDE
```

Now run it by calling `./mnist`! It outputs the following for the image in the test:

``` shell
prediction[0] = 1.000000
prediction[1] = 0.000000
prediction[2] = 0.000000
prediction[3] = 0.000000
prediction[4] = 0.000000
prediction[5] = 0.000000
prediction[6] = 0.000000
prediction[7] = 0.000000
prediction[8] = 0.000000
prediction[9] = 0.000000
The digit is 0.
```

The full code is available [here](mnist.cpp).

## Write a Python Driver Code

You will find most of the details of they Python driver interface described [here](../UsingPyRuntime.md). We summarize here quickly how to execute mnist in python.

First, we include the necessary Python runtime library. The library path can be set by using the PYTHONPATH or simply creating a soft link in the current directory to the Python shared library (typically: `build/Debug/lib/PyRuntime.cpython-<target>.so`).

``` Python
import numpy as np
from PyRuntime import ExecutionSession
```

The runtime use an `ExecutionSession` object to hold a specific model and entry point. On this object, we can perform in inference using the `run(input)` call where `input` is a list of numpy arrays. The signature of the input and output model can be extracted using, respectively, the `input_signature()` and `output_signature()` formatted as JSON strings. The code is shown below.

``` Python
# Load the model mnist.so compiled with onnx-mlir.
model = 'mnist.so'
session = ExecutionSession(model)
# Print the models input/output signature, for display.
# If there are problems with the signature functions, they can be simply commented out.
print("input signature in json", session.input_signature())
print("output signature in json", session.output_signature())
# Create an input arbitrarily filled of 1.0 values (file has the actual values).
input = np.full((1, 1, 28, 28), 1, np.dtype(np.float32))
# Run the model.
outputs = session.run([input])
```
The outputs can then be analyzed by inspecting the values inside the `output` list of numpy arrays.

The full code is available [here](mnist.py). It finds that `0` is the most likely digit for the given input. The command is:
``` shell
python mnist.py
```
and produces this output.
``` shell
input signature in json [    { "type" : "f32" , "dims" : [1 , 1 , 28 , 28] , "name" : "image" }
]
output signature in json [   { "type" : "f32" , "dims" : [1 , 10] , "name" : "prediction" }
]
prediction  0 = 0.9999999
prediction  1 = 6.745636e-18
prediction  2 = 5.504603e-09
prediction  3 = 9.146374e-12
prediction  4 = 3.2389183e-15
prediction  5 = 1.2362976e-07
prediction  6 = 9.871477e-12
prediction  7 = 2.1788185e-13
prediction  8 = 2.0332518e-08
prediction  9 = 1.6744228e-15
The digit is 0
```

