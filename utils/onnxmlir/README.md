This package provides a python interface to use onnx-mlir compiler to run inference of an onnx model similar to onnxruntime interface. The basic parameters of the interface are supported with options ignored. 

## Description
Let's start with [an onnxrutime example](https://onnxruntime.ai/docs/get-started/with-python#pytorch-cv):
```
import onnxruntime as ort
import numpy as np
x, y = test_data[0][0], test_data[0][1]
ort_sess = ort.InferenceSession('fashion_mnist_model.onnx')
outputs = ort_sess.run(None, {'input': x.numpy()})

# Print Result
predicted, actual = classes[outputs[0][0].argmax(0)], classes[y]
print(f'Predicted: "{predicted}", Actual: "{actual}"')
```

With onnxmlir package, onnx-mlir can be used to replace onnxrutnime as follows:
```
import onnxmlir as ort
import numpy as np
x, y = test_data[0][0], test_data[0][1]
ort_sess = ort.InferenceSession('fashion_mnist_model.onnx')
outputs = ort_sess.run(None, {'input': x.numpy()})

# Print Result
predicted, actual = classes[outputs[0][0].argmax(0)], classes[y]
print(f'Predicted: "{predicted}", Actual: "{actual}"')
```

In current version, the onnx-mlir compiler is not contained in the python 
package yet. Use env variable ONNX_MLIR_HOME to specify the location of the onnx-mlir compiler to be used. For example `export ONNX_MLIR_HOME=/mypath/onnx-mlir/build/Debug`. 

Another way to run the onnx model is to precompile it into a static library first. 
```
onnx-mlir -O3 fashin_mnist_mode.onnx
```

This compilation will generate fashin_mnist_mode.so. Then the library can be used as model in the Python script as follows:
```
import onnxmlir as ort
import numpy as np
x, y = test_data[0][0], test_data[0][1]
ort_sess = ort.InferenceSession('fashion_mnist_model.so')
outputs = ort_sess.run(None, {'input': x.numpy()})

# Print Result
predicted, actual = classes[outputs[0][0].argmax(0)], classes[y]
print(f'Predicted: "{predicted}", Actual: "{actual}"')
```

This package supports list or dictionary for input and output. For example, the input for run could be list of tensor.
```
outputs = ort_sess.run(None, [a, b, c])
```

Another extra named argment for InferenceSession is introduced to specify the extra arguments accepted by onnx-mlir/utils/RunONNXModel.py. Here is an example:
```
sess = onnxmlir.inferenceSession("test_add.onnx", options='--compile-args="-O3 --parallel" --print-output')
```

## Installation

### Install from local directory
At top of onnx-mlir: `pip3 install utils/onnxmlir`

### Install from repo
After the package is uploaded, you can install with 'pip3 install onnxmlir`

