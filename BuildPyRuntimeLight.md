# Building and Using PyRuntimeC in Lightweight Mode

## Overview

The onnx-mlir compiler can compile an ONNX model into a shared library (`.so` file) and provides a Python driver called PyRuntimeC to execute the generated shared library through Python code. 

Traditionally, PyRuntimeC is built alongside the onnx-mlir compiler, which requires building the entire llvm_project. This document describes a lightweight approach to build PyRuntimeC **without** requiring llvm_project or other onnx-mlir compiler components. This enables users to easily build the Python driver for model execution on different systems.

### What Gets Built

In lightweight mode, only the following components are built:
- **OMTensorUtils** (`src/Runtime`)
- **Python driver** (`src/Runtime/python`)
- **Utility functions**
- **third_party/pybind11**

The lightweight PyRuntimeC build is controlled by the CMake option: `ONNX_MLIR_TARGET_TO_BUILD=OMPyRt`

## Prerequisites

- CMake (version 3.15 or higher recommended)
- Python 3.x with pip
- C++ compiler with C++17 support
- onnx-mlir source code (cloned from repository)

## Building PyRuntimeC

Assuming you have cloned the onnx-mlir source code and are using a `build` directory for your normal onnx-mlir compiler build, you need to create a separate build directory for the lightweight PyRuntimeC build, for example build-light.

### Build Steps

1. **Create a new build directory:**
   ```bash
   git clone --recursive https://github.com/onnx/onnx-mlir.git
   mkdir onnx-mlir/build-light
   cd onnx-mlir/build-light
   ```

2. **Configure and build:**
   ```bash
   cmake .. -DONNX_MLIR_TARGET_TO_BUILD=OMPyRt
   make
   ```

## Installing PyRuntimeC

### 1. Set Up Python Virtual Environment

First, create and activate a Python virtual environment (recommended):

```bash
python -m venv path/to/store/your/venv
source path/to/store/your/venv/bin/activate
```

### 2. Build and Install the Package

From the `build-light` directory, execute:

```bash
# Create the package
cmake --build . --target OMCreateOMPyRtPackage

# Install the package
pip3 install src/Runtime/python/om_pyrt
```

Alternatively, for development mode (editable install):
```bash
pip3 install -e src/Runtime/python/om_pyrt
```

## Using PyRuntimeC

The Python driver can be integrated with different Python packages:

### Recommended Package: `om_pyrt`

The `om_pyrt` package (located in `src/Runtime/python/om_pyrt`) provides:
- Python code for model compilation
- Inference capabilities
- Test utilities

**Basic usage example:**
```python
import numpy as np
import om_pyrt

# Initialize the inference session with a compiled model
sess = om_pyrt.InferenceSession("./model.so")

# Prepare inputs
input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

# Run inference
outputs = sess.run([input_data])

# Process outputs
print(outputs)
```

### Integration with PyTorch: `torch_onnxmlir`

The driver is also used by `torch_onnxmlir`, which enables onnx-mlir to function as a PyTorch backend. Refer to [doc](RunTorchModel.md)

## Additional Resources

- For detailed information about the `om_pyrt` package, see `src/Runtime/python/om_pyrt/README.md`
- For model compilation utilities, see `src/Runtime/python/OMPyCompile/README.md`
- For general onnx-mlir build instructions, refer to the main documentation

## Troubleshooting

### Common Issues

- **CMake configuration fails**: Ensure you're using CMake 3.15 or higher
- **Python package installation fails**: Verify your virtual environment is activated
- **Import errors**: Confirm the package was installed successfully with `pip list | grep om_pyrt`%  
