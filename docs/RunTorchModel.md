<!--- SPDX-License-Identifier: Apache-2.0 -->

# Outlines
This document describes how to use onnx-mlir compiler to compile and run a torch model. 

1. [Installation](#installation)
2. [How to use](#howto)

# Installation <a name="installation"></a>

The package torch_onnxmlir depends on the package `om_pyrt`. Follow the instruction [here](https://github.com/onnx/onnx-mlir/tree/main/src/Runtime/python/om_pyrt) to install `om_pyrt`. 

## Install from local directory

If onnx-mlir source code already exists locally, the step of git clone can be skipped.
```bash
git clone --recursive https://github.com/onnx/onnx-mlir.git
cd onnx-mlir
pip3 install -e src/Runtime/python/torch_onnxmlir --prefix=/usr
```

## Install from pip repository
Not supported yet.

# How to use <a name="howto"></a>

Plese refer to [Readme.md](https://github.com/onnx/onnx-mlir/blob/main/src/Runtime/python/torch_onnxmlir/README.md) of the package.
