#!/bin/bash
# Export environment variables pointing to LLVM-Projects
export LLVM_PROJ_SRC=$(pwd)/llvm-project/
export LLVM_PROJ_BUILD=$(pwd)/llvm-project/build
HOME=/
PYENV_ROOT=$HOME/.pyenv
PATH=$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
eval "$(pyenv init -)"
# use python 3
pyenv shell 3.7.0
mkdir onnx-mlir/build && cd onnx-mlir/build
cmake ..
cmake --build . --target onnx-mlir
