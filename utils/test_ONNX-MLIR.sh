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
# Run FileCheck tests:
export LIT_OPTS=-v
cmake --build . --target check-onnx-lit
