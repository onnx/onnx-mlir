#!/bin/bash
# Export environment variables pointing to LLVM-Projects.
export LLVM_PROJ_SRC=/llvm-project/
export LLVM_PROJ_BUILD=/llvm-project/build
HOME=/
PYENV_ROOT=$HOME/.pyenv
PATH=$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
eval "$(pyenv init -)"
pyenv shell 3.7.0
cd build
mkdir dlcpp-build && cd dlcpp-build
cmake ../DLCpp
make -j64

# Test
make check-dlcpp
