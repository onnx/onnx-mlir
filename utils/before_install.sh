#!/bin/bash
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y git cmake ninja-build libprotobuf-dev protobuf-compiler 
DEBIAN_FRONTEND=noninteractive \
    apt-get update && \
    apt-get -y install \
        make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
        libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils \
        libffi-dev liblzma-dev
git clone git://github.com/yyuu/pyenv.git .pyenv
git clone https://github.com/yyuu/pyenv-virtualenv.git ~/.pyenv/plugins/pyenv-virtualenv

REPO=$(echo $1 | cut -d / -f 5 -)
HOME=/
PYENV_ROOT=$HOME/.pyenv
PATH=$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

pyenv install 3.7.0

pyenv global 3.7.0
pyenv rehash
mkdir build
cd build
git clone https://git@github.com/$REPO/onnx-mlir
cd onnx-mlir
git submodule update --init --recursive
