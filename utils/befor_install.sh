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

pyenv install 2.7.15
pyenv install 3.7.0
pyenv install pypy2.7-6.0.0
pyenv install pypy3.5-6.0.0

pyenv global 2.7.15 3.7.0 pypy3.5-6.0.0 pypy2.7-6.0.0
pyenv rehash
mkdir build
cd build
git clone https://caomhin:429818b2b1e7471cbc99b8d11681eaf305974d91@github.ibm.com/$REPO/DLCpp
cd DLCpp
git submodule update --init --recursive
