FROM ubuntu:focal

WORKDIR /build

# install stuff that is needed for compiling LLVM, MLIR and ONNX
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y git cmake ninja-build libprotobuf-dev protobuf-compiler 
RUN DEBIAN_FRONTEND=noninteractive \
    apt-get update && \
    apt-get -y install \
        make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
        libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils \
        libffi-dev liblzma-dev
RUN git clone git://github.com/yyuu/pyenv.git .pyenv
RUN git clone https://github.com/yyuu/pyenv-virtualenv.git ~/.pyenv/plugins/pyenv-virtualenv

ENV HOME=/build
ENV PYENV_ROOT=$HOME/.pyenv
ENV PATH=$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

RUN pyenv install 3.7.0

RUN pyenv global 3.7.0
RUN pyenv rehash

# first install MLIR in llvm-project
#RUN git clone https://DLCadmin:af52950e231dac7157c3d78d263d7468375fb102@github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX MLIR.
#WORKDIR /build/llvm-project
#RUN git checkout 3ce0ad1b336e67a76d78ae7ff7d66fe127586620
RUN mkdir bin
ENV PATH=$PATH:/build/bin
COPY clone-mlir.sh bin/clone-mlir.sh
RUN chmod a+x bin/clone-mlir.sh
RUN clone-mlir.sh


RUN pwd
RUN ls -al
WORKDIR /build
RUN pwd
RUN ls -al

WORKDIR /build/llvm-project/build
RUN cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="host" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_ENABLE_RTTI=ON

RUN cmake --build . --target -- ${MAKEFLAGS}
RUN cmake --build . --target check-mlir
