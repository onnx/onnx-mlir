ARG BASE_IMAGE

# By default, use ubuntu:focal;
FROM $BASE_IMAGE

WORKDIR /build
# force prereq build
# install stuff that is needed for compiling LLVM, MLIR and ONNX
RUN apt-get update
RUN apt-get -y install default-jdk
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y git cmake ninja-build libprotobuf-dev protobuf-compiler
RUN DEBIAN_FRONTEND=noninteractive \
    apt-get update && \
    apt-get -y install \
        make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev \
        libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils \
        libffi-dev liblzma-dev

RUN if [ ! -d .pyenv ]; then \
      git clone git://github.com/yyuu/pyenv.git .pyenv && \
      git clone https://github.com/yyuu/pyenv-virtualenv.git ~/.pyenv/plugins/pyenv-virtualenv; \
    fi

ENV HOME=/build
ENV PYENV_ROOT=$HOME/.pyenv
ENV PATH=$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

RUN if [ ! -f "./pyenv-installed" ]; then \
      pyenv install 3.7.0 && \
      pyenv global 3.7.0 && \
      pyenv rehash && \
      touch ./pyenv-installed; \
    fi

# first install MLIR in llvm-project
RUN mkdir -p bin
ENV PATH=$PATH:/build/bin
COPY clone-mlir.sh bin/clone-mlir.sh
RUN chmod a+x bin/clone-mlir.sh
RUN if [ ! -d /build/llvm-project ]; then \
      clone-mlir.sh; \
    fi

WORKDIR /build/llvm-project/build
RUN if [ ! -f "/build/llvm-project/build/CMakeCache.txt" ]; then \
      cmake -G Ninja ../llvm \
       -DLLVM_ENABLE_PROJECTS=mlir \
       -DLLVM_BUILD_EXAMPLES=ON \
       -DLLVM_TARGETS_TO_BUILD="host" \
       -DCMAKE_BUILD_TYPE=Release \
       -DLLVM_ENABLE_ASSERTIONS=ON \
       -DLLVM_ENABLE_RTTI=ON; \
    fi

# Build for 30 minutes:
RUN timeout 30m cmake --build . --target -- ${MAKEFLAGS} || true

# RUN cmake --build . --target check-mlir
