FROM onnxmlirczar/onnx-mlir-build:x86

WORKDIR /build
ENV HOME=/build
ENV PYENV_ROOT=$HOME/.pyenv
ENV PATH=$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN pyenv global 3.7.0
RUN pyenv rehash

ENV PATH=$PATH:/build/bin
RUN apt-get update
RUN apt-get install -y python-numpy
RUN apt-get install -y python3-pip
RUN apt-get install -y gdb
RUN apt-get install -y lldb
RUN apt-get install -y emacs
WORKDIR /build/.vscode
ADD .vscode /build/.vscode
WORKDIR /build
