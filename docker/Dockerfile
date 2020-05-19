FROM caoimhinuibrian/onnx-mlir-llvmimage:initial

WORKDIR /build
ENV HOME=/build
ENV PYENV_ROOT=$HOME/.pyenv
ENV PATH=$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN pyenv global 3.7.0
RUN pyenv rehash

RUN mkdir bin
ENV PATH=$PATH:/build/bin
RUN git clone --recursive https://github.com/onnx/onnx-mlir.git
COPY compile-onnx-mlir.sh bin/compile-onnx-mlir.sh
RUN compile-onnx-mlir.sh
WORKDIR /build/onnx-mlir/build
ENV LIT_OPT=v
RUN cmake --build . --target check-onnx-lit
