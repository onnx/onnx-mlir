ARG BASE
# Grab pre-built LLVM from docker hub
FROM $BASE

WORKDIR /build/bin
WORKDIR /build
ENV HOME=/build
ENV PYENV_ROOT=$HOME/.pyenv
ENV PATH=$PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH
RUN pyenv global 3.7.0
RUN pyenv rehash

ENV PATH=$PATH:/build/bin
