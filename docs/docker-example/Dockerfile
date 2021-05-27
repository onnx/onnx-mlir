FROM onnxmlirczar/onnx-mlir-dev

WORKDIR /workdir
ENV HOME=/workdir

# install needed packages
ENV PATH=$PATH:/workdir/bin
RUN apt-get update
RUN apt-get install -y python-numpy
RUN apt-get install -y python3-pip
RUN python -m pip install --upgrade pip
RUN apt-get install -y gdb
RUN apt-get install -y lldb
RUN apt-get install -y emacs
RUN apt-get install -y libeigen3-dev
RUN apt-get install -y clang-format
# install clang-12
RUN apt-get install -y lsb-release wget software-properties-common
RUN bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"
RUN python -m pip install wheel
RUN python -m pip install numpy
RUN python -m pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN git clone https://github.com/onnx/tutorials.git


# set env variables for make - these should be set in any shell that is started
ENV LLVM_PROJ_SRC=/workdir/llvm-project
ENV LLVM_PROJ_WORKDIR=/workdir/llvm-project/build
ENV LLVM_PROJ_BUILD=/workdir/llvm-project/build
ENV NPROC=4

WORKDIR /workdir/onnx-mlir/build

# copy in vscode configuration info
WORKDIR /workdir/.vscode
ADD .vscode /workdir/.vscode

# copy workspace folder (typically contains local/personal testcases etc)
WORKDIR /workdir/workspace
ADD workspace /workdir/workspace

# do stuff to fix git
WORKDIR /workdir/onnx-mlir
# reattach HEAD
RUN git checkout master
# make git see branches other than master
RUN git config remote.origin.fetch "+refs/heads/*:refs/remotes/origin/*"

WORKDIR /workdir
ENV PATH=$PATH:/workdir/onnx-mlir/build/Debug/bin/:/workdir/onnx-mlir/build/Debug/lib:/workdir/llvm-project/build/Debug/bin

