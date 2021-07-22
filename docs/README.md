<!--- SPDX-License-Identifier: Apache-2.0 -->

# ONNX MLIR
The Open Neural Network Exchange implementation in MLIR (http://onnx.ai/onnx-mlir/).

| System        | Build Status |
|---------------|--------------|
| s390x-Linux   | [![Build Status](https://yktpandb.watson.ibm.com/jenkins/buildStatus/icon?build=lastSuccessful:${params.GITHUB_PR_NUMBER_PUSH=master})](https://yktpandb.watson.ibm.com/jenkins/job/ONNX-MLIR-Pipeline-Docker-Build/)             |
| ppc64le-Linux | [![Build Status](https://yktpandb.watson.ibm.com/jenkinp/buildStatus/icon?build=lastSuccessful:${params.GITHUB_PR_NUMBER_PUSH=master})](https://yktpandb.watson.ibm.com/jenkinp/job/ONNX-MLIR-Pipeline-Docker-Build/)             |
| amd64-Linux   | [![Build Status](https://yktpandb.watson.ibm.com/jenkinx/buildStatus/icon?build=lastSuccessful:${params.GITHUB_PR_NUMBER_PUSH=master})](https://yktpandb.watson.ibm.com/jenkinx/job/ONNX-MLIR-Pipeline-Docker-Build/)             |
| amd64-Windows | [![Build Status](https://dev.azure.com/onnx-pipelines/onnx/_apis/build/status/MLIR-Windows-CI?branchName=master)](https://dev.azure.com/onnx-pipelines/onnx/_build/latest?definitionId=9&branchName=master)             |
| amd64-macOS   | [![Build Status](https://github.com/onnx/onnx-mlir/workflows/Build%20x86%20onnx-mlir%20on%20macOS/badge.svg)](https://github.com/onnx/onnx-mlir/actions?query=workflow%3A%22Build+x86+onnx-mlir+on+macOS%22)             |

## Prebuilt Containers
An easy way to get started with ONNX-MLIR is to use a prebuilt docker image.
These images are created as a result of a successful merge build on the trunk.
This means that the latest image represents the tip of the trunk.
Currently there are both Release and Debug mode images for `amd64`, `ppc64le` and `s390x` saved in Docker Hub as, respectively, [onnxmlirczar/onnx-mlir](https://hub.docker.com/r/onnxmlirczar/onnx-mlir) and [onnxmlirczar/onnx-mlir-dev](https://hub.docker.com/r/onnxmlirczar/onnx-mlir-dev).
To use one of these images either pull it directly from Docker Hub, launch a container and run an interactive bash shell in it, or use it as the base image in a dockerfile.
The onnx-mlir image just contains the built compiler and you can use it immediately to compile your model without any installation. A python convenience script is provided to allow you to run ONNX-MLIR inside a docker container as if running the ONNX-MLIR compiler directly on the host. For example,
```
# docker/onnx-mlir.py --EmitLib mnist/model.onnx
505a5a6fb7d0: Pulling fs layer
505a5a6fb7d0: Verifying Checksum
505a5a6fb7d0: Download complete
505a5a6fb7d0: Pull complete
Shared library model.so has been compiled.
```
The script will pull the onnx-mlir image if it's not available locally, mount the directory containing the `model.onnx` into the container, and compile and generate the `model.so` in the same directory.

The onnx-mlir-dev image contains the full build tree including the prerequisites and a clone of the source code.
The source can be modified and onnx-mlir rebuilt from within the container, so it is possible to use it
as a development environment.
It is also possible to attach vscode to the running container.
An example Dockerfile useful for development and vscode configuration files can be seen in the docs folder.
If the workspace directory and the vscode files are not present in the directory where the Docker build is run, then the lines referencing them should be commented out or deleted.
The Dockerfile is shown here.

[same-as-file]: <> (docs/docker-example/Dockerfile)
```
FROM onnxmlirczar/onnx-mlir-dev
WORKDIR /workdir
ENV HOME=/workdir

# 1) Install packages.
ENV PATH=$PATH:/workdir/bin
RUN apt-get update
RUN apt-get install -y python-numpy
RUN apt-get install -y python3-pip
RUN python -m pip install --upgrade pip
RUN apt-get install -y gdb
RUN apt-get install -y lldb
RUN apt-get install -y emacs
RUN apt-get install -y vim
# 2) Instal optional packages, uncomment/add as you see fit.
# RUN apt-get install -y valgrind
# RUN apt-get install -y libeigen3-dev
# RUN apt-get install -y clang-format
# RUN python -m pip install wheel
# RUN python -m pip install numpy
# RUN python -m pip install torch==1.6.0+cpu torchvision==0.7.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
# RUN git clone https://github.com/onnx/tutorials.git
# Install clang-12.
# RUN apt-get install -y lsb-release wget software-properties-common
# RUN bash -c "$(wget -O - https://apt.llvm.org/llvm.sh)"

# 3) When using vscode, copy your .vscode in the Dockerfile dir and
#    uncomment the two lines below.
# WORKDIR /workdir/.vscode
# ADD .vscode /workdir/.vscode

# 4) When using a personal workspace folder, set your workspace sub-directory
#    in the Dockerfile dir and uncomment the two lines below.
# WORKDIR /workdir/workspace
# ADD workspace /workdir/workspace

# 5) Fix git by reattaching head and making git see other branches than master.
WORKDIR /workdir/onnx-mlir
RUN git checkout master
RUN git fetch --unshallow

# 6) Set the PATH environment vars for make/debug mode. Replace Debug
#    with Release in the PATH below when using Release mode.
WORKDIR /workdir
ENV MLIR_DIR=/workdir/llvm-project/build/lib/cmake/mlir
ENV NPROC=4
ENV PATH=$PATH:/workdir/onnx-mlir/build/Debug/bin/:/workdir/onnx-mlir/build/Debug/lib:/workdir/llvm-project/build/bin
```

## Prerequisites

```
gcc >= 6.4
libprotoc >= 3.11.0
cmake >= 3.15.4
ninja >= 1.10.2
```
GCC can be found [here](https://gcc.gnu.org/install/), or if you have [Homebrew](https://docs.brew.sh/Installation), you can use `brew install gcc`. To check what version of gcc you have installed, run `gcc --version`.

The instructions to install libprotoc can be found [here](http://google.github.io/proto-lens/installing-protoc.htm). Or alternatively, if you have Homebrew, you can run `brew install protobuf`. To check what version you have installed, run `protoc --version`.

Cmake can be found [here](https://cmake.org/download/). However, to use Cmake, you need to follow the "How to Install For Command Line Use" tutorial, which can be found in Cmake under Tools>How to Install For Command Line Use. To check which version you have, you can either look in the desktop version under CMake>About, or run `cmake --version`.

The instructions for installing Ninja can be found [here](https://ninja-build.org/). Or, using Homebrew, you can run `brew install ninja`. To check the version, run `ninja --version`.



At any point in time, ONNX MLIR depends on a specific commit of the LLVM project that has been shown to work with the project. Periodically the maintainers
need to move to a more recent LLVM level. Among other things, this requires that the commit string in utils/clone-mlir.sh be updated. A consequence of
making this change is that the TravisCI build will fail until the Docker images that contain the prereqs are rebuilt. There is a GitHub workflow that rebuilds
this image for the amd64 architecture, but currently the ppc64le and s390x images must be rebuilt manually. The Dockerfiles to accomplish that are in the repo.

## Installation on UNIX

#### MLIR
Firstly, install MLIR (as a part of LLVM-Project):

[same-as-file]: <> (utils/clone-mlir.sh)
``` bash
git clone https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX MLIR.
cd llvm-project && git checkout 23dd750279c9e32ea631cc9e92c4413c7a3df60a && cd ..
```

[same-as-file]: <> (utils/build-mlir.sh)
``` bash
mkdir llvm-project/build
cd llvm-project/build
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_TARGETS_TO_BUILD="host" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_ENABLE_RTTI=ON

cmake --build . -- ${MAKEFLAGS}
cmake --build . --target check-mlir
```

#### ONNX-MLIR (this project)
The following environment variables can be set before building onnx-mlir (or alternatively, they need to be passed as CMake variables):
- MLIR_DIR should point to the mlir cmake module inside an llvm-project build or install directory (e.g., llvm-project/build/lib/cmake/mlir).

This project uses lit ([LLVM's Integrated Tester](http://llvm.org/docs/CommandGuide/lit.html)) for unit tests. When running CMake, we can also specify the path to the lit tool from LLVM using the LLVM_EXTERNAL_LIT define but it is not required as long as MLIR_DIR points to a build directory of llvm-project. If MLIR_DIR points to an install directory of llvm-project, LLVM_EXTERNAL_LIT is required.

To build ONNX-MLIR, use the following commands:

[same-as-file]: <> ({"ref": "utils/install-onnx-mlir.sh", "skip-doc": 2})
```bash
git clone --recursive https://github.com/onnx/onnx-mlir.git

# Export environment variables pointing to LLVM-Projects.
export MLIR_DIR=$(pwd)/llvm-project/build/lib/cmake/mlir

mkdir onnx-mlir/build && cd onnx-mlir/build
cmake ..
cmake --build .

# Run lit tests:
export LIT_OPTS=-v
cmake --build . --target check-onnx-lit
```

If you are running on OSX Big Sur, you need to add `-DCMAKE_CXX_COMPILER=/usr/bin/c++`
to the `cmake ..` command due to changes in the compilers.
After the above commands succeed, an `onnx-mlir` executable should appear in the `bin` directory.

##### LLVM and ONNX-MLIR CMake variables

The following CMake variables from LLVM and ONNX MLIR can be used when compiling ONNX MLIR.

**MLIR_DIR**:PATH
  Path to to the mlir cmake module inside an llvm-project build or install directory (e.g., c:/repos/llvm-project/build/lib/cmake/mlir).
  This is required if **MLIR_DIR** is not specified as an environment variable.

**LLVM_EXTERNAL_LIT**:PATH
  Path to the lit tool. Defaults to an empty string and LLVM will find the tool based on **MLIR_DIR** if possible.
  This is required when **MLIR_DIR** points to an install directory.

## Installation on Windows
Building onnx-mlir on Windows requires building some additional prerequisites that are not available by default.

Note that the instructions in this file assume you are using [Visual Studio  2019 Community Edition](https://visualstudio.microsoft.com/downloads/). It is recommended that you have the **Desktop development with C++** and **Linux development with C++** workloads installed. This ensures you have all toolchains and libraries needed to compile this project and its dependencies on Windows.

Run all the commands from a shell started from **"Developer Command Prompt for VS 2019"**.

#### Protobuf
Build protobuf as a static library.

[same-as-file]: <> (utils/install-protobuf.cmd)
```shell
git clone --recurse-submodules https://github.com/protocolbuffers/protobuf.git
REM Check out a specific branch that is known to work with ONNX MLIR.
REM This corresponds to the v3.11.4 tag
cd protobuf && git checkout d0bfd5221182da1a7cc280f3337b5e41a89539cf && cd ..

set root_dir=%cd%
md protobuf_build
cd protobuf_build
call cmake %root_dir%\protobuf\cmake -G "Visual Studio 16 2019" -A x64 -T host=x64 ^
   -DCMAKE_INSTALL_PREFIX="%root_dir%\protobuf_install" ^
   -DCMAKE_BUILD_TYPE=Release ^
   -Dprotobuf_BUILD_EXAMPLES=OFF ^
   -Dprotobuf_BUILD_SHARED_LIBS=OFF ^
   -Dprotobuf_BUILD_TESTS=OFF ^
   -Dprotobuf_MSVC_STATIC_RUNTIME=OFF ^
   -Dprotobuf_WITH_ZLIB=OFF

call cmake --build . --config Release -- /m
call cmake --build . --config Release --target install -- /m
```

Before running CMake for onnx-mlir, ensure that the bin directory to this protobuf is before any others in your PATH:
```shell
set PATH=%root_dir%\protobuf_install\bin;%PATH%
```

#### MLIR
Install MLIR (as a part of LLVM-Project):

[same-as-file]: <> (utils/clone-mlir.sh)
```shell
git clone https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX MLIR.
cd llvm-project && git checkout 23dd750279c9e32ea631cc9e92c4413c7a3df60a && cd ..
```

[same-as-file]: <> (utils/build-mlir.cmd)
```shell
set root_dir=%cd%
md llvm-project\build
cd llvm-project\build
call cmake %root_dir%\llvm-project\llvm -G "Visual Studio 16 2019" -A x64 -T host=x64 ^
   -DCMAKE_INSTALL_PREFIX="%root_dir%\llvm-project\build\install" ^
   -DLLVM_ENABLE_PROJECTS=mlir ^
   -DLLVM_TARGETS_TO_BUILD="host" ^
   -DCMAKE_BUILD_TYPE=Release ^
   -DLLVM_ENABLE_ASSERTIONS=ON ^
   -DLLVM_ENABLE_RTTI=ON ^
   -DLLVM_ENABLE_ZLIB=OFF

call cmake --build . --config Release -- /m
call cmake --build . --config Release --target install -- /m
call cmake --build . --config Release --target check-mlir -- /m
```

#### ONNX-MLIR (this project)
The following environment variables can be set before building onnx-mlir (or alternatively, they need to be passed as CMake variables):
- MLIR_DIR should point to the mlir cmake module inside an llvm-project build or install directory (e.g., c:/repos/llvm-project/build/lib/cmake/mlir).

This project uses lit ([LLVM's Integrated Tester](http://llvm.org/docs/CommandGuide/lit.html)) for unit tests. When running CMake, we can also specify the path to the lit tool from LLVM using the LLVM_EXTERNAL_LIT define but it is not required as long as MLIR_DIR points to a build directory of llvm-project. If MLIR_DIR points to an install directory of llvm-project, LLVM_EXTERNAL_LIT is required.

To build ONNX MLIR, use the following commands:

[same-as-file]: <> ({"ref": "utils/build-onnx-mlir.cmd", "skip-doc": 2})
```shell
git clone --recursive https://github.com/onnx/onnx-mlir.git

set root_dir=%cd%

md onnx-mlir\build
cd onnx-mlir\build
call cmake %root_dir%\onnx-mlir -G "Visual Studio 16 2019" -A x64 -T host=x64 ^
   -DCMAKE_BUILD_TYPE=Release ^
   -DCMAKE_PREFIX_PATH=%root_dir%\protobuf_install ^
   -DLLVM_LIT_ARGS=-v ^
   -DMLIR_DIR=%root_dir%\llvm-project\build\lib\cmake\mlir

call cmake --build . --config Release --target onnx-mlir -- /m
```

To test ONNX MLIR, use the following command:

[same-as-file]: <> ({"ref": "utils/check-onnx-mlir.cmd", "skip-ref": 1})
```shell
call cmake --build . --config Release --target check-onnx-lit -- /m
```

After the above commands succeed, an `onnx-mlir` executable should appear in the `bin` directory.

##### LLVM and ONNX-MLIR CMake variables

The following CMake variables from LLVM and ONNX MLIR can be used when compiling ONNX MLIR.

**MLIR_DIR**:PATH
  Path to to the mlir cmake module inside an llvm-project build or install directory (e.g., c:/repos/llvm-project/build/lib/cmake/mlir).
  This is required if **MLIR_DIR** is not specified as an environment variable.

**LLVM_EXTERNAL_LIT**:PATH
  Path to the lit tool. Defaults to an empty string and LLVM will find the tool based on **MLIR_DIR** if possible.
  This is required when **MLIR_DIR** points to an install directory.

## Using ONNX-MLIR

The usage of `onnx-mlir` is as such:
```
OVERVIEW: ONNX MLIR modular optimizer driver

USAGE: onnx-mlir [options] <input file>

OPTIONS:

Generic Options:

  --help        - Display available options (--help-hidden for more)
  --help-list   - Display list of available options (--help-list-hidden for more)
  --version     - Display the version of this program

ONNX MLIR Options:
These are frontend options.

  Choose target to emit:
      --EmitONNXBasic - Ingest ONNX and emit the basic ONNX operations without inferred shapes.
      --EmitONNXIR    - Ingest ONNX and emit corresponding ONNX dialect.
      --EmitMLIR      - Lower model to MLIR built-in transformation dialect.
      --EmitLLVMIR    - Lower model to LLVM IR (LLVM dialect).
      --EmitLib       - Lower model to LLVM IR, emit (to file) LLVM bitcode for model, compile and link it to a shared library.
```

## Example

For example, to lower an ONNX model (e.g., add.onnx) to ONNX dialect, use the following command:
```shell
./onnx-mlir --EmitONNXIR add.onnx
```
The output should look like:
```mlir
module {
  func @main_graph(%arg0: tensor<10x10x10xf32>, %arg1: tensor<10x10x10xf32>) -> tensor<10x10x10xf32> {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<10x10x10xf32>, tensor<10x10x10xf32>) -> tensor<10x10x10xf32>
    return %0 : tensor<10x10x10xf32>
  }
}
```

## Troubleshooting

If the latest LLVM project fails to work due to the latest changes to the MLIR subproject please consider using a slightly older version of LLVM. One such version, which we use, can be found [here](https://github.com/clang-ykt/llvm-project).

## Installing `third_party ONNX` for Backend Tests or Rebuilding ONNX Operations

Backend tests are triggered by `make check-onnx-backend` in the build directory and require a few preliminary steps to run successfully. Similarily, rebuilding the ONNX operations in ONNX-MLIR from their ONNX descriptions is triggered by `make OMONNXOpsIncTranslation`.

You will need to install python 3.x if its not default in your environment, and possibly set the cmake `PYTHON_EXECUTABLE` varialbe in your top cmake file.

You will also need `pybind11` which may need to be installed (mac: `brew install pybind11` for example) and you may need to indicate where to find the software (Mac, POWER, possibly other platforms: `export pybind11_DIR=<your path to pybind>`). Then install the `third_party/onnx` software (Mac: `pip install -e third_party/onnx`) typed in the top directory.

On Macs/POWER and possibly other platforms, there is currently an issue that arises when installing ONNX. If you get an error during the build, try a fix where you edit the top CMakefile as reported in this PR: `https://github.com/onnx/onnx/pull/2482/files`.

## Slack channel

We have a slack channel established under the Linux Foundation AI and Data Workspace, named `#onnx-mlir-discussion`. This channel can be used for asking quick questions related to this project. A direct link is [here](https://lfaifoundation.slack.com/archives/C01J4NAL4A2).
