<!--- SPDX-License-Identifier: Apache-2.0 -->
<p align="center"><img width="50%" src="docs/logo/onnx-mlir-1280x640.png" /></p>

# ONNX-MLIR
The Open Neural Network Exchange implementation in MLIR (http://onnx.ai/onnx-mlir/).

| System        | Build Status |
|---------------|--------------|
| s390x-Linux   | [![Build Status](https://www.onnxmlir.xyz/jenkins/buildStatus/icon?job=ONNX-MLIR-Pipeline-Docker-Build&build=last:%24%7Bparams.GITHUB_PR_NUMBER_PUSH=main%7D&subject=Jenkins%20CI)](https://www.onnxmlir.xyz/jenkins/job/ONNX-MLIR-Pipeline-Docker-Build/)             |
| ppc64le-Linux | [![Build Status](https://www.onnxmlir.xyz/jenkinp/buildStatus/icon?job=ONNX-MLIR-Pipeline-Docker-Build&build=last:%24%7Bparams.GITHUB_PR_NUMBER_PUSH=main%7D&subject=Jenkins%20CI)](https://www.onnxmlir.xyz/jenkinp/job/ONNX-MLIR-Pipeline-Docker-Build/)             |
| amd64-Linux   | [![Build Status](https://www.onnxmlir.xyz/jenkinx/buildStatus/icon?job=ONNX-MLIR-Pipeline-Docker-Build&build=last:%24%7Bparams.GITHUB_PR_NUMBER_PUSH=main%7D&subject=Jenkins%20CI)](https://www.onnxmlir.xyz/jenkinx/job/ONNX-MLIR-Pipeline-Docker-Build/)             |
| amd64-Windows | [![Build Status](https://dev.azure.com/onnx-pipelines/onnx/_apis/build/status/MLIR-Windows-CI?branchName=main)](https://dev.azure.com/onnx-pipelines/onnx/_build/latest?definitionId=9&branchName=main)             |
| amd64-macOS   | [![Build Status](https://github.com/onnx/onnx-mlir/actions/workflows/macos-amd64-build.yml/badge.svg)](https://github.com/onnx/onnx-mlir/actions/workflows/macos-amd64-build.yml)             |

## Setting up ONNX-MLIR using Prebuilt Containers

The prefered approach to using and developing ONNX-MLIR is to used Docker Images and Containers, as getting the proper code dependences may be tricky on some systems. Our instructions on using ONNX-MLIR with dockers are [here](docs/Docker.md).

## Setting up ONNX-MLIR directly

### Prerequisites

<!-- Keep list below in sync with docs/Prerequisite.md. -->
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
cd llvm-project && git checkout 0bf230d4220660af8b2667506f8905df2f716bdf && cd ..
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
if [[ -z "$pythonLocation" ]]; then
  cmake -G Ninja -DCMAKE_CXX_COMPILER=/usr/bin/c++ ..
else
  cmake -G Ninja -DCMAKE_CXX_COMPILER=/usr/bin/c++ -DPython3_ROOT_DIR=$pythonLocation ..
fi
cmake --build .

# Run lit tests:
export LIT_OPTS=-v
cmake --build . --target check-onnx-lit
```

If you are running on OSX Big Sur, you need to add `-DCMAKE_CXX_COMPILER=/usr/bin/c++`
to the `cmake ..` command due to changes in the compilers. The environment variable 
`$pythonLocation` may be used to specify the base directory of the Python compiler.
After the above commands succeed, an `onnx-mlir` executable should appear in the `bin` directory.

##### LLVM and ONNX-MLIR CMake variables

The following CMake variables from LLVM and ONNX MLIR can be used when compiling ONNX MLIR.

**MLIR_DIR**:PATH
  Path to to the mlir cmake module inside an llvm-project build or install directory (e.g., c:/repos/llvm-project/build/lib/cmake/mlir).
  This is required if **MLIR_DIR** is not specified as an environment variable.

**LLVM_EXTERNAL_LIT**:PATH
  Path to the lit tool. Defaults to an empty string and LLVM will find the tool based on **MLIR_DIR** if possible.
  This is required when **MLIR_DIR** points to an install directory.

### MacOS Issues

There is a known issue when building onnx-mlir. If you see a error of this sorts
``` shell
Cloning into '/home/chentong/onnx-mlir/build/src/Runtime/jni/jsoniter'...

[...]

make[2]: *** [src/Runtime/jni/CMakeFiles/jsoniter.dir/build.make:74: src/Runtime/jni/jsoniter/target/jsoniter-0.9.23.jar] Error 127
make[1]: *** [CMakeFiles/Makefile2:3349: src/Runtime/jni/CMakeFiles/jsoniter.dir/all] Error 2
make: *** [Makefile:146: all] Error 2
```

Help to update the prerequisites is found [here](docs/Prerequisite.md).

At any point in time, ONNX-MLIR depends on a specific commit of the LLVM project that has been shown to work with the project. 
Periodically the maintainers need to move to a more recent LLVM level. 
Among other things, this requires to update the commit string in (utils/clone-mlir.sh). 
When updating ONNX-MLIR, it is good practice to check that the commit string of the MLIR/LLVM is the same as the one listed in that file.

### Build on Linux or OSX

Directions to install MLIR and ONNX-MLIR are provided [here](docs/BuildOnLinuxOSX.md).

### Build on Windows

Directions to install Protobuf, MLIR, and ONNX-MLIR are provided [here](docs/BuildOnWindows.md).

### Testing build and summary of custom envrionment variables

After installation, an `onnx-mlir` executable should appear in the `build/Debug/bin` or `build/Release/bin` directory.

There are several cmake targets that are used to verify the validity of the `onnx-mlir` compiler, which are listed [here](docs/TestingHighLevel.md).

## Using ONNX-MLIR

The usage of `onnx-mlir` is as such:

```
OVERVIEW: ONNX-MLIR modular optimizer driver

USAGE: onnx-mlir [options] <input file>

OPTIONS:

Generic Options:

  --help        - Display available options (--help-hidden for more)
  --help-list   - Display list of available options (--help-list-hidden for more)
  --version     - Display the version of this program

ONNX-MLIR Options:
These are frontend options.

  Choose target to emit:
      --EmitONNXBasic - Ingest ONNX and emit the basic ONNX operations without inferred shapes.
      --EmitONNXIR   - Ingest ONNX and emit corresponding ONNX dialect.
      --EmitMLIR     - Lower input to MLIR built-in transformation dialect.
      --EmitLLVMIR   - Lower input to LLVM IR (LLVM MLIR dialect).
      --EmitLib      - Lower input to LLVM IR, emit LLVM bitcode,
                       compile and link it to a shared library (default).
      --EmitJNI      - Lower input to LLVM IR -> LLVM bitcode -> JNI shared library ->
                       jar.

  Optimization levels:
      --O0           - Optimization level 0 (default).
      --O1           - Optimization level 1.
      --O2           - Optimization level 2.
      --O3           - Optimization level 3.
```

The full list of options is given by the `--help` option. Note that just as most compilers, the default optimization level is `-O0`. 
We recommend using `-O3` for most applications.

### Simple Example

For example, use the following command to lower an ONNX model (e.g., add.onnx) to ONNX dialect:
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

An example based on the add operation is found [here](docs/doc_example), which build an ONNX model using a python script, and then provide a main program to load the model's value, compute, and print the models output.

### End to End Example

An end to end example is provided [here](docs/mnist_example/README.md), which train, compile, and execute a simple MNIST example using both the C++ or Python interface.

## Slack channel

We have a slack channel established under the Linux Foundation AI and Data Workspace, named `#onnx-mlir-discussion`. This channel can be used for asking quick questions related to this project. A direct link is [here](https://lfaifoundation.slack.com/archives/C01J4NAL4A2).

## Contributing

Want to contribute, consult this page for specific help on our project [here](CONTRIBUTING.md) or the docs sub-directory.
