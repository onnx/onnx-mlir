<!--- SPDX-License-Identifier: Apache-2.0 -->

# ONNX MLIR
The Open Neural Network Exchange implementation in MLIR.

[![CircleCI](https://circleci.com/gh/onnx/onnx-mlir/tree/master.svg?style=svg)](https://circleci.com/gh/onnx/onnx-mlir/tree/master)

## Prerequisites

```
gcc >= 6.4
libprotoc >= 3.11.0
cmake >= 3.15.4
```

## Installation on UNIX

#### MLIR
Firstly, install MLIR (as a part of LLVM-Project):

[same-as-file]: <> (utils/clone-mlir.sh)
``` bash
git clone https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX MLIR.
cd llvm-project && git checkout 15d1ee36720ff24323f55452ae3cfb63f318c3f3 && cd ..
```

[same-as-file]: <> (utils/build-mlir.sh)
``` bash
mkdir llvm-project/build
cd llvm-project/build
cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_TARGETS_TO_BUILD="host" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_ENABLE_RTTI=ON

cmake --build . --target -- ${MAKEFLAGS}
cmake --build . --target check-mlir
```

#### ONNX-MLIR (this project)
The following environment variables can be set before building onnx-mlir (or alternatively, they need to be passed as CMake variables):
- LLVM_PROJ_SRC should point to the llvm-project src directory (e.g., llvm-project/).
- LLVM_PROJ_BUILD should point to the llvm-project build directory (e.g., llvm-project/build).

To build ONNX-MLIR, use the following command:

[same-as-file]: <> ({"ref": "utils/install-onnx-mlir.sh", "skip-doc": 2})
```bash
git clone --recursive https://github.com/onnx/onnx-mlir.git

# Export environment variables pointing to LLVM-Projects.
export LLVM_PROJ_SRC=$(pwd)/llvm-project/
export LLVM_PROJ_BUILD=$(pwd)/llvm-project/build

mkdir onnx-mlir/build && cd onnx-mlir/build
cmake ..
cmake --build .

# Run FileCheck tests:
export LIT_OPTS=-v
cmake --build . --target check-onnx-lit
```

After the above commands succeed, an `onnx-mlir` executable should appear in the `bin` directory. 

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
cd llvm-project && git checkout 15d1ee36720ff24323f55452ae3cfb63f318c3f3 && cd ..
```

[same-as-file]: <> (utils/build-mlir.cmd)
```shell
set root_dir=%cd%
md llvm-project\build
cd llvm-project\build
call cmake %root_dir%\llvm-project\llvm -G "Visual Studio 16 2019" -A x64 -T host=x64 ^
   -DCMAKE_INSTALL_PREFIX="%root_dir%\llvm-project\build\install" ^
   -DLLVM_ENABLE_PROJECTS=mlir ^
   -DLLVM_BUILD_EXAMPLES=ON ^
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
- LLVM_PROJ_BUILD: Path to the build directory for LLVM (e.g. c:/repos/llvm-project/build)
- LLVM_PROJ_SRC: Path to the source directory for LLVM (e.g. c:/repos/llvm-project)

This project uses lit ([LLVM's Integrated Tester](http://llvm.org/docs/CommandGuide/lit.html)) for unit tests. When running CMake, we can also specify the path to the lit tool from LLVM using the LLVM_EXTERNAL_LIT define but it is not required as long as LLVM_PROJ_BUILD is set correctly.

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
   -DLLVM_PROJ_BUILD=%root_dir%\llvm-project\build ^
   -DLLVM_PROJ_SRC=%root_dir%\llvm-project

call cmake --build . --config Release --target onnx-mlir -- /m
```

To test ONNX MLIR, use the following command:

[same-as-file]: <> ({"ref": "utils/check-onnx-mlir.cmd", "skip-ref": 1})
```shell
call cmake --build . --config Release --target check-onnx-lit -- /m
```

After the above commands succeed, an `onnx-mlir` executable should appear in the `bin` directory.

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
      --EmitONNXIR - Ingest ONNX and emit corresponding ONNX dialect.
      --EmitMLIR   - Lower model to MLIR built-in transformation dialect.
      --EmitLLVMIR - Lower model to LLVM IR (LLVM dialect).
      --EmitLLVMBC - Lower model to LLVM IR and emit (to file) LLVM bitcode for model.
```

## Example

For example, to lower an ONNX model (e.g., add.onnx) to ONNX dialect, use the following command:
```
./onnx-mlir --EmitONNXIR add.onnx
```
The output should look like:
```
module {
  func @main_graph(%arg0: tensor<10x10x10xf32>, %arg1: tensor<10x10x10xf32>) -> tensor<10x10x10xf32> {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<10x10x10xf32>, tensor<10x10x10xf32>) -> tensor<10x10x10xf32>
    return %0 : tensor<10x10x10xf32>
  }
}
```

## Troubleshooting

If the latest LLVM project fails to work due to the latest changes to the MLIR subproject please consider using a slightly older version of LLVM. One such version, which we use, can be found [here](https://github.com/clang-ykt/llvm-project).
