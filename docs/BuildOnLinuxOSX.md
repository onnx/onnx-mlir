<!--- SPDX-License-Identifier: Apache-2.0 -->

# Installation of ONNX-MLIR on Linux / OSX

We provide here directions to install ONNX-MLIR on Linux and OSX.
On Mac, there are a couple of commands that are different.
These differences will be listed in the explanation below, when relevant. Installing ONNX-MLIR on Apple silicon is natively supported and it is recommended to use brew to manage prerequisites.


## MLIR

Firstly, install MLIR (as a part of LLVM-Project):

[same-as-file]: <> (utils/clone-mlir.sh)
``` bash
git clone -n https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX-MLIR.
cd llvm-project && git checkout 6cf7fe4a9a715bcdf3f4913753109e22dfc9940b && cd ..
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
   -DLLVM_ENABLE_RTTI=ON \
   -DLLVM_ENABLE_LIBEDIT=OFF

cmake --build . -- ${MAKEFLAGS}
cmake --build . --target check-mlir
```

## ONNX-MLIR (this project)

### Build

The `MLIR_DIR` cmake variable must be set before building onnx-mlir. It should point to the mlir cmake module inside an llvm-project build or install directory (e.g., llvm-project/build/lib/cmake/mlir).

This project uses lit ([LLVM's Integrated Tester](https://llvm.org/docs/CommandGuide/lit.html)) for unit tests. When running cmake, we can also specify the path to the lit tool from LLVM using the `LLVM_EXTERNAL_LIT` variable but it is not required as long as MLIR_DIR points to a build directory of llvm-project. If `MLIR_DIR` points to an install directory of llvm-project, `LLVM_EXTERNAL_LIT` is required.

To build ONNX-MLIR, use the following commands (maybe with additional `-DCMAKE_CXX_FLAGS` argument described [below](#enable-cpu-optimizations)):

[same-as-file]: <> ({"ref": "utils/install-onnx-mlir.sh", "skip-doc": 2})
```bash
git clone --recursive https://github.com/onnx/onnx-mlir.git

# MLIR_DIR must be set with cmake option now
MLIR_DIR=$(pwd)/llvm-project/build/lib/cmake/mlir
mkdir onnx-mlir/build && cd onnx-mlir/build
if [[ -z "$pythonLocation" ]]; then
  cmake -G Ninja \
        -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
        -DMLIR_DIR=${MLIR_DIR} \
        ..
else
  cmake -G Ninja \
        -DCMAKE_CXX_COMPILER=/usr/bin/c++ \
        -DPython3_ROOT_DIR=$pythonLocation \
        -DMLIR_DIR=${MLIR_DIR} \
        ..
fi
cmake --build .

# Run lit tests:
export LIT_OPTS=-v
cmake --build . --target check-onnx-lit
```

Since OSX Big Sur, add the `-DCMAKE_CXX_COMPILER=/usr/bin/c++` option to the above `cmake ..` command due to changes in default compilers.

The environment variable `$pythonLocation` may be used to specify the base directory of the Python compiler.

After the above commands succeed, an `onnx-mlir` executable should appear in the `Debug/bin` or `Release/bin` directory.

### Enable CPU Optimizations

To make the compiler run faster (without any affect on the generated code)
you can pass `-DCMAKE_CXX_FLAGS=-march=native` to the `cmake -G Ninja ..` build configuration step above to generate code that exploits all the features of your local CPU, at the expense of portability. Or you can enable a specific CPU feature, e.g. `-DCMAKE_CXX_FLAGS=-mf16c` to enable the F16C feature to enable native conversion between float16 and (32 bit) float. It is used in `src/Support/SmallFP.hpp`.

### Known MacOS Issues

There is a known issue when building onnx-mlir. If you see an error of this sorts:

``` shell
Cloning into '/home/user/onnx-mlir/build/src/Runtime/jni/jsoniter'...

[...]

make[2]: *** [src/Runtime/jni/CMakeFiles/jsoniter.dir/build.make:74: src/Runtime/jni/jsoniter/target/jsoniter-0.9.23.jar] Error 127
make[1]: *** [CMakeFiles/Makefile2:3349: src/Runtime/jni/CMakeFiles/jsoniter.dir/all] Error 2
make: *** [Makefile:146: all] Error 2.
```

The suggested workaround until jsoniter is fixed is as follows: install maven (e.g. `brew install maven`) and run `alias nproc="sysctl -n hw.logicalcpu"` in your shell.

### Trouble shooting build issues

Check this [page](TestingHighLevel.md) for helpful hints.
