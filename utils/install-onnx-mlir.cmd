git clone --recursive https://github.com/onnx/onnx-mlir.git

REM Export environment variables pointing to LLVM-Projects.
set root_dir=%cd%
set CURSES_LIB_PATH=%cd%/PDCurses/pdcurses.lib
set LLVM_PROJ_BUILD=%cd%/llvm-project/build
set LLVM_PROJ_SRC=%cd%/llvm-project

md onnx-mlir/build && cd onnx-mlir/build
cmake -G "Visual Studio 16 2019" -A x64 -T host=x64 -DLLVM_EXTERNAL_LIT=%root_dir%/llvm-project/build/Release/bin/llvm-lit.py -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release --target onnx-mlir

REM Run FileCheck tests
set LIT_OPTS=-v
cmake --build . --target check-onnx-lit