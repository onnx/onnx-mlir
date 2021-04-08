REM Export environment variables pointing to LLVM-Projects.
set root_dir=%cd%
set LLVM_PROJ_BUILD=%root_dir%/llvm-project/build
set LLVM_PROJ_SRC=%root_dir%/llvm-project

md onnx-mlir\build
cd onnx-mlir\build
call cmake -G "Visual Studio 16 2019" -A x64 -T host=x64 -DCMAKE_BUILD_TYPE=Release ..
call cmake --build . --config Release --target onnx-mlir -- /m
