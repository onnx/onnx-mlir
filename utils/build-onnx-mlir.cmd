set root_dir=%cd%

md onnx-mlir\build
cd onnx-mlir\build
call cmake %root_dir%\onnx-mlir -G "Ninja" ^
   -DCMAKE_BUILD_TYPE=Release ^
   -DCMAKE_PREFIX_PATH=%root_dir%\protobuf_install ^
   -DLLVM_LIT_ARGS=-v ^
   -DMLIR_DIR=%root_dir%\llvm-project\build\lib\cmake\mlir

call cmake --build . --config Release --target onnx-mlir
