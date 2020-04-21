set root_dir=%cd%
md llvm-project\build
cd llvm-project\build
call cmake -G "Visual Studio 16 2019" -A x64 -T host=x64 ..\llvm ^
   -DCMAKE_INSTALL_PREFIX="%root_dir%\llvm-project\build\install" ^
   -DLLVM_ENABLE_PROJECTS=mlir ^
   -DLLVM_BUILD_EXAMPLES=ON ^
   -DLLVM_TARGETS_TO_BUILD="host" ^
   -DCMAKE_BUILD_TYPE=Release ^
   -DLLVM_ENABLE_ASSERTIONS=ON ^
   -DLLVM_ENABLE_RTTI=ON ^
   -DLLVM_ENABLE_ZLIB=OFF

call cmake --build . --config Release --target -- /m
call cmake --build . --config Release --target install
call cmake --build . --config Release --target check-mlir
