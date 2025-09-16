:: Note: LLVM_ENABLE_PROJECTS is being phased out for runtimes; use LLVM_ENABLE_RUNTIMES instead (e.g. for OpenMP).  
:: Going forward, all LLVM/MLIR-dependent runtimes should be built with LLVM_ENABLE_RUNTIMES, not LLVM_ENABLE_PROJECTS. 
set root_dir=%cd%
md llvm-project\build
cd llvm-project\build

call cmake %root_dir%\llvm-project\llvm -G "Ninja" ^
   -DCMAKE_INSTALL_PREFIX="%root_dir%\llvm-project\build\install" ^
   -DLLVM_ENABLE_PROJECTS="mlir;clang" ^
   -DLLVM_ENABLE_RUNTIMES="openmp" ^
   -DLLVM_TARGETS_TO_BUILD="host" ^
   -DCMAKE_BUILD_TYPE=Release ^
   -DLLVM_ENABLE_ASSERTIONS=ON ^
   -DLLVM_ENABLE_RTTI=ON ^
   -DLLVM_ENABLE_ZLIB=OFF ^
   -DLLVM_INSTALL_UTILS=ON ^
   -DENABLE_LIBOMPTARGET=OFF ^
   -DLLVM_ENABLE_LIBEDIT=OFF

call cmake --build . --config Release
call cmake --build . --config Release --target install
call cmake --build . --config Release --target check-mlir
