# Note: LLVM_ENABLE_PROJECTS is being phased out for runtimes; use LLVM_ENABLE_RUNTIMES instead (e.g. for OpenMP).  
# Going forward, all LLVM/MLIR-dependent runtimes should be built with LLVM_ENABLE_RUNTIMES, not LLVM_ENABLE_PROJECTS. 
mkdir llvm-project/build
cd llvm-project/build

cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS="mlir;clang" \
   -DLLVM_ENABLE_RUNTIMES="openmp" \
   -DLLVM_TARGETS_TO_BUILD="host" \
   -DCMAKE_BUILD_TYPE=Release \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DLLVM_ENABLE_RTTI=ON \
   -DENABLE_LIBOMPTARGET=OFF \
   -DLLVM_ENABLE_LIBEDIT=OFF

cmake --build . -- ${MAKEFLAGS}
cmake --build . --target check-mlir
