git clone https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNF.
cd llvm-project && git checkout 391cc4dd41db934081c37ee523cb2149bf0e3a41 && cd ..
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