git clone https://github.com/llvm/llvm-project.git
# Check out a specific branch that is known to work with ONNX MLIR.
cd llvm-project && git checkout 076475713c236081a3247a53e9dbab9043c3eac2 && cd ..
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